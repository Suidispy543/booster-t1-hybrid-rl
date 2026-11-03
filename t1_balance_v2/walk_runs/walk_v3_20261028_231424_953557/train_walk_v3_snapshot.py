import argparse
import json
from datetime import datetime
from pathlib import Path
import shutil

import mujoco
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from balance_env import BalanceEnv


ROOT = Path(__file__).resolve().parents[1]

START_CHECKPOINT = (
    ROOT / "t1_balance_v2" / "runs"
    / "stance_alpha0.5_20261028_215029_608658"
    / "checkpoint_000025600.zip"
)


class WalkEnvV3(BalanceEnv):
    """Phase-conditioned locomotion environment using the original T1."""

    def __init__(
        self,
        episode_seconds=20.0,
        reset_noise=0.002,
        target_speed=0.10,
        gait_frequency=0.8,
    ):
        self.phase = 0.0
        self.target_speed = float(target_speed)
        self.gait_frequency = float(gait_frequency)
        self.foot_home_z = np.zeros(2, dtype=np.float64)

        super().__init__(
            episode_seconds=episode_seconds,
            reset_noise=reset_noise,
            alpha=0.5,
            stance_weight=0.3,
            drift_weight=0.0,
            rate_weight=0.08,
            joint_velocity_weight=0.05,
        )

        # The original 72 features remain in place. The two added inputs
        # encode gait phase without changing the trained policy's action size.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(74,),
            dtype=np.float32,
        )

    def _observation(self):
        obs = super()._observation().copy()

        # Absolute displacement from the starting position should not
        # encourage the policy to remain near its initial location.
        obs[57:59] = 0.0

        phase_features = np.array(
            [np.sin(self.phase), np.cos(self.phase)],
            dtype=np.float32,
        )

        return np.concatenate(
            [obs, phase_features]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.phase = 0.0

        obs, info = super().reset(seed=seed, options=options)

        self.foot_home_z = np.array(
            [self.data.xpos[body, 2] for body in self.feet],
            dtype=np.float64,
        )

        return self._observation(), info

    def _foot_contact(self, foot_body_id):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body_a = int(self.model.geom_bodyid[contact.geom1])
            body_b = int(self.model.geom_bodyid[contact.geom2])

            if (
                body_a == foot_body_id and body_b == 0
            ) or (
                body_b == foot_body_id and body_a == 0
            ):
                return True

        return False

    def step(self, action):
        phase_used = self.phase

        _, _, terminated, truncated, info = super().step(action)

        # Remove only the penalty on travelling away from the origin.
        # Preserve the underlying balance, stance, action and velocity terms.
        original_reward = float(info["original_reward"])
        costs = info.get("reward_costs", {})
        retained_cost = sum(
            float(value)
            for key, value in costs.items()
            if key != "drift"
        )
        reward = original_reward - retained_cost

        rotation = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        rotation = rotation.reshape(3, 3)

        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(
            np.arctan2(
                -rotation[2, 0],
                np.hypot(rotation[2, 1], rotation[2, 2]),
            )
        )

        forward_speed = float(self.data.qvel[0])
        lateral_speed = float(self.data.qvel[1])

        # A strong speed-tracking reward makes forward walking preferable
        # to standing still, while discouraging excessive speed.
        speed_reward = 12.0 * np.exp(
            -((forward_speed - self.target_speed) / 0.12) ** 2
        )
        progress_reward = 1.2 * float(
            np.clip(forward_speed, -0.20, 0.30)
        )

        left_lift = float(
            self.data.xpos[self.feet[0], 2] - self.foot_home_z[0]
        )
        right_lift = float(
            self.data.xpos[self.feet[1], 2] - self.foot_home_z[1]
        )

        left_contact = self._foot_contact(self.feet[0])
        right_contact = self._foot_contact(self.feet[1])

        # Alternate the intended swing leg using the explicit phase input.
        if np.sin(phase_used) >= 0.0:
            swing_lift = left_lift
            swing_contact = left_contact
            support_contact = right_contact
        else:
            swing_lift = right_lift
            swing_contact = right_contact
            support_contact = left_contact

        swing_reward = 0.6 * np.exp(
            -((swing_lift - 0.035) / 0.025) ** 2
        )
        contact_reward = (
            0.25 * float(support_contact)
            - 0.10 * float(swing_contact)
        )

        # Penalize torso lean, sideways drift, and excessive joint motion.
        tilt_cost = (
            15.0 * (roll**2 + pitch**2)
            + 0.5 * (abs(roll) + abs(pitch))
        )
        lateral_cost = 0.8 * lateral_speed**2
        joint_velocity_cost = 0.03 * float(
            np.mean(self.data.qvel[self.leg_dofs] ** 2)
        )

        reward += (
            speed_reward
            + progress_reward
            + swing_reward
            + contact_reward
            - tilt_cost
            - lateral_cost
            - joint_velocity_cost
        )

        if terminated:
            reward -= 15.0

        # Advance phase after simulating the action.
        decision_dt = self.model.opt.timestep * self.frame_skip
        self.phase = (
            self.phase
            + 2.0 * np.pi * self.gait_frequency * decision_dt
        ) % (2.0 * np.pi)

        info.update(
            forward_speed_m_s=forward_speed,
            lateral_speed_m_s=lateral_speed,
            root_roll_deg=float(np.degrees(roll)),
            root_pitch_deg=float(np.degrees(pitch)),
            root_tilt_deg=float(
                np.degrees(
                    np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))
                )
            ),
            speed_reward=float(speed_reward),
            swing_reward=float(swing_reward),
        )

        return self._observation(), float(reward), terminated, truncated, info


def initialize_policy(env, learning_rate, exploration_std):
    """Transfer compatible standing weights into the 74-input policy."""
    standing = PPO.load(str(START_CHECKPOINT), device="cpu")

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [128, 128]},
        learning_rate=learning_rate,
        n_steps=1024,
        batch_size=256,
        n_epochs=5,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.0,
        clip_range=0.1,
        target_kl=0.03,
        device="cpu",
        seed=42,
        verbose=0,
    )

    src = standing.policy.state_dict()
    dst = model.policy.state_dict()

    copied = 0
    expanded = 0

    with torch.no_grad():
        for name, target_tensor in dst.items():
            if name not in src:
                continue

            source_tensor = src[name]

            if target_tensor.shape == source_tensor.shape:
                target_tensor.copy_(source_tensor)
                copied += 1

            elif (
                target_tensor.ndim == 2
                and source_tensor.ndim == 2
                and target_tensor.shape[0] == source_tensor.shape[0]
                and target_tensor.shape[1] == source_tensor.shape[1] + 2
            ):
                target_tensor.zero_()
                target_tensor[:, :source_tensor.shape[1]].copy_(
                    source_tensor
                )
                expanded += 1

    model.policy.load_state_dict(dst)

    with torch.no_grad():
        model.policy.log_std.fill_(float(np.log(exploration_std)))

    if expanded == 0:
        raise RuntimeError(
            "Could not find the expanded 72-to-74 input layer. "
            "The policy architecture may differ from the standing checkpoint."
        )

    print(f"Transferred {copied} compatible parameter tensors.")
    print(f"Expanded {expanded} input layer(s) from 72 to 74 features.")
    print("The two new gait-phase input weights start at zero.")

    return model


def evaluate(model, target_speed, gait_frequency, seconds=20.0):
    env = WalkEnvV3(
        episode_seconds=seconds,
        reset_noise=0.0,
        target_speed=target_speed,
        gait_frequency=gait_frequency,
    )

    try:
        obs, _ = env.reset(seed=42)
        start_xy = env.data.qpos[:2].copy()

        speeds = []
        lateral_speeds = []
        tilts = []
        fell = False
        steps = 0

        max_steps = int(
            seconds / (env.model.opt.timestep * env.frame_skip)
        )

        for i in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            if i >= 50:
                speeds.append(info["forward_speed_m_s"])
                lateral_speeds.append(info["lateral_speed_m_s"])
                tilts.append(info["root_tilt_deg"])

            steps += 1

            if terminated or truncated:
                fell = bool(terminated)
                break

        displacement = env.data.qpos[:2] - start_xy

        return {
            "duration_s": round(
                steps * env.model.opt.timestep * env.frame_skip, 2
            ),
            "fell": fell,
            "forward_displacement_m": round(float(displacement[0]), 4),
            "lateral_displacement_m": round(float(displacement[1]), 4),
            "mean_forward_speed_m_s": (
                round(float(np.mean(speeds)), 4) if speeds else None
            ),
            "mean_lateral_speed_m_s": (
                round(float(np.mean(lateral_speeds)), 4)
                if lateral_speeds else None
            ),
            "mean_tilt_deg": (
                round(float(np.mean(tilts)), 3) if tilts else None
            ),
            "max_tilt_deg": (
                round(float(np.max(tilts)), 3) if tilts else None
            ),
            "target_speed_m_s": target_speed,
        }

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--target-speed", type=float, default=0.10)
    parser.add_argument("--gait-frequency", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--exploration-std", type=float, default=0.25)
    args = parser.parse_args()

    if not START_CHECKPOINT.is_file():
        raise FileNotFoundError(START_CHECKPOINT)
    if args.steps <= 0 or args.eval_every <= 0:
        raise ValueError("--steps and --eval-every must be positive.")

    validation_env = WalkEnvV3(
        episode_seconds=5.0,
        reset_noise=0.0,
        target_speed=args.target_speed,
        gait_frequency=args.gait_frequency,
    )
    try:
        check_env(validation_env, warn=True)
        print("Walking environment check passed.")
        print("Observation space:", validation_env.observation_space)
        print("Action space:", validation_env.action_space)
        print("Actuators:", validation_env.model.nu)
    finally:
        validation_env.close()

    out_dir = (
        ROOT / "t1_balance_v2" / "walk_runs"
        / f"walk_v3_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(
        Path(__file__).resolve(),
        out_dir / "train_walk_v3_snapshot.py",
    )

    settings = {
        "starting_checkpoint": str(START_CHECKPOINT),
        "steps_requested": args.steps,
        "eval_every": args.eval_every,
        "target_speed_m_s": args.target_speed,
        "gait_frequency_hz": args.gait_frequency,
        "learning_rate": args.learning_rate,
        "exploration_std": args.exploration_std,
        "observation_size": 74,
        "action_size": 12,
        "approach": "PPO fine-tuning with explicit gait-phase observations",
    }
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    train_env = Monitor(
        WalkEnvV3(
            episode_seconds=20.0,
            reset_noise=0.002,
            target_speed=args.target_speed,
            gait_frequency=args.gait_frequency,
        )
    )

    try:
        model = initialize_policy(
            train_env,
            learning_rate=args.learning_rate,
            exploration_std=args.exploration_std,
        )

        before = evaluate(
            model, args.target_speed, args.gait_frequency
        )
        print("\nEvaluation before walking training:")
        print(json.dumps(before, indent=2))

        model.save(str(out_dir / "initial_policy"))
        records = []
        added = 0

        while added < args.steps:
            chunk = min(args.eval_every, args.steps - added)

            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                progress_bar=False,
            )

            added = int(model.num_timesteps)

            checkpoint = out_dir / f"checkpoint_after_{added:06d}.zip"
            model.save(str(checkpoint))

            result = evaluate(
                model, args.target_speed, args.gait_frequency
            )
            record = {
                "training_steps_added": added,
                "checkpoint": str(checkpoint),
                "evaluation": result,
            }
            records.append(record)

            (out_dir / "evaluations.json").write_text(
                json.dumps(records, indent=2)
            )

            print("\nTraining/evaluation result:")
            print(json.dumps(record, indent=2))

        print("\nWalking experiment complete.")
        print("Results saved to:", out_dir)

    finally:
        train_env.close()


if __name__ == "__main__":
    main()
