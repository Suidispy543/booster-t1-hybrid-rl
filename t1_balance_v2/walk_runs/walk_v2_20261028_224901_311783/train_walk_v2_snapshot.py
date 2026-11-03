import argparse
import json
from datetime import datetime
from pathlib import Path
import shutil

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from balance_env import BalanceEnv as BaseBalanceEnv


ROOT = Path(__file__).resolve().parents[1]

START_CHECKPOINT = (
    ROOT / "t1_balance_v2" / "runs"
    / "stance_alpha0.5_20261028_215029_608658"
    / "checkpoint_000025600.zip"
)


class WalkEnvV2(BaseBalanceEnv):
    """Walking curriculum using the original T1 model and standing policy."""

    def __init__(
        self,
        episode_seconds=20.0,
        reset_noise=0.002,
        target_speed=0.15,
        gait_frequency=1.0,
    ):
        self.phase = 0.0
        self.target_speed = float(target_speed)
        self.gait_frequency = float(gait_frequency)
        self.foot_home_z = np.zeros(2, dtype=np.float64)

        super().__init__(
            episode_seconds=episode_seconds,
            alpha=0.5,
            reset_noise=reset_noise,
            stance_weight=0.3,
            drift_weight=0.0,
            rate_weight=0.08,
            joint_velocity_weight=0.05,
        )

        # Allow larger leg movements than the standing controller.
        self.action_scale = 0.35

    def _observation(self):
        obs = super()._observation().copy()

        # Preserve the 72-input policy interface.
        # Use the two former displacement features as a gait clock.
        obs[57] = np.sin(self.phase)
        obs[58] = np.cos(self.phase)

        return obs.astype(np.float32)

    def _gait_action(self):
        """
        Mild alternating-leg reference.

        Action ordering:
        [L hip pitch, roll, yaw, knee, ankle pitch, roll,
         R hip pitch, roll, yaw, knee, ankle pitch, roll]
        """
        s = float(np.sin(self.phase))
        left_swing = max(0.0, s)
        right_swing = max(0.0, -s)

        gait = np.zeros(12, dtype=np.float32)

        # Opposite hip-pitch movement.
        gait[0] = 0.50 * s
        gait[6] = -0.50 * s

        # Flex the knee of the leg in its swing phase.
        gait[3] = 0.55 * left_swing
        gait[9] = 0.55 * right_swing

        # Small ankle compensation.
        gait[4] = -0.20 * s - 0.10 * left_swing
        gait[10] = 0.20 * s - 0.10 * right_swing

        return gait

    def reset(self, *, seed=None, options=None):
        self.phase = 0.0
        obs, info = super().reset(seed=seed, options=options)

        self.foot_home_z = np.asarray(
            [self.data.xpos[body_id, 2] for body_id in self.feet],
            dtype=np.float64,
        )

        return self._observation(), info

    def _foot_contact(self, foot_body_id):
        """Detect contact between a foot and the world ground."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])

            if (
                body1 == foot_body_id and body2 == 0
            ) or (
                body2 == foot_body_id and body1 == 0
            ):
                return True

        return False

    def step(self, action):
        previous_action = self.previous_action.copy()
        phase_used = self.phase

        # PPO learns a correction to the reference gait.
        policy_action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(12),
            -1.0,
            1.0,
        )
        combined_action = np.clip(
            policy_action + self._gait_action(),
            -1.0,
            1.0,
        )

        _, _, terminated, truncated, info = super().step(
            combined_action
        )

        applied_action = self.previous_action.copy()

        # MuJoCo free-joint qvel uses global linear velocity in [0:3]
        # and angular velocity in [3:6].
        linear_velocity = self.data.qvel[:3]
        angular_velocity = self.data.qvel[3:6]

        forward_speed = float(linear_velocity[0])
        lateral_speed = float(linear_velocity[1])
        vertical_speed = float(linear_velocity[2])

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

        upright = float(rotation[2, 2])
        height = float(self.data.qpos[2])

        height_score = float(
            np.clip(
                1.0 - abs(height - self.home_height) / 0.25,
                -1.0,
                1.0,
            )
        )

        # Reward the commanded speed, while preventing runaway motion.
        velocity_reward = 5.0 * np.exp(
            -((forward_speed - self.target_speed) / 0.12) ** 2
        )
        progress_reward = 0.7 * float(
            np.clip(forward_speed, -0.20, 0.30)
        )

        # Alternate foot swing and stance phases.
        left_contact = self._foot_contact(self.feet[0])
        right_contact = self._foot_contact(self.feet[1])

        left_lift = float(
            self.data.xpos[self.feet[0], 2] - self.foot_home_z[0]
        )
        right_lift = float(
            self.data.xpos[self.feet[1], 2] - self.foot_home_z[1]
        )

        if np.sin(phase_used) >= 0.0:
            # Left leg swings; right leg supports.
            contact_reward = (
                0.20 * float(right_contact)
                - 0.20 * float(left_contact)
            )
            swing_reward = 0.35 * np.exp(
                -((left_lift - 0.04) / 0.025) ** 2
            )
        else:
            # Right leg swings; left leg supports.
            contact_reward = (
                0.20 * float(left_contact)
                - 0.20 * float(right_contact)
            )
            swing_reward = 0.35 * np.exp(
                -((right_lift - 0.04) / 0.025) ** 2
            )

        # Keep the torso reasonably upright, without overwhelming gait
        # learning with a very strong standing-only tilt penalty.
        tilt_cost = (
            2.0 * (roll**2 + pitch**2)
            + 0.15 * (abs(roll) + abs(pitch))
        )

        joint_velocity_cost = 0.015 * float(
            np.mean(self.data.qvel[self.leg_dofs] ** 2)
        )
        angular_cost = 0.08 * float(np.sum(angular_velocity**2))
        lateral_cost = 0.8 * lateral_speed**2
        vertical_cost = 0.4 * vertical_speed**2
        action_cost = 0.003 * float(np.mean(combined_action**2))
        smooth_cost = 0.03 * float(
            np.mean((applied_action - previous_action) ** 2)
        )

        width_error = float(info.get("foot_width_m", self.home_width)) - self.home_width
        width_cost = 0.05 * min((width_error / 0.10) ** 2, 9.0)

        reward = (
            velocity_reward
            + progress_reward
            + 0.8 * upright
            + 0.8 * height_score
            + contact_reward
            + swing_reward
            - tilt_cost
            - joint_velocity_cost
            - angular_cost
            - lateral_cost
            - vertical_cost
            - action_cost
            - smooth_cost
            - width_cost
        )

        if terminated:
            reward -= 12.0

        # Advance gait phase and give PPO the next phase observation.
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
                    np.arccos(np.clip(upright, -1.0, 1.0))
                )
            ),
            velocity_reward=float(velocity_reward),
            contact_reward=float(contact_reward),
            swing_reward=float(swing_reward),
        )

        return self._observation(), float(reward), terminated, truncated, info


def evaluate(model, target_speed, gait_frequency, seed=42, seconds=20.0):
    env = WalkEnvV2(
        episode_seconds=seconds,
        reset_noise=0.0,
        target_speed=target_speed,
        gait_frequency=gait_frequency,
    )

    try:
        obs, _ = env.reset(seed=seed)
        start_x = float(env.data.qpos[0])
        start_y = float(env.data.qpos[1])

        speeds = []
        tilts = []
        steps = 0
        fell = False

        max_steps = int(
            seconds / (env.model.opt.timestep * env.frame_skip)
        )

        for i in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            if i >= 50:
                speeds.append(info["forward_speed_m_s"])
                tilts.append(info["root_tilt_deg"])

            steps += 1

            if terminated or truncated:
                fell = bool(terminated)
                break

        return {
            "duration_s": round(
                steps * env.model.opt.timestep * env.frame_skip, 2
            ),
            "fell": fell,
            "forward_displacement_m": round(
                float(env.data.qpos[0]) - start_x, 4
            ),
            "lateral_displacement_m": round(
                float(env.data.qpos[1]) - start_y, 4
            ),
            "mean_forward_speed_m_s": (
                round(float(np.mean(speeds)), 4) if speeds else None
            ),
            "mean_tilt_deg": (
                round(float(np.mean(tilts)), 3) if tilts else None
            ),
            "max_tilt_deg": (
                round(float(np.max(tilts)), 3) if tilts else None
            ),
            "target_speed_m_s": target_speed,
            "gait_frequency_hz": gait_frequency,
        }

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--target-speed", type=float, default=0.15)
    parser.add_argument("--gait-frequency", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--exploration-std", type=float, default=0.25)
    args = parser.parse_args()

    if not START_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Standing checkpoint not found:\n{START_CHECKPOINT}"
        )
    if args.steps <= 0 or args.eval_every <= 0:
        raise ValueError("--steps and --eval-every must be positive.")
    if args.target_speed <= 0 or args.gait_frequency <= 0:
        raise ValueError("Target speed and gait frequency must be positive.")

    test_env = WalkEnvV2(
        episode_seconds=5.0,
        reset_noise=0.0,
        target_speed=args.target_speed,
        gait_frequency=args.gait_frequency,
    )
    try:
        check_env(test_env, warn=True)
        print("Walking environment check passed.")
        print("Observation space:", test_env.observation_space)
        print("Action space:", test_env.action_space)
        print("Model:", test_env.model.nq, "qpos,", test_env.model.nu, "actuators")
    finally:
        test_env.close()

    out_dir = (
        ROOT / "t1_balance_v2" / "walk_runs"
        / f"walk_v2_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__).resolve(), out_dir / "train_walk_v2_snapshot.py")

    settings = {
        "starting_checkpoint": str(START_CHECKPOINT),
        "steps_requested": args.steps,
        "eval_every": args.eval_every,
        "target_speed_m_s": args.target_speed,
        "gait_frequency_hz": args.gait_frequency,
        "learning_rate": args.learning_rate,
        "exploration_std": args.exploration_std,
        "action_scale": 0.35,
        "approach": "phase-conditioned gait prior with PPO residual learning",
        "model": "original Google DeepMind MuJoCo Menagerie T1 scene.xml",
    }
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    train_env = Monitor(
        WalkEnvV2(
            episode_seconds=20.0,
            reset_noise=0.002,
            target_speed=args.target_speed,
            gait_frequency=args.gait_frequency,
        )
    )

    try:
        model = PPO.load(
            str(START_CHECKPOINT),
            env=train_env,
            device="cpu",
            learning_rate=args.learning_rate,
        )

        # Increase exploration in the training copy, not the source model.
        with torch.no_grad():
            model.policy.log_std.fill_(float(np.log(args.exploration_std)))
        model.policy.optimizer.state.clear()

        initial_steps = int(model.num_timesteps)
        model.save(str(out_dir / "initial_policy"))

        before = evaluate(
            model, args.target_speed, args.gait_frequency
        )
        print("\nEvaluation before PPO fine-tuning:")
        print(json.dumps(before, indent=2))

        records = []
        added = 0

        while added < args.steps:
            chunk = min(args.eval_every, args.steps - added)
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            added = int(model.num_timesteps) - initial_steps

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
        print("Results:", out_dir)

    finally:
        train_env.close()


if __name__ == "__main__":
    main()
