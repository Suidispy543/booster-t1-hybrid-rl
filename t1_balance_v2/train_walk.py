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

# Preserve and fine-tune the best stable standing model.
START_CHECKPOINT = (
    ROOT / "t1_balance_v2" / "runs"
    / "stance_alpha0.5_20261028_215029_608658"
    / "checkpoint_000025600.zip"
)

# Target is a slow initial walk, not running.
DEFAULT_TARGET_SPEED = 0.20

# Explicit torso orientation penalty, with angles in radians.
TILT_WEIGHT = 50.0
TILT_LINEAR_WEIGHT = 6.0


class WalkEnv(BaseBalanceEnv):
    """
    Walking fine-tuning environment.

    Keeps the existing 72-dimensional observation and 12-action interface.
    Root XY displacement inputs are set to zero because walking is allowed
    to travel away from the starting point.
    """

    def __init__(
        self,
        episode_seconds=20.0,
        reset_noise=0.002,
        target_speed=DEFAULT_TARGET_SPEED,
    ):
        super().__init__(
            episode_seconds=episode_seconds,
            alpha=0.5,
            reset_noise=reset_noise,
            stance_weight=0.5,
            drift_weight=0.0,
            rate_weight=0.08,
            joint_velocity_weight=0.05,
        )
        self.target_speed = float(target_speed)

    def _observation(self):
        obs = super()._observation().copy()

        # BalanceEnv's observation layout:
        # 0:57  original robot state
        # 57:59 root XY displacement from start
        # 59:71 previous applied leg actions
        # 71    foot-width error
        #
        # Remove absolute displacement from the policy input, because
        # walking should not be penalized for travelling away from origin.
        if obs.shape == (72,):
            obs[57:59] = 0.0

        return obs.astype(np.float32)

    def step(self, action):
        obs, old_reward, terminated, truncated, info = super().step(action)

        # Rebuild the balance reward without the displacement-from-origin
        # penalty. Keep the original posture, height, action, stance,
        # joint-velocity and smoothness terms.
        original_reward = float(info["original_reward"])
        costs = info.get("reward_costs", {})
        retained_costs = sum(
            float(value)
            for key, value in costs.items()
            if key != "drift"
        )

        reward = original_reward - retained_costs

        # Root torso orientation.
        R = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        R = R.reshape(3, 3)

        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(
            -R[2, 0],
            np.hypot(R[2, 1], R[2, 2]),
        )

        tilt_cost = (
            TILT_WEIGHT * (roll**2 + pitch**2)
            + TILT_LINEAR_WEIGHT * (abs(roll) + abs(pitch))
        )
        reward -= float(tilt_cost)

        # T1 forward direction is assumed to be world +X.
        forward_speed = float(self.data.qvel[0])
        lateral_speed = float(self.data.qvel[1])

        # Reward reaching the commanded walking speed.
        speed_error = forward_speed - self.target_speed
        velocity_reward = 3.0 * np.exp(
            -((speed_error / 0.15) ** 2)
        )

        # A smaller forward-progress bonus helps the policy discover
        # movement while the tracking reward discourages excessive speed.
        progress_reward = 0.5 * float(
            np.clip(forward_speed, -0.2, 0.4)
        )

        # Avoid solving the task by drifting sideways.
        lateral_cost = 0.5 * lateral_speed**2

        reward += float(velocity_reward + progress_reward - lateral_cost)

        info.update(
            walk_forward_speed_m_s=forward_speed,
            walk_lateral_speed_m_s=lateral_speed,
            walk_velocity_reward=float(velocity_reward),
            walk_progress_reward=float(progress_reward),
            walk_tilt_cost=float(tilt_cost),
            root_roll_deg=float(np.degrees(roll)),
            root_pitch_deg=float(np.degrees(pitch)),
            root_tilt_deg=float(
                np.degrees(
                    np.arccos(np.clip(R[2, 2], -1.0, 1.0))
                )
            ),
        )

        return obs, float(reward), terminated, truncated, info


def evaluate(model, target_speed, seed=42, seconds=20.0):
    """Evaluate a walking checkpoint without training it."""
    env = WalkEnv(
        episode_seconds=seconds,
        reset_noise=0.0,
        target_speed=target_speed,
    )

    try:
        obs, _ = env.reset(seed=seed)
        start_x = float(env.data.qpos[0])
        start_y = float(env.data.qpos[1])

        speeds = []
        tilts = []
        falls = False
        steps = 0

        for i in range(int(seconds / (
            env.model.opt.timestep * env.frame_skip
        ))):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            if i >= 50:  # Exclude the first second.
                speeds.append(info["walk_forward_speed_m_s"])
                tilts.append(info["root_tilt_deg"])

            steps += 1

            if terminated or truncated:
                falls = bool(terminated)
                break

        return {
            "duration_s": round(
                steps * env.model.opt.timestep * env.frame_skip, 2
            ),
            "fell": falls,
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
            "maximum_tilt_deg": (
                round(float(np.max(tilts)), 3) if tilts else None
            ),
            "target_speed_m_s": target_speed,
        }

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--target-speed", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--exploration-std", type=float, default=0.18)
    args = parser.parse_args()

    if not START_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Standing checkpoint not found:\n{START_CHECKPOINT}"
        )
    if args.steps <= 0 or args.eval_every <= 0:
        raise ValueError("--steps and --eval-every must be positive.")
    if args.target_speed <= 0:
        raise ValueError("--target-speed must be positive.")

    # Validate the new environment before training.
    test_env = WalkEnv(
        episode_seconds=5.0,
        reset_noise=0.0,
        target_speed=args.target_speed,
    )
    try:
        check_env(test_env, warn=True)
        print("Walking environment check passed.")
        print("Observation space:", test_env.observation_space)
        print("Action space:", test_env.action_space)
    finally:
        test_env.close()

    out_dir = (
        ROOT / "t1_balance_v2" / "walk_runs"
        / f"walk_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(
        Path(__file__).resolve(),
        out_dir / "train_walk_snapshot.py",
    )

    settings = {
        "starting_checkpoint": str(START_CHECKPOINT),
        "steps_requested": args.steps,
        "evaluation_interval": args.eval_every,
        "target_speed_m_s": args.target_speed,
        "learning_rate": args.learning_rate,
        "exploration_std": args.exploration_std,
        "tilt_weight": TILT_WEIGHT,
        "tilt_linear_weight": TILT_LINEAR_WEIGHT,
        "forward_axis": "+X",
        "observation_size": 72,
        "action_size": 12,
    }
    (out_dir / "settings.json").write_text(
        json.dumps(settings, indent=2)
    )

    train_env = Monitor(
        WalkEnv(
            episode_seconds=20.0,
            reset_noise=0.002,
            target_speed=args.target_speed,
        )
    )

    model = None

    try:
        # Load the existing PPO policy; do not modify the standing file.
        model = PPO.load(
            str(START_CHECKPOINT),
            env=train_env,
            device="cpu",
            learning_rate=args.learning_rate,
        )

        # Re-open a modest amount of exploration for learning the gait.
        # This changes the training copy, not the source checkpoint.
        with torch.no_grad():
            model.policy.log_std.fill_(
                float(np.log(args.exploration_std))
            )

        # Reset optimizer moments for the new locomotion objective.
        model.policy.optimizer.state.clear()

        initial_model_steps = int(model.num_timesteps)
        model.save(str(out_dir / "walking_fork_initial.zip"))

        baseline = evaluate(
            model,
            target_speed=args.target_speed,
            seed=42,
        )
        print("\nEvaluation before walking fine-tuning:")
        print(json.dumps(baseline, indent=2))

        records = []
        (out_dir / "evaluations.json").write_text("[]")

        while model.num_timesteps - initial_model_steps < args.steps:
            already_trained = model.num_timesteps - initial_model_steps
            remaining = args.steps - already_trained
            chunk = min(args.eval_every, remaining)

            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                progress_bar=False,
            )

            actual_added_steps = (
                model.num_timesteps - initial_model_steps
            )
            checkpoint = (
                out_dir
                / f"checkpoint_after_{actual_added_steps:06d}.zip"
            )
            model.save(str(checkpoint))

            result = evaluate(
                model,
                target_speed=args.target_speed,
                seed=42,
            )

            record = {
                "training_steps_added": int(actual_added_steps),
                "total_model_steps": int(model.num_timesteps),
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
