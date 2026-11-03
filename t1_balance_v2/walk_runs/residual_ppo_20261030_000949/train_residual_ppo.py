import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from balance_env import BalanceEnv


ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "booster_gym_reference"
CONFIG_PATH = REFERENCE_DIR / "envs" / "T1.yaml"
TEACHER_PATH = REFERENCE_DIR / "deploy" / "models" / "T1.pt"

COMMAND_X = 0.10
COMMAND_Y = 0.0
COMMAND_YAW = 0.015
RESIDUAL_SCALE = 0.05


def quat_rotate_inverse(q, v):
    qw = q[-1]
    qv = q[:3]
    return (
        v * (2.0 * qw**2 - 1.0)
        - np.cross(qv, v) * (2.0 * qw)
        + qv * (2.0 * np.dot(qv, v))
    )


class ResidualPPOEnv(gym.Env):
    """PPO learns small corrections to a fixed T1 walking teacher."""

    metadata = {"render_modes": []}

    def __init__(self, episode_seconds=20.0):
        super().__init__()

        self.config = yaml.safe_load(CONFIG_PATH.read_text())
        self.norm = self.config["normalization"]
        self.teacher = torch.jit.load(
            str(TEACHER_PATH), map_location="cpu"
        ).eval()

        # Keep the original T1 model and use the same leg controller
        # configuration as the successful teacher playback.
        self.core = BalanceEnv(
            episode_seconds=episode_seconds,
            reset_noise=0.0,
            alpha=1.0,
            stance_weight=0.0,
            drift_weight=0.0,
            rate_weight=0.0,
            joint_velocity_weight=0.0,
        )

        self.model = self.core.model
        self.data = self.core.data
        self.leg_actuators = self.core.lower_body_ids.copy()

        self.dt = float(self.model.opt.timestep)
        self.decimation = int(self.config["control"]["decimation"])
        if self.core.frame_skip != self.decimation:
            raise RuntimeError(
                f"Frame skip mismatch: environment={self.core.frame_skip}, "
                f"teacher={self.decimation}"
            )

        self.action_scale = float(self.config["control"]["action_scale"])
        self.gait_frequency = float(
            np.mean(self.config["commands"]["gait_frequency"])
        )
        self.target_speed = COMMAND_X
        self.residual_scale = RESIDUAL_SCALE
        self.max_episode_steps = max(
            1,
            int(episode_seconds / (self.dt * self.decimation)),
        )

        self.leg_joints = self.model.actuator_trnid[
            self.leg_actuators, 0
        ].astype(int)
        self.leg_qpos = self.model.jnt_qposadr[
            self.leg_joints
        ].astype(int)
        self.leg_qvel = self.model.jnt_dofadr[
            self.leg_joints
        ].astype(int)

        defaults = self.config["init_state"]["default_joint_angles"]
        stiffness = self.config["control"]["stiffness"]
        damping = self.config["control"]["damping"]

        self.default_pos = np.zeros(12, dtype=np.float64)

        for i, actuator in enumerate(self.leg_actuators):
            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                int(actuator),
            ) or ""

            default = float(defaults["default"])
            for key, value in defaults.items():
                if key != "default" and key in name:
                    default = float(value)
                    break

            gain_key = next(
                (key for key in stiffness if key in name), None
            )
            if gain_key is None:
                raise RuntimeError(f"No teacher PD gains for {name}")

            kp = float(stiffness[gain_key])
            kd = float(damping[gain_key])

            self.default_pos[i] = default
            self.core.home_ctrl[actuator] = default

            self.model.actuator_gainprm[actuator, 0] = kp
            self.model.actuator_biasprm[actuator, 1] = -kp
            self.model.actuator_biasprm[actuator, 2] = -kd

        self.home_height = float(self.core.home_height)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # 47 teacher inputs + body-frame linear velocity (3)
        # + root height offset (1) + previous PPO residual (12).
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(63,), dtype=np.float32
        )

        self.phase = 0.0
        self.steps = 0
        self.previous_teacher_action = np.zeros(12, dtype=np.float32)
        self.previous_residual = np.zeros(12, dtype=np.float32)

    def _teacher_observation(self):
        quat = self.data.sensor("orientation").data[
            [1, 2, 3, 0]
        ].astype(np.float32)

        angular_vel = self.data.sensor(
            "angular-velocity"
        ).data.astype(np.float32)

        gravity = quat_rotate_inverse(
            quat,
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )

        dof_pos = self.data.qpos[self.leg_qpos].astype(np.float32)
        dof_vel = self.data.qvel[self.leg_qvel].astype(np.float32)

        obs = np.zeros(47, dtype=np.float32)
        obs[0:3] = gravity * self.norm["gravity"]
        obs[3:6] = angular_vel * self.norm["ang_vel"]
        obs[6] = COMMAND_X * self.norm["lin_vel"]
        obs[7] = COMMAND_Y * self.norm["lin_vel"]
        obs[8] = COMMAND_YAW * self.norm["ang_vel"]
        obs[9] = np.cos(2.0 * np.pi * self.phase)
        obs[10] = np.sin(2.0 * np.pi * self.phase)
        obs[11:23] = (
            dof_pos - self.default_pos
        ) * self.norm["dof_pos"]
        obs[23:35] = dof_vel * self.norm["dof_vel"]
        obs[35:47] = self.previous_teacher_action

        return obs

    def _rotation_and_velocity(self):
        rotation = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        rotation = rotation.reshape(3, 3)

        body_velocity = rotation.T @ self.data.qvel[:3]

        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(
            np.arctan2(
                -rotation[2, 0],
                np.hypot(rotation[2, 1], rotation[2, 2]),
            )
        )

        return rotation, body_velocity, roll, pitch

    def _observation(self):
        teacher_obs = self._teacher_observation()
        _, body_velocity, _, _ = self._rotation_and_velocity()

        return np.concatenate(
            [
                teacher_obs,
                body_velocity.astype(np.float32),
                np.array(
                    [self.data.qpos[2] - self.home_height],
                    dtype=np.float32,
                ),
                self.previous_residual,
            ]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.core.reset(seed=seed)
        self.phase = 0.0
        self.steps = 0
        self.previous_teacher_action.fill(0.0)
        self.previous_residual.fill(0.0)

        return self._observation(), {}

    def step(self, residual_action):
        residual = np.clip(
            np.asarray(residual_action, dtype=np.float32).reshape(12),
            -1.0,
            1.0,
        )

        # The teacher sees exactly the 47 features used in its
        # successful standalone playback.
        teacher_obs = self._teacher_observation()
        with torch.no_grad():
            teacher_action = self.teacher(
                torch.from_numpy(teacher_obs).unsqueeze(0)
            ).squeeze(0).cpu().numpy().astype(np.float32)

        teacher_action = np.clip(
            teacher_action,
            -self.norm["clip_actions"],
            self.norm["clip_actions"],
        )

        # PPO changes the teacher action by at most 0.05 in each
        # normalized action dimension before joint-target conversion.
        combined_action = np.clip(
            teacher_action + self.residual_scale * residual,
            -self.norm["clip_actions"],
            self.norm["clip_actions"],
        )

        self.data.ctrl[:] = self.core.home_ctrl

        targets = self.default_pos + self.action_scale * combined_action
        limits = self.model.actuator_ctrlrange[self.leg_actuators]
        targets = np.clip(targets, limits[:, 0], limits[:, 1])
        self.data.ctrl[self.leg_actuators] = targets

        # Preserve the teacher's physics and gait-phase timing.
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
            self.phase = (
                self.phase + self.dt * self.gait_frequency
            ) % 1.0

        self.previous_teacher_action = teacher_action.copy()
        self.previous_residual = residual.copy()
        self.steps += 1

        rotation, body_velocity, roll, pitch = (
            self._rotation_and_velocity()
        )
        forward_speed = float(body_velocity[0])
        lateral_speed = float(body_velocity[1])
        height = float(self.data.qpos[2])
        upright = float(rotation[2, 2])
        tilt_deg = float(
            np.degrees(
                np.arccos(np.clip(upright, -1.0, 1.0))
            )
        )

        # Reward forward speed tracking while preserving stability.
        speed_reward = 3.0 * np.exp(
            -((forward_speed - self.target_speed) / 0.12) ** 2
        )
        progress_reward = 1.0 * float(
            np.clip(forward_speed, -0.20, 0.30)
        )
        tilt_cost = 8.0 * (roll**2 + pitch**2)
        lateral_cost = 0.8 * lateral_speed**2
        height_cost = 4.0 * (height - self.home_height) ** 2
        residual_cost = 0.03 * float(np.mean(residual**2))

        reward = float(
            speed_reward
            + progress_reward
            - tilt_cost
            - lateral_cost
            - height_cost
            - residual_cost
        )

        terminated = bool(height < 0.38 or upright < 0.40)
        truncated = bool(
            self.steps >= self.max_episode_steps and not terminated
        )

        if terminated:
            reward -= 15.0

        info = {
            "forward_speed_m_s": forward_speed,
            "lateral_speed_m_s": lateral_speed,
            "root_tilt_deg": tilt_deg,
            "root_height_m": height,
            "speed_reward": float(speed_reward),
            "residual_rms": float(np.sqrt(np.mean(residual**2))),
        }

        return self._observation(), reward, terminated, truncated, info

    def close(self):
        self.core.close()


def evaluate(env, policy=None, seed=42):
    obs, _ = env.reset(seed=seed)
    start_xy = env.data.qpos[:2].copy()
    speeds, lateral_speeds, tilts = [], [], []
    fell = False
    steps = 0

    for _ in range(env.max_episode_steps):
        if policy is None:
            # Zero residual = teacher-only baseline.
            action = np.zeros(12, dtype=np.float32)
        else:
            action, _ = policy.predict(obs, deterministic=True)

        obs, _, terminated, truncated, info = env.step(action)
        speeds.append(info["forward_speed_m_s"])
        lateral_speeds.append(info["lateral_speed_m_s"])
        tilts.append(info["root_tilt_deg"])
        steps += 1

        if terminated:
            fell = True
            break
        if truncated:
            break

    displacement = env.data.qpos[:2] - start_xy

    return {
        "duration_s": round(steps * env.dt * env.decimation, 2),
        "fell": fell,
        "forward_displacement_m": round(float(displacement[0]), 4),
        "lateral_displacement_m": round(float(displacement[1]), 4),
        "mean_body_forward_speed_m_s": round(float(np.mean(speeds)), 4),
        "mean_lateral_speed_m_s": round(float(np.mean(lateral_speeds)), 4),
        "max_tilt_deg": round(float(np.max(tilts)), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5120)
    parser.add_argument("--eval-every", type=int, default=5120)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    if args.steps <= 0 or args.eval_every <= 0:
        raise ValueError("--steps and --eval-every must be positive.")

    if not TEACHER_PATH.is_file():
        raise FileNotFoundError(f"Teacher policy not found: {TEACHER_PATH}")

    print("===== RESIDUAL PPO: TEACHER BASELINE TEST =====")
    eval_env = ResidualPPOEnv(episode_seconds=20.0)

    try:
        baseline = evaluate(eval_env, policy=None)
        print(json.dumps(baseline, indent=2))

        if baseline["fell"] or baseline["duration_s"] < 19.9:
            print(
                "\nSTOP: the teacher baseline failed in this environment. "
                "PPO training has not started."
            )
            return

        out_dir = ROOT / "walk_runs" / (
            f"residual_ppo_{datetime.now():%Y%m%d_%H%M%S}"
        )
        out_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(Path(__file__).resolve(), out_dir / "train_residual_ppo.py")

        settings = {
            "method": "teacher-guided residual PPO",
            "teacher_policy": str(TEACHER_PATH),
            "controller_config": str(CONFIG_PATH),
            "residual_scale": RESIDUAL_SCALE,
            "target_speed_m_s": COMMAND_X,
            "gait_frequency_hz": eval_env.gait_frequency,
            "steps_requested": args.steps,
            "eval_every": args.eval_every,
            "learning_rate": args.learning_rate,
            "observation_size": 63,
            "action_size": 12,
            "teacher_only_baseline": baseline,
        }
        (out_dir / "settings.json").write_text(
            json.dumps(settings, indent=2)
        )

        train_env = Monitor(ResidualPPOEnv(episode_seconds=20.0))

        try:
            policy = PPO(
                "MlpPolicy",
                train_env,
                policy_kwargs={
                    "net_arch": [128, 128],
                    "log_std_init": -2.3,
                },
                learning_rate=args.learning_rate,
                n_steps=1024,
                batch_size=256,
                n_epochs=5,
                gamma=0.995,
                gae_lambda=0.95,
                ent_coef=0.001,
                clip_range=0.1,
                target_kl=0.02,
                device="cpu",
                seed=42,
                verbose=1,
            )

            policy.save(str(out_dir / "initial_policy"))
            results = []
            trained = 0

            while trained < args.steps:
                chunk = min(args.eval_every, args.steps - trained)

                policy.learn(
                    total_timesteps=chunk,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                trained += chunk

                checkpoint = out_dir / f"checkpoint_after_{trained:06d}.zip"
                policy.save(str(checkpoint))

                result = evaluate(eval_env, policy=policy)
                record = {
                    "training_steps": trained,
                    "checkpoint": str(checkpoint),
                    "evaluation": result,
                }
                results.append(record)
                (out_dir / "evaluations.json").write_text(
                    json.dumps(results, indent=2)
                )

                print("\n===== PPO CHECKPOINT EVALUATION =====")
                print(json.dumps(record, indent=2))

                if result["fell"]:
                    print(
                        "Evaluation fell; ending this experiment early. "
                        "The saved checkpoint is retained."
                    )
                    break

            print(f"\nResults saved to: {out_dir}")

        finally:
            train_env.close()
            eval_env.close()

    except Exception:
        eval_env.close()
        raise


if __name__ == "__main__":
    main()
