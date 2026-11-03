from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from balance_env import BalanceEnv

HERE = Path(__file__).resolve().parent
REF = HERE / "booster_gym_reference"
CONFIG = yaml.safe_load((REF / "envs" / "T1.yaml").read_text())
TEACHER_PATH = REF / "deploy" / "models" / "T1.pt"

COMMAND_X = 0.10
COMMAND_Y = 0.0
COMMAND_YAW = 0.015


def quat_rotate_inverse(q, v):
    qw = q[-1]
    qv = q[:3]
    return (
        v * (2.0 * qw**2 - 1.0)
        - np.cross(qv, v) * (2.0 * qw)
        + qv * (2.0 * np.dot(qv, v))
    )


def main():
    teacher = torch.jit.load(
        str(TEACHER_PATH), map_location="cpu"
    ).eval()

    env = BalanceEnv(
        episode_seconds=20.0,
        reset_noise=0.0,
        alpha=1.0,
        stance_weight=0.0,
        drift_weight=0.0,
        rate_weight=0.0,
        joint_velocity_weight=0.0,
    )

    try:
        model, data = env.model, env.data
        norm = CONFIG["normalization"]
        decimation = int(CONFIG["control"]["decimation"])
        dt = float(model.opt.timestep)
        gait_frequency = float(
            np.mean(CONFIG["commands"]["gait_frequency"])
        )

        if env.frame_skip != decimation:
            raise RuntimeError(
                f"Environment frame_skip={env.frame_skip}, "
                f"but teacher decimation={decimation}"
            )

        leg_actuators = env.lower_body_ids.copy()
        leg_joints = model.actuator_trnid[
            leg_actuators, 0
        ].astype(int)
        leg_qpos = model.jnt_qposadr[leg_joints].astype(int)
        leg_qvel = model.jnt_dofadr[leg_joints].astype(int)

        defaults = CONFIG["init_state"]["default_joint_angles"]
        stiffness = CONFIG["control"]["stiffness"]
        damping = CONFIG["control"]["damping"]

        for actuator in leg_actuators:
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(actuator)
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
                raise RuntimeError(f"No PD gains for {name}")

            kp = float(stiffness[gain_key])
            kd = float(damping[gain_key])

            env.home_ctrl[actuator] = default
            model.actuator_gainprm[actuator, 0] = kp
            model.actuator_biasprm[actuator, 1] = -kp
            model.actuator_biasprm[actuator, 2] = -kd

        env.action_scale = float(CONFIG["control"]["action_scale"])
        obs_sensors = ("orientation", "angular-velocity")
        for sensor_name in obs_sensors:
            sid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
            )
            if sid < 0:
                raise RuntimeError(f"Missing sensor: {sensor_name}")

        env.reset(seed=42)

        phase = 0.0
        previous_action = np.zeros(12, dtype=np.float32)
        start_xy = data.qpos[:2].copy()
        speeds, tilts = [], []
        fell = False
        steps = 0
        seconds = 20.0
        max_steps = int(seconds / (dt * decimation))

        for step in range(max_steps):
            quat = data.sensor("orientation").data[
                [1, 2, 3, 0]
            ].astype(np.float32)

            angular_vel = data.sensor(
                "angular-velocity"
            ).data.astype(np.float32)

            gravity = quat_rotate_inverse(
                quat,
                np.array([0.0, 0.0, -1.0], dtype=np.float32),
            )

            dof_pos = data.qpos[leg_qpos].astype(np.float32)
            dof_vel = data.qvel[leg_qvel].astype(np.float32)

            obs = np.zeros(47, dtype=np.float32)
            obs[0:3] = gravity * norm["gravity"]
            obs[3:6] = angular_vel * norm["ang_vel"]
            obs[6] = COMMAND_X * norm["lin_vel"]
            obs[7] = COMMAND_Y * norm["lin_vel"]
            obs[8] = COMMAND_YAW * norm["ang_vel"]
            obs[9] = np.cos(2.0 * np.pi * phase)
            obs[10] = np.sin(2.0 * np.pi * phase)
            obs[11:23] = (
                dof_pos
                - np.array([
                    next(
                        (
                            float(value)
                            for key, value in defaults.items()
                            if key != "default" and key in (
                                mujoco.mj_id2name(
                                    model,
                                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    int(a),
                                ) or ""
                            )
                        ),
                        float(defaults["default"]),
                    )
                    for a in leg_actuators
                ], dtype=np.float32)
            ) * norm["dof_pos"]
            obs[23:35] = dof_vel * norm["dof_vel"]
            obs[35:47] = previous_action

            with torch.no_grad():
                action = teacher(
                    torch.from_numpy(obs).unsqueeze(0)
                ).squeeze(0).numpy().astype(np.float32)

            action = np.clip(
                action,
                -norm["clip_actions"],
                norm["clip_actions"],
            )

            _, _, terminated, truncated, _ = env.step(action)
            previous_action = action.copy()

            phase = (
                phase + dt * gait_frequency * decimation
            ) % 1.0

            rotation = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            rotation = rotation.reshape(3, 3)

            tilt = float(np.degrees(np.arccos(
                np.clip(rotation[2, 2], -1.0, 1.0)
            )))
            body_velocity = rotation.T @ data.qvel[:3]
            speeds.append(float(body_velocity[0]))
            tilts.append(tilt)
            steps += 1

            if (step + 1) % 100 == 0:
                print(
                    f"t={(step + 1) * dt * decimation:.1f}s | "
                    f"forward_speed={body_velocity[0]:+.3f} m/s | "
                    f"tilt={tilt:.1f} deg | "
                    f"height={data.qpos[2]:.3f} m"
                )

            if terminated:
                fell = True
                break
            if truncated:
                break

        displacement = data.qpos[:2] - start_xy

        print("\n===== TEACHER GYMNASIUM TEST =====")
        print(f"Duration: {steps * dt * decimation:.2f} s")
        print(f"Fell: {fell}")
        print(f"Forward displacement: {displacement[0]:+.3f} m")
        print(f"Lateral displacement: {displacement[1]:+.3f} m")
        print(f"Mean body-forward speed: {np.mean(speeds):+.3f} m/s")
        print(f"Maximum torso tilt: {np.max(tilts):.2f} deg")

    finally:
        env.close()


if __name__ == "__main__":
    main()
