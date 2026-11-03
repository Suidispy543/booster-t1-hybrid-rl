from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from balance_env import BalanceEnv

ROOT = Path(__file__).resolve().parent
REF = ROOT / "booster_gym_reference"
CONFIG = yaml.safe_load((REF / "envs" / "T1.yaml").read_text())
TEACHER_PATH = REF / "deploy" / "models" / "T1.pt"

COMMAND_X, COMMAND_Y, COMMAND_YAW = 0.10, 0.0, 0.015
ARM_NAMES = [
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]
ARM_TARGET = np.array([-0.300, 1.382, 1.459, 0.664])


def quat_rotate_inverse(q, v):
    qw, qv = q[-1], q[:3]
    return (
        v * (2 * qw**2 - 1)
        - np.cross(qv, v) * (2 * qw)
        + qv * (2 * np.dot(qv, v))
    )


def main():
    teacher = torch.jit.load(str(TEACHER_PATH), map_location="cpu").eval()

    env = BalanceEnv(
        episode_seconds=12.0,
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
        dt = float(model.opt.timestep)
        decimation = int(CONFIG["control"]["decimation"])
        frequency = float(np.mean(CONFIG["commands"]["gait_frequency"]))
        env.action_scale = float(CONFIG["control"]["action_scale"])

        leg_actuators = env.lower_body_ids.copy()
        leg_joints = model.actuator_trnid[leg_actuators, 0].astype(int)
        leg_qpos = model.jnt_qposadr[leg_joints].astype(int)
        leg_qvel = model.jnt_dofadr[leg_joints].astype(int)

        defaults = CONFIG["init_state"]["default_joint_angles"]
        stiffness = CONFIG["control"]["stiffness"]
        damping = CONFIG["control"]["damping"]
        default_pos = np.zeros(12)

        for i, actuator in enumerate(leg_actuators):
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(actuator)
            ) or ""
            default_pos[i] = float(defaults["default"])
            for key, value in defaults.items():
                if key != "default" and key in name:
                    default_pos[i] = float(value)
                    break

            gain_key = next((key for key in stiffness if key in name), None)
            if gain_key is None:
                raise RuntimeError(f"No configured PD gains for {name}")

            kp, kd = float(stiffness[gain_key]), float(damping[gain_key])
            env.home_ctrl[actuator] = default_pos[i]
            model.actuator_gainprm[actuator, 0] = kp
            model.actuator_biasprm[actuator, 1] = -kp
            model.actuator_biasprm[actuator, 2] = -kd

        arm_ids = np.array([
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            for name in ARM_NAMES
        ])
        if np.any(arm_ids < 0):
            raise RuntimeError("Could not find all right arm actuators")

        arm_home = env.home_ctrl[arm_ids].copy()
        arm_target = ARM_TARGET.copy()
        for i, actuator in enumerate(arm_ids):
            if model.actuator_ctrllimited[actuator]:
                lo, hi = model.actuator_ctrlrange[actuator]
                arm_target[i] = np.clip(arm_target[i], lo, hi)

        env.reset(seed=42)
        data.ctrl[:] = env.home_ctrl

        start_xy = data.qpos[:2].copy()
        phase = 0.0
        previous_action = np.zeros(12, dtype=np.float32)
        max_tilt = 0.0
        fell = False
        duration = 12.0
        steps = int(duration / (dt * decimation))

        print("Teacher walking + right-arm trajectory; 12-second test")
        print("Arm moves from 2 to 4 seconds, then holds the target pose.")

        for step in range(steps):
            t = step * dt * decimation
            blend = np.clip((t - 2.0) / 2.0, 0.0, 1.0)
            blend = blend * blend * (3.0 - 2.0 * blend)
            env.home_ctrl[arm_ids] = arm_home + blend * (arm_target - arm_home)

            quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            angular_vel = data.sensor("angular-velocity").data.astype(np.float32)
            gravity = quat_rotate_inverse(
                quat, np.array([0.0, 0.0, -1.0], dtype=np.float32)
            )
            dof_pos = data.qpos[leg_qpos].astype(np.float32)
            dof_vel = data.qvel[leg_qvel].astype(np.float32)

            obs = np.zeros(47, dtype=np.float32)
            obs[0:3] = gravity * norm["gravity"]
            obs[3:6] = angular_vel * norm["ang_vel"]
            obs[6] = COMMAND_X * norm["lin_vel"]
            obs[7] = COMMAND_Y * norm["lin_vel"]
            obs[8] = COMMAND_YAW * norm["ang_vel"]
            obs[9] = np.cos(2 * np.pi * phase)
            obs[10] = np.sin(2 * np.pi * phase)
            obs[11:23] = (dof_pos - default_pos) * norm["dof_pos"]
            obs[23:35] = dof_vel * norm["dof_vel"]
            obs[35:47] = previous_action

            with torch.no_grad():
                action = teacher(
                    torch.from_numpy(obs).unsqueeze(0)
                ).squeeze(0).cpu().numpy().astype(np.float32)

            action = np.clip(
                action, -norm["clip_actions"], norm["clip_actions"]
            )
            _, _, terminated, truncated, _ = env.step(action)
            previous_action = action.copy()
            phase = (phase + dt * frequency * decimation) % 1.0

            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, data.qpos[3:7])
            R = R.reshape(3, 3)
            tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
            max_tilt = max(max_tilt, tilt)
            body_velocity = R.T @ data.qvel[:3]

            if (step + 1) % 50 == 0:
                print(
                    f"t={(step+1)*dt*decimation:.1f}s | "
                    f"speed={body_velocity[0]:+.3f}m/s | "
                    f"tilt={tilt:.1f}deg | height={data.qpos[2]:.3f}m"
                )

            if terminated:
                fell = True
                print(f"Robot fell at t={(step+1)*dt*decimation:.2f}s")
                break
            if truncated:
                break

        displacement = data.qpos[:2] - start_xy
        print("\n===== RESULT =====")
        print(f"Duration: {(step+1)*dt*decimation:.2f}s")
        print(f"Fell: {fell}")
        print(f"Forward displacement: {displacement[0]:+.3f}m")
        print(f"Lateral displacement: {displacement[1]:+.3f}m")
        print(f"Maximum torso tilt: {max_tilt:.2f}deg")
        print("This test uses the teacher, not the PPO walking student.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
