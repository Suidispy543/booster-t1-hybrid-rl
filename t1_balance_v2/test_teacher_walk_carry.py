from pathlib import Path

import numpy as np
import mujoco
import torch
import yaml
import base_env

SCENE = Path("../assets/mujoco_menagerie/booster_t1/scene_pick_place.xml").resolve()
REF = Path("booster_gym_reference")
CONFIG = yaml.safe_load((REF / "envs/T1.yaml").read_text())
TEACHER_PATH = REF / "deploy/models/T1.pt"

COMMAND_X = 0.07
COMMAND_Y = 0.0
COMMAND_YAW = 0.0
DURATION = 8.0

ARM_NAMES = [
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]

CARRY_ARM = np.array([-0.6486, 1.3963, 1.4596, 0.6612])


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3.0 - 2.0*x)


def quat_rotate_inverse(q, v):
    qw, qv = q[-1], q[:3]
    return (
        v * (2.0 * qw**2 - 1.0)
        - np.cross(qv, v) * (2.0 * qw)
        + qv * (2.0 * np.dot(qv, v))
    )


def main():
    if not SCENE.is_file():
        raise FileNotFoundError(SCENE)
    if not TEACHER_PATH.is_file():
        raise FileNotFoundError(TEACHER_PATH)

    # Load the actual pick-and-place scene.
    base_env.MODEL_PATH = SCENE
    from balance_env import BalanceEnv

    teacher = torch.jit.load(
        str(TEACHER_PATH), map_location="cpu"
    ).eval()

    env = BalanceEnv(
        episode_seconds=DURATION + 2.0,
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
        policy_dt = dt * decimation
        frequency = float(
            np.mean(CONFIG["commands"]["gait_frequency"])
        )

        if env.frame_skip != decimation:
            raise RuntimeError("Teacher and environment decimation differ.")

        leg_ids = env.lower_body_ids.copy()
        leg_joints = model.actuator_trnid[leg_ids, 0].astype(int)
        leg_qpos = model.jnt_qposadr[leg_joints].astype(int)
        leg_qvel = model.jnt_dofadr[leg_joints].astype(int)

        # Apply the exact teacher-compatible PD gains and defaults.
        defaults = CONFIG["init_state"]["default_joint_angles"]
        stiffness = CONFIG["control"]["stiffness"]
        damping = CONFIG["control"]["damping"]
        default_pos = np.zeros(12)

        for i, actuator in enumerate(leg_ids):
            name = (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(actuator)
                ) or ""
            )

            value = float(defaults["default"])
            for key, val in defaults.items():
                if key != "default" and key in name:
                    value = float(val)
                    break

            gain_key = next(
                (key for key in stiffness if key in name), None
            )
            if gain_key is None:
                raise RuntimeError(f"No PD gains for {name}")

            kp = float(stiffness[gain_key])
            kd = float(damping[gain_key])

            default_pos[i] = value
            env.home_ctrl[actuator] = value

            model.actuator_gainprm[actuator, 0] = kp
            model.actuator_biasprm[actuator, 1] = -kp
            model.actuator_biasprm[actuator, 2] = -kd

        env.action_scale = float(CONFIG["control"]["action_scale"])

        arm_ids = np.array([
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            for name in ARM_NAMES
        ])
        if np.any(arm_ids < 0):
            raise RuntimeError("Right arm actuators missing.")

        arm_home = env.home_ctrl[arm_ids].copy()

        for i, actuator in enumerate(arm_ids):
            if model.actuator_ctrllimited[actuator]:
                lo, hi = model.actuator_ctrlrange[actuator]
                CARRY_ARM[i] = np.clip(CARRY_ARM[i], lo, hi)

        hand_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_link"
        )
        trunk_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "Trunk"
        )
        object_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pickup_object"
        )

        mocap_id = int(model.body_mocapid[object_id])
        if min(hand_id, trunk_id, object_id) < 0 or mocap_id < 0:
            raise RuntimeError("Required hand, trunk, or mocap object missing.")

        # Start the robot and attach the box below the right hand.
        obs, _ = env.reset(seed=42)
        R_hand = data.xmat[hand_id].reshape(3, 3).copy()

        attachment_local = np.array([0.0, 0.0, -0.04])
        object_start = (
            data.xpos[hand_id] + R_hand @ attachment_local
        )

        data.mocap_pos[mocap_id] = object_start.copy()
        data.mocap_quat[mocap_id] = np.array([1., 0., 0., 0.])
        mujoco.mj_forward(model, data)

        start_xy = data.qpos[:2].copy()
        previous_action = np.zeros(12, dtype=np.float32)
        phase = 0.0

        max_tilt = 0.0
        min_height = float(data.qpos[2])
        fell = False
        speeds = []
        next_report = 0.0

        steps = int(DURATION / policy_dt)

        print("TEACHER WALKING WITH OBJECT")
        print("Walking speed command:", COMMAND_X, "m/s")
        print("Arm: smooth transition into validated carry pose")
        print("Object: kinematic attachment to right hand")
        print("Duration:", DURATION, "seconds")

        for step in range(steps):
            t = step * policy_dt

            # Bring arm into its carrying pose during the first 1.5 s.
            blend = smooth(t / 1.5)
            env.home_ctrl[arm_ids] = (
                arm_home + blend * (CARRY_ARM - arm_home)
            )

            quat = data.sensor("orientation").data[
                [1, 2, 3, 0]
            ].astype(np.float32)

            angular = data.sensor(
                "angular-velocity"
            ).data.astype(np.float32)

            gravity = quat_rotate_inverse(
                quat,
                np.array([0., 0., -1.], dtype=np.float32)
            )

            pos = data.qpos[leg_qpos].astype(np.float32)
            vel = data.qvel[leg_qvel].astype(np.float32)

            observation = np.zeros(47, dtype=np.float32)
            observation[0:3] = gravity * norm["gravity"]
            observation[3:6] = angular * norm["ang_vel"]
            observation[6] = COMMAND_X * norm["lin_vel"]
            observation[7] = COMMAND_Y * norm["lin_vel"]
            observation[8] = COMMAND_YAW * norm["ang_vel"]
            observation[9] = np.cos(2.0 * np.pi * phase)
            observation[10] = np.sin(2.0 * np.pi * phase)
            observation[11:23] = (
                pos - default_pos
            ) * norm["dof_pos"]
            observation[23:35] = vel * norm["dof_vel"]
            observation[35:47] = previous_action

            with torch.no_grad():
                action = teacher(
                    torch.from_numpy(observation).unsqueeze(0)
                ).squeeze(0).cpu().numpy().astype(np.float32)

            action = np.clip(
                action, -norm["clip_actions"], norm["clip_actions"]
            )

            _, _, terminated, truncated, info = env.step(action)
            previous_action = action.copy()

            phase = (
                phase + dt * frequency * decimation
            ) % 1.0

            # Update the kinematic object attachment after each policy step.
            R_hand = data.xmat[hand_id].reshape(3, 3).copy()
            data.mocap_pos[mocap_id] = (
                data.xpos[hand_id] + R_hand @ attachment_local
            )
            mujoco.mj_forward(model, data)

            R_trunk = data.xmat[trunk_id].reshape(3, 3)
            tilt = float(np.degrees(np.arccos(
                np.clip(R_trunk[2, 2], -1.0, 1.0)
            )))

            body_velocity = R_trunk.T @ data.qvel[:3]
            speeds.append(float(body_velocity[0]))

            height = float(data.qpos[2])
            min_height = min(min_height, height)
            max_tilt = max(max_tilt, tilt)

            if t >= next_report:
                displacement = data.qpos[:2] - start_xy
                print(
                    f"t={t:.1f}s | speed={body_velocity[0]:+.3f}m/s | "
                    f"tilt={tilt:.2f}deg | height={height:.3f}m | "
                    f"forward={displacement[0]:+.3f}m | "
                    f"object={np.round(data.mocap_pos[mocap_id], 3)}"
                )
                next_report += 0.5

            if terminated or height < 0.38 or tilt > 35.0:
                fell = True
                print("Safety stop at", round(t + policy_dt, 2), "s")
                break

            if truncated:
                break

        displacement = data.qpos[:2] - start_xy
        object_end = data.mocap_pos[mocap_id].copy()

        print("\n========== WALK-CARRY RESULTS ==========")
        print("Fell / safety stop:", fell)
        print("Forward displacement:", round(float(displacement[0]), 3), "m")
        print("Lateral displacement:", round(float(displacement[1]), 3), "m")
        print("Mean forward speed:", round(float(np.mean(speeds)), 3), "m/s")
        print("Maximum torso tilt:", round(max_tilt, 2), "degrees")
        print("Minimum root height:", round(min_height, 3), "m")
        print("Object start:", np.round(object_start, 3))
        print("Object end:", np.round(object_end, 3))
        print("Object displacement:", np.round(object_end - object_start, 3))
        print("Final hand:", np.round(data.xpos[hand_id], 3))
        print("Carry arm target:", np.round(CARRY_ARM, 3))

    finally:
        env.close()


if __name__ == "__main__":
    main()
