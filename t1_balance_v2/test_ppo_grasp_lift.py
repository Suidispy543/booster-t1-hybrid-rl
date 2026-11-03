from pathlib import Path

import numpy as np
import mujoco
import base_env
from stable_baselines3 import PPO

SCENE = Path(
    "../assets/mujoco_menagerie/booster_t1/scene_pick_place.xml"
).resolve()

CHECKPOINT = Path(
    "runs/stance_alpha0.5_20261028_215029_608658/"
    "checkpoint_000025600.zip"
)

base_env.MODEL_PATH = SCENE

from balance_env import BalanceEnv

ARM_NAMES = [
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]

ARM_TARGET = np.array([-0.300, 1.382, 1.459, 0.664])

CROUCH = np.array([
    0.03, 0, 0, 0.10, -0.03, 0,
    0.03, 0, 0, 0.10, -0.03, 0
], dtype=float)

PICKUP_POINT = np.array([0.130, -0.200, 0.650])
OBJECT_START = PICKUP_POINT - np.array([0.0, 0.0, 0.040])


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def main():
    if not SCENE.exists() or not CHECKPOINT.exists():
        raise FileNotFoundError("Scene or PPO checkpoint not found.")

    env = BalanceEnv(
        episode_seconds=9.0,
        reset_noise=0.0,
        alpha=0.5,
        stance_weight=0.5,
        drift_weight=0.25,
        rate_weight=0.2,
        joint_velocity_weight=0.5,
    )

    try:
        policy = PPO.load(str(CHECKPOINT), device="cpu")
        model, data = env.model, env.data

        arm_ids = np.array([
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            for name in ARM_NAMES
        ])

        if np.any(arm_ids < 0):
            raise RuntimeError("Arm actuator missing.")

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
        if mocap_id < 0:
            raise RuntimeError("Pickup object is not a mocap body.")

        obs, _ = env.reset(seed=42)

        home_legs = env.home_ctrl[env.lower_body_ids].copy()
        home_arm = env.home_ctrl[arm_ids].copy()
        initial_xy = data.qpos[:2].copy()

        # Position the box beneath the expected hand contact point.
        data.mocap_pos[mocap_id] = OBJECT_START.copy()
        data.mocap_quat[mocap_id] = np.array([1., 0., 0., 0.])
        mujoco.mj_forward(model, data)

        dt = model.opt.timestep * env.frame_skip
        steps = int(8.0 / dt)

        attached_local = None
        grasped = False

        max_tilt = 0.0
        min_height = float(data.qpos[2])
        max_drift = 0.0
        max_attachment_error = 0.0

        print("PPO GRASP AND LIFT TEST")
        print("0-1s: stabilise")
        print("1-3.5s: bend and reach")
        print("3.5s: attach object to hand")
        print("3.5-6s: stand while holding object")
        print("6-8s: maintain upright holding pose")

        next_report = 0.0

        for step in range(steps):
            t = step * dt

            # Crouch trajectory.
            if t < 1.0:
                crouch_blend = 0.0
            elif t < 3.5:
                crouch_blend = smooth((t - 1.0) / 2.5)
            elif t < 4.0:
                crouch_blend = 1.0
            elif t < 6.0:
                crouch_blend = 1.0 - smooth((t - 4.0) / 2.0)
            else:
                crouch_blend = 0.0

            # Reach and hold the arm in the pickup pose.
            arm_blend = (
                smooth((t - 1.0) / 2.5) if t < 3.5 else 1.0
            )

            env.home_ctrl[env.lower_body_ids] = (
                home_legs + crouch_blend * CROUCH
            )

            env.home_ctrl[arm_ids] = (
                home_arm + arm_blend * (ARM_TARGET - home_arm)
            )

            # PPO controls the lower-body actions.
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Capture the hand-relative object position at grasp time.
            if not grasped and t >= 3.5:
                R_hand = data.xmat[hand_id].reshape(3, 3).copy()

                attached_local = R_hand.T @ (
                    data.mocap_pos[mocap_id] - data.xpos[hand_id]
                )

                grasped = True
                print("\nOBJECT ATTACHED TO RIGHT HAND")
                print("Hand position:", np.round(data.xpos[hand_id], 3))
                print("Object position:", np.round(data.mocap_pos[mocap_id], 3))
                print("Hand-relative attachment:",
                      np.round(attached_local, 3))

            # Move the kinematic object with the hand.
            if grasped:
                R_hand = data.xmat[hand_id].reshape(3, 3).copy()
                expected_offset = R_hand @ attached_local

                data.mocap_pos[mocap_id] = (
                    data.xpos[hand_id] + expected_offset
                )

                # Keep the box upright in this first test.
                data.mocap_quat[mocap_id] = np.array([1., 0., 0., 0.])

                mujoco.mj_forward(model, data)

                actual_offset = (
                    data.mocap_pos[mocap_id] - data.xpos[hand_id]
                )

                attachment_error = float(np.linalg.norm(
                    actual_offset - expected_offset
                ))

                max_attachment_error = max(
                    max_attachment_error, attachment_error
                )

            # Monitor stability.
            R_trunk = data.xmat[trunk_id].reshape(3, 3)
            tilt = float(np.degrees(np.arccos(
                np.clip(R_trunk[2, 2], -1.0, 1.0)
            )))

            height = float(data.qpos[2])
            drift = float(np.linalg.norm(data.qpos[:2] - initial_xy))

            max_tilt = max(max_tilt, tilt)
            min_height = min(min_height, height)
            max_drift = max(max_drift, drift)

            if t >= next_report:
                print(
                    f"t={t:.1f}s | crouch={crouch_blend:.2f} | "
                    f"height={height:.3f}m | tilt={tilt:.2f}deg | "
                    f"drift={drift:.3f}m | "
                    f"object={np.round(data.mocap_pos[mocap_id], 3)}"
                )
                next_report += 0.5

            if terminated or truncated:
                print("Episode ended at:", round(t + dt, 2), "seconds")
                break

            if tilt > 25.0 or height < 0.45:
                print("Safety stop triggered.")
                break

        print("\n========== GRASP AND LIFT RESULTS ==========")
        print("Grasp attachment created:", grasped)
        print("Maximum torso tilt:", round(max_tilt, 2), "degrees")
        print("Minimum root height:", round(min_height, 3), "m")
        print("Maximum horizontal drift:", round(max_drift, 3), "m")
        print("Initial object position:", np.round(OBJECT_START, 3))
        print("Final object position:",
              np.round(data.mocap_pos[mocap_id], 3))
        print("Object vertical displacement:",
              round(float(data.mocap_pos[mocap_id][2] - OBJECT_START[2]), 3), "m")
        print("Maximum attachment error:",
              round(max_attachment_error, 5), "m")

    finally:
        env.close()


if __name__ == "__main__":
    main()
