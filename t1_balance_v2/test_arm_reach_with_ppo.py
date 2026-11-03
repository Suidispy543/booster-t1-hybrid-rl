import numpy as np
import mujoco
from stable_baselines3 import PPO

from balance_env import BalanceEnv

CHECKPOINT = (
    "runs/stance_alpha0.5_20261028_215029_608658/"
    "checkpoint_000025600.zip"
)

ARM_NAMES = [
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]

# Candidate pose from the numerical IK test.
TARGET_ANGLES = np.array([-0.300, 1.382, 1.459, 0.664])
PICKUP_POINT = np.array([0.130, -0.200, 0.650])


def main():
    policy = PPO.load(CHECKPOINT, device="cpu")
    env = BalanceEnv(
        episode_seconds=10.0,
        reset_noise=0.0,
        alpha=0.5,
        stance_weight=0.5,
        drift_weight=0.25,
        rate_weight=0.2,
        joint_velocity_weight=0.5,
    )

    try:
        model, data = env.model, env.data
        arm_ids = np.array([
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            for name in ARM_NAMES
        ])

        if np.any(arm_ids < 0):
            raise RuntimeError("One or more arm actuators were not found.")

        arm_home = env.home_ctrl[arm_ids].copy()
        target_angles = TARGET_ANGLES.copy()

        # Respect actuator control limits.
        for i, actuator in enumerate(arm_ids):
            if model.actuator_ctrllimited[actuator]:
                lo, hi = model.actuator_ctrlrange[actuator]
                target_angles[i] = np.clip(target_angles[i], lo, hi)

        hand_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_link"
        )

        obs, _ = env.reset(seed=42)
        dt = model.opt.timestep * env.frame_skip
        total_steps = int(10.0 / dt)
        max_tilt = 0.0
        fell = False

        print("PPO standing policy: loaded")
        print("Arm trajectory: smooth interpolation from home to IK target")
        print("Arms begin moving at 2 seconds and reach target at 4 seconds.")

        for step in range(total_steps):
            t = step * dt

            # Keep the trained PPO policy controlling the legs.
            # Smoothly move only the right arm from 2s to 4s.
            blend = np.clip((t - 2.0) / 2.0, 0.0, 1.0)
            blend = blend * blend * (3.0 - 2.0 * blend)
            arm_target = arm_home + blend * (target_angles - arm_home)
            env.home_ctrl[arm_ids] = arm_target

            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            rotation = rotation.reshape(3, 3)
            tilt = float(np.degrees(
                np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))
            ))
            max_tilt = max(max_tilt, tilt)

            if step % max(1, int(1.0 / dt)) == 0:
                hand_pos = data.xpos[hand_id].copy()
                print(
                    f"t={t:.1f}s | height={data.qpos[2]:.3f}m | "
                    f"tilt={tilt:.2f}deg | "
                    f"hand={np.round(hand_pos, 3)}"
                )

            if terminated:
                fell = True
                print(f"Robot fell at t={t + dt:.2f}s.")
                break
            if truncated:
                break

        final_hand = data.xpos[hand_id].copy()
        print("\n===== ARM REACH TEST RESULTS =====")
        print(f"Fell: {fell}")
        print(f"Maximum torso tilt: {max_tilt:.2f} degrees")
        print("Target point:", np.round(PICKUP_POINT, 3))
        print("Final hand point:", np.round(final_hand, 3))
        print(
            "Hand-to-target distance:",
            f"{np.linalg.norm(final_hand - PICKUP_POINT):.3f} m",
        )
        print("Final arm targets:", np.round(target_angles, 3))

    finally:
        env.close()


if __name__ == "__main__":
    main()
