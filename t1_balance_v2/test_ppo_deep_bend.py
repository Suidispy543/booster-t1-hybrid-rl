from pathlib import Path
import numpy as np
import mujoco
import base_env
from stable_baselines3 import PPO

SCENE = Path("../assets/mujoco_menagerie/booster_t1/scene_pick_place.xml").resolve()
CHECKPOINT = Path(
    "runs/stance_alpha0.5_20261028_215029_608658/"
    "checkpoint_000025600.zip"
)

base_env.MODEL_PATH = SCENE
from balance_env import BalanceEnv

# 2.5x the previously tested crouch offsets.
# This is a controlled test, not yet the final bending trajectory.
BEND = 2.5 * np.array([
    0.03, 0, 0, 0.10, -0.03, 0,
    0.03, 0, 0, 0.10, -0.03, 0
], dtype=float)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3.0 - 2.0*x)


def trunk_angles(data, body_id):
    R = data.xmat[body_id].reshape(3, 3)
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    pitch = np.degrees(np.arctan2(
        -R[2, 0], np.hypot(R[2, 1], R[2, 2])
    ))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return float(roll), float(pitch), float(yaw)


def main():
    env = BalanceEnv(
        episode_seconds=8.0,
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

        trunk_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "Trunk"
        )
        hand_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_link"
        )

        if min(trunk_id, hand_id) < 0:
            raise RuntimeError("Trunk or right hand body missing.")

        obs, _ = env.reset(seed=42)
        home_legs = env.home_ctrl[env.lower_body_ids].copy()
        initial_xy = data.qpos[:2].copy()
        initial_hand = data.xpos[hand_id].copy()

        joint_ids = model.actuator_trnid[
            env.lower_body_ids, 0
        ].astype(int)
        qpos_ids = model.jnt_qposadr[joint_ids].astype(int)
        initial_joints = data.qpos[qpos_ids].copy()

        dt = model.opt.timestep * env.frame_skip
        steps = int(7.0 / dt)

        max_tilt = 0.0
        min_height = float(data.qpos[2])
        max_drift = 0.0
        max_abs_pitch = 0.0
        fell = False
        next_report = 0.0

        print("PPO DEEP BENDING TEST")
        print("0-1s: standing")
        print("1-3s: gradual bend")
        print("3-4s: hold")
        print("4-6s: return upright")
        print("6-7s: stabilise")
        print("Bend multiplier:", 2.5)

        for step in range(steps):
            t = step * dt

            if t < 1.0:
                blend = 0.0
            elif t < 3.0:
                blend = smooth((t - 1.0) / 2.0)
            elif t < 4.0:
                blend = 1.0
            elif t < 6.0:
                blend = 1.0 - smooth((t - 4.0) / 2.0)
            else:
                blend = 0.0

            env.home_ctrl[env.lower_body_ids] = (
                home_legs + blend * BEND
            )

            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            roll, pitch, yaw = trunk_angles(data, trunk_id)
            tilt = float(np.degrees(np.arccos(np.clip(
                data.xmat[trunk_id].reshape(3, 3)[2, 2], -1.0, 1.0
            ))))

            height = float(data.qpos[2])
            drift = float(np.linalg.norm(data.qpos[:2] - initial_xy))
            hand = data.xpos[hand_id].copy()
            hand_delta = hand - initial_hand
            joint_delta = data.qpos[qpos_ids] - initial_joints

            max_tilt = max(max_tilt, tilt)
            max_abs_pitch = max(max_abs_pitch, abs(pitch))
            min_height = min(min_height, height)
            max_drift = max(max_drift, drift)

            if t >= next_report:
                print(
                    f"t={t:.1f}s | bend={blend:.2f} | "
                    f"height={height:.3f}m | "
                    f"pitch={pitch:+.2f}deg | "
                    f"roll={roll:+.2f}deg | "
                    f"tilt={tilt:.2f}deg | drift={drift:.3f}m"
                )
                print("  Hand displacement:", np.round(hand_delta, 3))
                print("  Actual leg joint changes:",
                      np.round(joint_delta, 3))
                next_report += 0.5

            if terminated:
                fell = True
                print("PPO episode terminated at", round(t + dt, 2), "s")
                break

            if truncated:
                break

            if tilt > 20.0 or height < 0.45:
                fell = True
                print("Safety stop triggered.")
                break

        final_hand = data.xpos[hand_id].copy()

        print("\n========== DEEP BEND RESULTS ==========")
        print("Fell / safety stop:", fell)
        print("Maximum torso tilt:", round(max_tilt, 2), "degrees")
        print("Maximum absolute trunk pitch:", round(max_abs_pitch, 2), "degrees")
        print("Minimum root height:", round(min_height, 3), "m")
        print("Maximum horizontal drift:", round(max_drift, 3), "m")
        print("Hand displacement:", np.round(final_hand-initial_hand, 3))
        print("Final root height:", round(float(data.qpos[2]), 3), "m")

    finally:
        env.close()


if __name__ == "__main__":
    main()
