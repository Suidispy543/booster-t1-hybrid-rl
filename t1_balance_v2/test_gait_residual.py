import numpy as np
import mujoco
from stable_baselines3 import PPO
from balance_env import BalanceEnv

CHECKPOINT = (
    "t1_balance_v2/runs/stance_alpha0.5_20261028_215029_608658/"
    "checkpoint_000025600.zip"
)

GAIT_FREQUENCY_HZ = 0.5
GAIT_GAIN = 0.35  # Start conservatively.

env = BalanceEnv(
    episode_seconds=15.0,
    reset_noise=0.0,
    alpha=0.5,
    stance_weight=0.5,
    drift_weight=0.25,
    rate_weight=0.5,
    joint_velocity_weight=5.0,
)

def gait_action(phase):
    s = np.sin(phase)
    left_swing = max(0.0, s)
    right_swing = max(0.0, -s)

    gait = np.zeros(12, dtype=np.float32)

    gait[0] = 0.25 * s
    gait[6] = -0.25 * s

    gait[3] = 0.35 * left_swing
    gait[9] = 0.35 * right_swing

    gait[4] = -0.10 * s - 0.10 * left_swing
    gait[10] = 0.10 * s - 0.10 * right_swing

    return gait

try:
    model = PPO.load(CHECKPOINT, device="cpu")
    obs, _ = env.reset(seed=42)

    dt = env.model.opt.timestep * env.frame_skip
    total_steps = int(15.0 / dt)
    phase = 0.0

    start_xy = env.data.qpos[:2].copy()
    tilts = []
    forward_speeds = []
    lateral_speeds = []
    fell = False

    print("Original T1 scene.xml")
    print("Standing PPO checkpoint: loaded")
    print("Training: none")
    print(f"Gait frequency: {GAIT_FREQUENCY_HZ:.2f} Hz")
    print(f"Gait residual gain: {GAIT_GAIN:.2f}")

    for i in range(total_steps):
        # Preserve the learned balance action.
        standing_action, _ = model.predict(obs, deterministic=True)

        # Add a gentle stepping signal to the learned action.
        residual = GAIT_GAIN * gait_action(phase)
        combined_action = np.clip(
            standing_action + residual, -1.0, 1.0
        )

        obs, _, terminated, truncated, _ = env.step(combined_action)

        R = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(R, env.data.qpos[3:7])
        R = R.reshape(3, 3)

        tilt = float(np.degrees(
            np.arccos(np.clip(R[2, 2], -1.0, 1.0))
        ))

        tilts.append(tilt)
        forward_speeds.append(float(env.data.qvel[0]))
        lateral_speeds.append(float(env.data.qvel[1]))

        phase = (
            phase + 2.0 * np.pi * GAIT_FREQUENCY_HZ * dt
        ) % (2.0 * np.pi)

        if (i + 1) % 250 == 0:
            print(
                f"t={(i + 1) * dt:5.1f}s | "
                f"tilt={tilt:6.2f} deg | "
                f"vx={env.data.qvel[0]:+.3f} m/s | "
                f"vy={env.data.qvel[1]:+.3f} m/s"
            )

        if terminated:
            fell = True
            print(f"Fall detected at {(i + 1) * dt:.2f} s")
            break

        if truncated:
            break

    displacement = env.data.qpos[:2] - start_xy

    print("\n===== GAIT-RESIDUAL TEST SUMMARY =====")
    print(f"Duration: {(i + 1) * dt:.2f} s")
    print(f"Fell: {fell}")
    print(f"Forward displacement: {displacement[0]:+.4f} m")
    print(f"Lateral displacement: {displacement[1]:+.4f} m")
    print(f"Mean forward speed: {np.mean(forward_speeds):+.4f} m/s")
    print(f"Mean lateral speed: {np.mean(lateral_speeds):+.4f} m/s")
    print(f"Mean torso tilt: {np.mean(tilts):.3f} deg")
    print(f"Maximum torso tilt: {np.max(tilts):.3f} deg")

finally:
    env.close()
