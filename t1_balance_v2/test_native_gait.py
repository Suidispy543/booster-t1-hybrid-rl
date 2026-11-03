import numpy as np
from balance_env import BalanceEnv
import mujoco

# Original Google DeepMind MuJoCo Menagerie T1 model.
# This test does not train or load/change any PPO checkpoint.

env = BalanceEnv(
    episode_seconds=15.0,
    reset_noise=0.0,
    alpha=0.5,
    stance_weight=0.5,
    drift_weight=0.0,
    rate_weight=0.5,
    joint_velocity_weight=1.0,
)

# Conservative reference gait.
# Start small and slow; don't override the environment's default action_scale.
GAIT_FREQUENCY_HZ = 0.6

def make_gait(phase):
    """
    Action order:
    left hip pitch, roll, yaw, knee, ankle pitch, roll,
    right hip pitch, roll, yaw, knee, ankle pitch, roll.
    """
    s = float(np.sin(phase))
    left_swing = max(0.0, s)
    right_swing = max(0.0, -s)

    action = np.zeros(12, dtype=np.float32)

    # Small opposing hip-pitch motions.
    action[0] = 0.25 * s
    action[6] = -0.25 * s

    # Gentle knee flexion during the nominal swing half-cycle.
    action[3] = 0.35 * left_swing
    action[9] = 0.35 * right_swing

    # Small ankle-pitch compensation.
    action[4] = -0.10 * s - 0.10 * left_swing
    action[10] = 0.10 * s - 0.10 * right_swing

    return action

try:
    obs, _ = env.reset(seed=42)

    dt = env.model.opt.timestep * env.frame_skip
    steps = int(15.0 / dt)
    phase = 0.0

    start_xy = env.data.qpos[:2].copy()
    tilts = []
    forward_speeds = []
    lateral_speeds = []
    fell = False
    completed_steps = 0

    print("Model: original T1 scene.xml")
    print("Training: none")
    print(f"Gait frequency: {GAIT_FREQUENCY_HZ:.2f} Hz")
    print(f"Action scale: {env.action_scale:.3f}")
    print(f"Controller interval: {dt:.3f} s")

    for i in range(steps):
        action = make_gait(phase)
        obs, _, terminated, truncated, info = env.step(action)

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

        completed_steps += 1

        if i % 250 == 249:
            print(
                f"t={(i + 1) * dt:5.1f}s | "
                f"tilt={tilt:6.2f} deg | "
                f"vx={env.data.qvel[0]:+.3f} m/s | "
                f"vy={env.data.qvel[1]:+.3f} m/s | "
                f"height={env.data.qpos[2]:.3f} m"
            )

        if terminated:
            fell = True
            print(f"Fall detected at {(i + 1) * dt:.2f} s")
            break

        if truncated:
            break

    displacement = env.data.qpos[:2] - start_xy

    print("\n===== NATIVE GAIT TEST SUMMARY =====")
    print(f"Duration: {completed_steps * dt:.2f} s")
    print(f"Fell: {fell}")
    print(f"Forward displacement (world X): {displacement[0]:+.4f} m")
    print(f"Lateral displacement (world Y): {displacement[1]:+.4f} m")

    if tilts:
        print(f"Mean torso tilt: {np.mean(tilts):.3f} deg")
        print(f"Maximum torso tilt: {np.max(tilts):.3f} deg")

    if forward_speeds:
        print(f"Mean forward speed: {np.mean(forward_speeds):+.4f} m/s")
        print(f"Mean lateral speed: {np.mean(lateral_speeds):+.4f} m/s")

finally:
    env.close()
