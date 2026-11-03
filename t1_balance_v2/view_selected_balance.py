import time
from pathlib import Path

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from experiment_tilt_linear import BalanceEnv

CHECKPOINT = Path(
    "t1_balance_v2/runs/stance_alpha0.5_20261028_215029_608658/"
    "checkpoint_000025600.zip"
)

config = dict(
    alpha=0.5,
    stance_weight=0.5,
    drift_weight=0.25,
    rate_weight=0.5,
    joint_velocity_weight=5.0,
)

env = BalanceEnv(
    episode_seconds=120.0,
    reset_noise=0.0,
    **config,
)

try:
    model = PPO.load(str(CHECKPOINT), device="cpu")
    obs, _ = env.reset(seed=42)

    print("Loaded standing checkpoint:", CHECKPOINT)
    print("MuJoCo viewer opened. Close the viewer to stop.")
    print("Inspect the torso from the front and side views.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -10
        viewer.cam.lookat[:] = [0.0, 0.0, 0.75]

        while viewer.is_running():
            start = time.perf_counter()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            viewer.sync()

            if terminated:
                print("The controller terminated: the robot may have fallen.")
                break

            if truncated:
                print("Episode finished. Resetting to the standing pose.")
                obs, _ = env.reset()
                continue

            # Approximately 50 controller updates per second.
            remaining = 0.02 - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)

finally:
    env.close()
    print("Viewer stopped.")
