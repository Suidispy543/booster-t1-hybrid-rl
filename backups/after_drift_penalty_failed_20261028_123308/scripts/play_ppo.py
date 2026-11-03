import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO

from t1_balance_env import T1BalanceEnv


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT
    / "experiments"
    / "stance_balance_20261027_224155_950849"
    / "stance"
    / "checkpoint_000075264.zip"
)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    env = T1BalanceEnv(episode_seconds=60.0)

    try:
        model = PPO.load(str(MODEL_PATH), device="cpu")
        obs, _ = env.reset(seed=42)

        episode = 1
        episode_return = 0.0
        sim_dt = env.model.opt.timestep * env.frame_skip

        with mujoco.viewer.launch_passive(
            env.model, env.data
        ) as viewer:
            print(
                "Viewing trained T1 PPO policy.\n"
                "Each episode lasts up to 30 simulated seconds.\n"
                "Close the viewer window to stop."
            )

            while viewer.is_running():
                start = time.perf_counter()

                action, _ = model.predict(
                    obs, deterministic=True
                )
                obs, reward, terminated, truncated, info = env.step(
                    action
                )

                episode_return += float(reward)
                viewer.sync()

                elapsed = time.perf_counter() - start
                time.sleep(max(0.0, sim_dt - elapsed))

                if terminated or truncated:
                    duration = env.steps * sim_dt
                    reason = (
                        "fall threshold reached"
                        if terminated
                        else "time limit reached"
                    )

                    print(
                        f"Episode {episode}: "
                        f"duration={duration:.2f} s, "
                        f"steps={env.steps}, "
                        f"return={episode_return:.2f}, "
                        f"height={info['root_height']:.3f} m, "
                        f"upright={info['upright_score']:.4f}, "
                        f"termination={reason}"
                    )

                    time.sleep(0.5)

                    if not viewer.is_running():
                        break

                    obs, _ = env.reset(seed=42)
                    episode += 1
                    episode_return = 0.0
                    viewer.sync()

    finally:
        env.close()


if __name__ == "__main__":
    main()