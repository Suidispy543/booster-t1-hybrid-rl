import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO
from t1_balance_env import T1BalanceEnv

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "t1_balance_ppo.zip"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    env = T1BalanceEnv()
    model = PPO.load(str(MODEL_PATH), device="cpu")
    obs, _ = env.reset()

    episode = 1
    episode_return = 0.0
    sim_dt = env.model.opt.timestep * env.frame_skip

    try:
        with mujoco.viewer.launch_passive(
            env.model, env.data
        ) as viewer:
            print("Viewing trained T1 PPO policy. Close the window to stop.")

            while viewer.is_running():
                start = time.perf_counter()

                action, _ = model.predict(
                    obs, deterministic=True
                )
                obs, reward, terminated, truncated, info = env.step(action)

                episode_return += float(reward)
                viewer.sync()

                elapsed = time.perf_counter() - start
                time.sleep(max(0.0, sim_dt - elapsed))

                if terminated or truncated:
                    print(
                        f"Episode {episode}: "
                        f"steps={env.steps}, "
                        f"return={episode_return:.2f}, "
                        f"height={info['root_height']:.3f} m, "
                        f"upright={info['upright_score']:.3f}, "
                        f"fell={info.get('fell', terminated)}"
                    )

                    time.sleep(0.5)
                    obs, _ = env.reset()
                    episode += 1
                    episode_return = 0.0
    finally:
        env.close()


if __name__ == "__main__":
    main()
