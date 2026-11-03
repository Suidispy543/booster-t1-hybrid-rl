"""Train a first PPO policy for T1 standing balance. Example: --timesteps 20000"""
from pathlib import Path
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt

from t1_balance_env import T1BalanceEnv

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20000,
                        help="Number of PPO environment steps (default: 20000).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate the environment without training.")
    args = parser.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    raw_env = T1BalanceEnv()
    print("Environment dimensions:", raw_env.observation_space.shape,
          "observations;", raw_env.action_space.shape, "actions")
    check_env(raw_env, warn=True)
    if args.check_only:
        print("Environment check passed. PPO training was not started.")
        raw_env.close()
        return

    env = Monitor(raw_env, filename=str(LOG_DIR / "t1_balance"))
    model = PPO(
        "MlpPolicy", env, seed=args.seed, verbose=1, device="cpu",
        learning_rate=3e-4, n_steps=512, batch_size=128,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model_path = MODEL_DIR / "t1_balance_ppo"
    model.save(str(model_path))
    print("Saved PPO policy to:", model_path.with_suffix(".zip"))

    # Save a learning curve from completed training episodes, if any.
    try:
        results = results_plotter.load_results(str(LOG_DIR))
        x, y = results_plotter.ts2xy(results, "timesteps")
        if len(x):
            plt.figure(figsize=(8, 4.5))
            plt.plot(x, y, linewidth=1)
            plt.xlabel("Environment steps")
            plt.ylabel("Episode return")
            plt.title("T1 Standing-Balance PPO Training")
            plt.tight_layout()
            plot_path = LOG_DIR / "t1_balance_learning_curve.png"
            plt.savefig(plot_path, dpi=160)
            plt.close()
            print("Saved learning curve to:", plot_path)
        else:
            print("No completed episodes yet; no learning curve was written.")
    except Exception as exc:
        print("Training completed, but plotting failed:", exc)

    env.close()


if __name__ == "__main__":
    main()
