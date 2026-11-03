"""Evaluate saved T1 balance policy over multiple episodes."""
from pathlib import Path
import argparse
import numpy as np
from stable_baselines3 import PPO
from t1_balance_env import T1BalanceEnv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "t1_balance_ppo.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Trained model not found: {args.model}. Run train_ppo.py first.")

    env = T1BalanceEnv()
    model = PPO.load(str(args.model), device="cpu")
    returns, lengths, falls, upright_values = [], [], [], []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=episode)
        total, length, terminated = 0.0, 0, False
        last_info = {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            length += 1
            last_info = info
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(length)
        falls.append(bool(last_info.get("fell", terminated)))
        upright_values.append(float(last_info.get("upright_score", np.nan)))
        print(f"Episode {episode+1:02d}: return={total:8.2f}, steps={length:4d}, "
              f"final_upright={upright_values[-1]: .3f}, fell={falls[-1]}")

    print("\nSummary")
    print(f"Mean return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"Mean episode length: {np.mean(lengths):.1f} steps")
    print(f"Episodes ending in a fall: {sum(falls)}/{len(falls)}")
    env.close()


if __name__ == "__main__":
    main()
