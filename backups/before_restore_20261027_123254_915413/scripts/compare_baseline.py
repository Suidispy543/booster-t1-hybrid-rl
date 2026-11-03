from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from t1_balance_env import T1BalanceEnv

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "t1_balance_ppo.zip"


def run_case(name, model=None):
    env = T1BalanceEnv()
    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    steps = 0

    while True:
        if model is None:
            # Zero action preserves the original home-pose targets.
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        if terminated or truncated:
            height = float(info["root_height"])
            upright = float(info["upright_score"])

            if height < 0.38:
                reason = "root height below threshold"
            elif upright < 0.40:
                reason = "upright score below threshold"
            else:
                reason = "time limit reached"

            print(
                f"{name}: steps={steps}, reward={total_reward:.2f}, "
                f"height={height:.3f} m, upright={upright:.3f}, "
                f"termination={reason}"
            )
            break

    env.close()


if __name__ == "__main__":
    model = PPO.load(str(MODEL_PATH), device="cpu")
    run_case("Fixed home-pose baseline")
    run_case("Trained PPO policy", model)
