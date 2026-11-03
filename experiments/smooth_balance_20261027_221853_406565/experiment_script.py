"""Isolated matched continuation experiment for the existing T1 PPO checkpoint.

Put in scripts/. Run: python scripts/train_smooth_balance.py --additional-steps 100000
Optional preflight only: python scripts/train_smooth_balance.py --check-only

Both branches restart from the same saved policy/optimizer with the same seed:
  control: original reward, smoothness coefficient 0.005
  smooth:  same reward except smoothness coefficient 0.05 (experimental)
No action filter, observation change, horizon change or new reset randomization.
Training episodes remain 4 s; deterministic evaluation episodes last 30 s.
All new checkpoints/logs go to a NEW experiments/smooth_balance_<timestamp>/.
Existing model, environment, training scripts and playback are not replaced.

This is a one-seed, nominal-start experiment, not a robustness demonstration.
The new reward might help or might regress. No candidate is auto-installed.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

BASE_SMOOTHNESS_WEIGHT = 0.005


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SmoothnessReward(gym.Wrapper):
    """Add only the difference from the existing 0.005 action-change penalty."""

    def __init__(self, env, weight):
        super().__init__(env)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Smoothness weight must be finite and nonnegative")
        self.weight = float(weight)

    def step(self, action):
        applied = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        previous = self.env.unwrapped.previous_action.copy()
        obs, base_reward, terminated, truncated, info = self.env.step(applied)
        cost = float(np.mean(np.square(applied - previous)))
        reward = float(base_reward) - (self.weight - BASE_SMOOTHNESS_WEIGHT) * cost
        info = dict(info, original_reward=float(base_reward), action_change_cost=cost)
        return obs, float(reward), terminated, truncated, info


@contextmanager
def preserve_training_state(model):
    """Keep deterministic evaluation from disturbing training RNG/mode."""
    import torch
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    was_training = model.policy.training
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        model.policy.set_training_mode(was_training)


def evaluate(model, env_class, seconds, seed, csv_path):
    env = env_class(episode_seconds=seconds)
    rows, episode_return = [], 0.0
    try:
        with preserve_training_state(model):
            obs, _ = env.reset(seed=seed)
            xy0 = env.data.qpos[:2].copy()
            previous_targets = env.data.ctrl[env.lower_body_ids].copy()
            dt = float(env.model.opt.timestep * env.frame_skip)
            for step in range(1, env.max_episode_steps + 1):
                action, _ = model.predict(obs, deterministic=True)
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Non-finite policy action during evaluation")
                obs, reward, terminated, truncated, info = env.step(action)
                if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                    raise RuntimeError("Non-finite simulation state/reward during evaluation")
                targets = env.data.ctrl[env.lower_body_ids].copy()
                delta = env.data.qpos[:2] - xy0
                rows.append({
                    "time_s": step * dt,
                    "root_dx_m": float(delta[0]),
                    "root_dy_m": float(delta[1]),
                    "root_distance_m": float(np.linalg.norm(delta)),
                    "root_height_m": float(info["root_height"]),
                    "upright": float(info["upright_score"]),
                    "tilt_deg": float(np.degrees(np.arccos(np.clip(info["upright_score"], -1, 1)))),
                    "xy_speed_m_s": float(np.linalg.norm(env.data.qvel[:2])),
                    "angular_speed_rad_s": float(np.linalg.norm(env.data.qvel[3:6])),
                    "target_delta_rms_rad": float(np.sqrt(np.mean((targets - previous_targets) ** 2))),
                    "action_near_limit_fraction": float(np.mean(np.abs(action) >= .98)),
                })
                previous_targets = targets
                episode_return += float(reward)
                if terminated or truncated:
                    break
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        def series(key):
            return np.array([r[key] for r in rows])
        changes = series("target_delta_rms_rad")[1:]
        return {
            "duration_s": rows[-1]["time_s"], "steps": len(rows),
            "outcome": "fall threshold reached" if terminated else "time limit reached",
            "original_reward_return": episode_return,
            "max_displacement_m": float(series("root_distance_m").max()),
            "final_displacement_m": rows[-1]["root_distance_m"],
            "minimum_height_m": float(series("root_height_m").min()),
            "minimum_upright": float(series("upright").min()),
            "max_tilt_deg": float(series("tilt_deg").max()),
            "rms_xy_speed_m_s": float(np.sqrt(np.mean(series("xy_speed_m_s") ** 2))),
            "rms_angular_speed_rad_s": float(np.sqrt(np.mean(series("angular_speed_rad_s") ** 2))),
            "target_delta_rms_rad": float(np.sqrt(np.mean(changes ** 2))) if len(changes) else None,
            "mean_action_near_limit_fraction": float(series("action_near_limit_fraction").mean()),
            "trajectory_csv": str(csv_path),
        }
    finally:
        env.close()


def write_report(out, report):
    path = out / "training_summary.json"
    temp = out / "training_summary.tmp"
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
    records = report["evaluations"]
    if records:
        keys = ("branch", "additional_steps", "duration_s", "outcome", "max_displacement_m",
                "max_tilt_deg", "target_delta_rms_rad", "model_path")
        with (out / "comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)


def plot_report(out, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    panels = [
        ("duration_s", "Evaluation survival", "s"),
        ("max_displacement_m", "Maximum displacement (check survival first)", "m"),
        ("target_delta_rms_rad", "Applied target changes", "RMS rad per decision"),
        ("max_tilt_deg", "Maximum trunk tilt", "degrees"),
    ]
    reference = report["evaluations"][0]
    for branch in ("control", "smooth"):
        rows = [reference] + [r for r in report["evaluations"] if r["branch"] == branch]
        if len(rows) == 1:
            continue
        for ax, (key, title, units) in zip(axes.flat, panels):
            ax.plot([r["additional_steps"] for r in rows], [r[key] for r in rows], "o-", label=branch)
            ax.set(title=title, ylabel=units)
            ax.grid(alpha=.25)
            ax.legend()
    for ax in axes[-1]:
        ax.set_xlabel("Additional training steps")
    fig.suptitle("Matched continuation runs: original vs stronger smoothness reward")
    fig.tight_layout()
    fig.savefig(out / "training_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additional-steps", type=int, default=100000, help="Additional steps PER branch.")
    parser.add_argument("--eval-every", type=int, default=25000)
    parser.add_argument("--eval-seconds", type=float, default=30.0)
    parser.add_argument("--smoothness-weight", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.additional_steps < 1 or args.eval_every < 1:
        parser.error("Step counts must be positive")
    if not np.isfinite(args.eval_seconds) or args.eval_seconds < .04:
        parser.error("--eval-seconds must be finite and >= 0.04")
    if not np.isfinite(args.smoothness_weight) or args.smoothness_weight <= BASE_SMOOTHNESS_WEIGHT:
        parser.error("--smoothness-weight must exceed the original 0.005")
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    from t1_balance_env import T1BalanceEnv
    import t1_balance_env

    root = Path(__file__).resolve().parents[1]
    source = Path(t1_balance_env.__file__).resolve()
    checkpoint = root / "models/t1_balance_ppo.zip"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Place this script inside scripts/. Missing: {checkpoint}")
    if "reward -= 0.005 * smoothness_cost" not in source.read_text():
        raise RuntimeError("Expected original 0.005 smoothness penalty not found; no training started.")
    hashes = {str(p): digest(p) for p in (checkpoint, source)}
    for weight in (BASE_SMOOTHNESS_WEIGHT, args.smoothness_weight):
        test_env = SmoothnessReward(T1BalanceEnv(episode_seconds=4.0), weight)
        try:
            check_env(test_env, warn=True)
        finally:
            test_env.close()
    print("Both reward variants passed the environment check.", flush=True)
    if args.check_only:
        print("No training started; no project files changed.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = root / "experiments" / f"smooth_balance_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    # Snapshot inputs so both branches read precisely the same files.
    base_checkpoint = out / "original_policy.zip"
    shutil.copy2(checkpoint, base_checkpoint)
    shutil.copy2(source, out / "environment_snapshot.py")
    shutil.copy2(Path(__file__), out / "experiment_script.py")
    report = {
        "settings": vars(args), "input_sha256": hashes,
        "versions": {n: importlib.metadata.version(n) for n in ("mujoco", "stable-baselines3", "gymnasium", "numpy", "torch")},
        "training_horizon_s": 4.0,
        "weights": {"control": BASE_SMOOTHNESS_WEIGHT, "smooth": args.smoothness_weight},
        "notes": [
            "Both branches continue the same checkpoint and saved optimizer with the same seed.",
            "Only the action-change reward coefficient differs between branches.",
            "No output filter, new observations, reset perturbations, or horizon change in training.",
            "All evaluations use the original unmodified reward and a fresh home-state reset.",
            "Evaluation preserves training RNG state. One seed and one nominal start do not establish robustness.",
            "Check survival first. Failed short runs can have deceptively small displacement.",
            "Additional steps may exceed requests slightly to finish PPO rollout batches.",
            "The current MLP observation does not include previous action; its existing history-dependent reward is retained.",
            "No candidate is automatically installed; original playback/model remain untouched.",
        ],
        "evaluations": [],
    }
    reference_model = PPO.load(str(base_checkpoint), device="cpu")
    reference = evaluate(reference_model, T1BalanceEnv, args.eval_seconds, args.seed, out / "reference.csv")
    reference.update(branch="reference", additional_steps=0, model_path=str(base_checkpoint))
    report["evaluations"].append(reference)
    write_report(out, report)
    print(f"Reference: {reference['duration_s']:.2f} s, {reference['outcome']}, drift {reference['max_displacement_m']:.3f} m", flush=True)
    if reference["outcome"] != "time limit reached":
        raise RuntimeError(f"Reference did not complete evaluation horizon; no training started. Results: {out}")
    del reference_model

    for branch, weight in report["weights"].items():
        branch_dir = out / branch
        branch_dir.mkdir()
        env = Monitor(SmoothnessReward(T1BalanceEnv(episode_seconds=4.0), weight), filename=str(branch_dir / "training"))
        try:
            model = PPO.load(str(base_checkpoint), env=env, device="cpu", seed=args.seed, verbose=0)
            # The supplied project uses constant schedules; chunked learn calls
            # retain optimizer and environment state but restart progress schedules.
            # Refuse other schedules rather than silently change their behavior.
            for name, schedule in (("learning rate", model.lr_schedule), ("clip range", model.clip_range)):
                values = [float(schedule(p)) for p in (0., .25, .5, .75, 1.)]
                if not np.allclose(values, values[0], rtol=0, atol=1e-12):
                    raise RuntimeError(f"This experiment expects a constant {name} schedule.")
            model.set_logger(configure(str(branch_dir), ["csv"]))
            start_steps = int(model.num_timesteps)
            while model.num_timesteps - start_steps < args.additional_steps:
                added = int(model.num_timesteps - start_steps)
                chunk = min(args.eval_every, args.additional_steps - added)
                print(f"\n{branch}: training next ~{chunk:,} steps (completed {added:,})...", flush=True)
                model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
                added = int(model.num_timesteps - start_steps)
                saved = branch_dir / f"checkpoint_{added:09d}.zip"
                model.save(str(saved))
                metric = evaluate(model, T1BalanceEnv, args.eval_seconds, args.seed, branch_dir / f"evaluation_{added:09d}.csv")
                metric.update(branch=branch, additional_steps=added, total_model_steps=int(model.num_timesteps), model_path=str(saved))
                report["evaluations"].append(metric)
                write_report(out, report)
                print(f"  Evaluation: {metric['duration_s']:.2f} s, {metric['outcome']}, "
                      f"max drift={metric['max_displacement_m']:.3f} m, "
                      f"target change={metric['target_delta_rms_rad']:.4f} rad RMS", flush=True)
        finally:
            env.close()
    if any(digest(Path(p)) != value for p, value in hashes.items()):
        raise RuntimeError("An original input file changed during training (possibly another process).")
    report["original_input_files_unchanged"] = True
    try:
        plot_report(out, report)
    except Exception as exc:
        report["plot_error"] = str(exc)
        print("Plot failed; checkpoints and tabular results are still saved:", exc)
    write_report(out, report)
    print("\nCOMPARISON")
    print("branch | extra steps | duration(s) | max drift(m) | target change RMS(rad) | outcome")
    for row in report["evaluations"]:
        print(f"{row['branch']:9} | {row['additional_steps']:11,d} | {row['duration_s']:11.2f} | "
              f"{row['max_displacement_m']:12.3f} | {row['target_delta_rms_rad']:22.4f} | {row['outcome']}")
    print(f"\nResults: {out}\nOriginal checkpoint/environment unchanged. No candidate installed.")
    print("Share training_summary.json and training_comparison.png.")


if __name__ == "__main__":
    main()
