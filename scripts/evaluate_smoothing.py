"""Evaluate action smoothing WITHOUT retraining or editing the T1 environment.

Place beside diagnose_balance.py in the project's scripts/ folder.
Run: python scripts/evaluate_smoothing.py --seconds 30

alpha=1 reproduces the original controller. Smaller alpha adds smoothing AND
lag, so it may destabilize this policy. All runs use fresh home-state resets.
No successful setting is automatically installed or treated as robust.
"""

import argparse
import csv
import importlib.metadata
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from diagnose_balance import run_trial, sha256


class SmoothedPolicy:
    def __init__(self, policy, alpha):
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.policy = policy
        self.alpha = float(alpha)
        self.previous = None
        self.raw_actions = []

    def predict(self, observation, deterministic=True):
        raw, state = self.policy.predict(observation, deterministic=deterministic)
        raw = np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)
        if not np.all(np.isfinite(raw)):
            raise RuntimeError("Non-finite raw policy action")
        self.raw_actions.append(raw.copy())
        if self.previous is None:
            self.previous = np.zeros_like(raw)
        # Explicit bypass preserves the original controller at alpha=1.
        applied = raw.copy() if self.alpha == 1.0 else (
            self.alpha * raw + (1.0 - self.alpha) * self.previous
        )
        self.previous = applied.copy()
        return applied, state


def comparison_plot(trials, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        ("root_distance_m", "Distance from starting root position", "m"),
        ("root_tilt_deg", "Trunk tilt from vertical", "degrees"),
        ("target_delta_rms_rad", "Applied target change per policy step", "RMS rad"),
        ("root_xy_speed_m_s", "Horizontal root speed", "m/s"),
    ]
    for label, rows in trials.items():
        t = [r["time_s"] for r in rows]
        for ax, (key, title, unit) in zip(axes.flat, panels):
            ax.plot(t, [r[key] for r in rows], label=label, linewidth=.85, alpha=.85)
            ax.set(title=title, ylabel=unit)
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Simulated time (s)")
    fig.suptitle("Same saved policy with different action smoothing; lines end at termination")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0, 0.9, 0.75, 0.5])
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < .02:
        parser.error("--seconds must be finite and >= 0.02")
    if any(not np.isfinite(a) or not 0 < a <= 1 for a in args.alphas):
        parser.error("All alphas must be in (0, 1]")
    # Always include the unchanged controller first.
    alphas = list(dict.fromkeys([1.0] + args.alphas))

    from stable_baselines3 import PPO
    from t1_balance_env import T1BalanceEnv
    import t1_balance_env
    import diagnose_balance

    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "models/t1_balance_ppo.zip"
    env_source = Path(t1_balance_env.__file__).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Put this file inside scripts/. Missing checkpoint: {checkpoint}")
    hashes = {str(p): sha256(p) for p in (checkpoint, env_source, Path(diagnose_balance.__file__))}
    policy = PPO.load(str(checkpoint), device="cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = root / "logs" / f"smoothing_evaluation_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "settings": {"seconds": args.seconds, "seed": args.seed, "alphas": alphas, "load_threshold_N": 1.0},
        "sha256": hashes,
        "versions": {n: importlib.metadata.version(n) for n in ("mujoco", "stable-baselines3", "numpy", "gymnasium")},
        "formula": "applied[t] = alpha*raw[t] + (1-alpha)*applied[t-1]; applied[-1]=0",
        "notes": [
            "One fresh deterministic home-state rollout per alpha; no disturbance or robustness test.",
            "Smoothing changes the feedback loop; it can make an otherwise upright policy fall.",
            "No model or environment modifications and no training.",
            "Compare drift, tilt and motion only after checking survival duration.",
            "A short failed run can have small displacement simply because it ended early.",
            "Contact metrics are sampled instantaneous estimates, not cumulative slip distance.",
            "Target-delta statistics exclude the initial home-to-first-action transition.",
        ],
        "trials": {},
    }
    trials, table = {}, []
    for index, alpha in enumerate(alphas):
        label = f"trial_{index}_alpha_{alpha:g}"
        wrapper = SmoothedPolicy(policy, alpha)
        rows, summary = run_trial(T1BalanceEnv, wrapper, args.seconds, args.seed, 1.0, out, label)
        summary["alpha"] = alpha
        raw = np.stack(wrapper.raw_actions)
        summary["raw_action_delta_rms_excluding_first"] = (
            float(np.sqrt(np.mean(np.diff(raw, axis=0) ** 2))) if len(raw) > 1 else None
        )
        summary["raw_action_near_limit_fraction"] = float(np.mean(np.abs(raw) >= .98))
        # Keep unsmoothed commands too, so reduced applied saturation is not
        # mistaken for the neural policy itself becoming less saturated.
        np.savez_compressed(out / f"{label}_raw_actions.npz", raw_actions=raw)
        common_rows = [r for r in rows if r["time_s"] <= min(4., args.seconds) + 1e-9]
        summary["first_4s_or_until_fall"] = {
            "duration_s": common_rows[-1]["time_s"],
            "maximum_displacement_m": max(r["root_distance_m"] for r in common_rows),
            "rms_xy_speed_m_s": float(np.sqrt(np.mean([r["root_xy_speed_m_s"] ** 2 for r in common_rows]))),
        }
        report["trials"][label] = summary
        trials[f"alpha={alpha:g}"] = rows
        entry = {
            "alpha": alpha,
            "duration_s": summary["duration_s"],
            "outcome": summary["outcome"],
            "max_displacement_m": summary["maximum_displacement_m"],
            "max_tilt_deg": max(r["root_tilt_deg"] for r in rows),
            "target_delta_rms_rad": summary["target_delta_rms_rad_excluding_first"],
            "left_slip_mean_m_s": summary["left"]["loaded_contact_slip_mean_m_s"],
            "right_slip_mean_m_s": summary["right"]["loaded_contact_slip_mean_m_s"],
        }
        table.append(entry)
        # Preserve completed trials even if a later trial errors or is stopped.
        (out / "smoothing_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    try:
        comparison_plot(trials, out / "smoothing_comparison.png")
    except Exception as exc:
        report["plot_error"] = str(exc)
        print("Plot failed; CSV and JSON are still available:", exc)
    if any(sha256(Path(p)) != digest for p, digest in hashes.items()):
        raise RuntimeError("An input file changed during the run (possibly another process).")
    report["input_files_unchanged"] = True
    (out / "smoothing_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print("\nCOMPARISON (check survival before comparing drift)")
    print("alpha | time(s) | outcome | max drift(m) | max tilt(deg) | target delta RMS(rad)")
    for item in table:
        delta = item["target_delta_rms_rad"]
        delta_text = f"{delta:.4f}" if delta is not None else "n/a"
        print(f"{item['alpha']:5.2f} | {item['duration_s']:7.2f} | {item['outcome']} | "
              f"{item['max_displacement_m']:.3f} | {item['max_tilt_deg']:.2f} | {delta_text}")
    print(f"\nSaved results: {out}")
    print("No setting was installed. Your normal playback still uses the original policy.")
    print("Share smoothing_summary.json and smoothing_comparison.png for review.")


if __name__ == "__main__":
    main()
