"""Matched stance-width reward experiment, starting from the selected checkpoint.

Put beside check_stance.py and train_smooth_balance.py in scripts/.
Run: python scripts/train_stance_balance.py --additional-steps 100000
Both branches resume the selected policy and optimizer with the same seed.
control adds no reward term; stance subtracts
    weight * min(((horizontal_foot_separation - home_separation)/0.10)**2, 4).
Default weight=0.25: maximum additional penalty is 1 reward point per step.
Only this reward term differs. Both train with the original 4-second horizon,
57 observations and 12 actions; no action filter, friction/gain/limit changes.
Evaluate every ~25k added steps for 60 seconds using the original environment.
All outputs are new; no existing checkpoint or playback setting is replaced.
"""

import argparse
import csv
import importlib.metadata
import json
import shutil
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from check_stance import StanceProbe, DEFAULT_MODEL
from train_smooth_balance import digest, preserve_training_state


def stance_cost(width, home_width):
    return float(min(((width - home_width) / .10) ** 2, 4.0))


class StanceReward(gym.Wrapper):
    def __init__(self, env, weight):
        super().__init__(env)
        self.weight = float(weight)
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Invalid stance reward weight")
        self.probe = StanceProbe(env.model)
        self.home_width = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.home_width = self.probe.sample(self.env.unwrapped.data)["foot_separation_xy_m"]
        return obs, info

    def step(self, action):
        if self.home_width is None:
            raise RuntimeError("Reset the environment before stepping")
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        width = self.probe.sample(self.env.unwrapped.data)["foot_separation_xy_m"]
        cost = stance_cost(width, self.home_width)
        reward = float(base_reward) - self.weight * cost
        info = dict(info, original_reward=float(base_reward), stance_width_m=width,
                    home_width_m=self.home_width, stance_cost=cost)
        return obs, reward, terminated, truncated, info


def hip_settings(model, probe):
    result = {}
    for side, (aid, jid, qadr) in probe.hips.items():
        if not np.isclose(model.actuator_gear[aid, 0], 1.):
            raise RuntimeError("This stance report expects unit-gear hip-roll actuators.")
        item = {
            "actuator_gainprm": model.actuator_gainprm[aid].tolist(),
            "actuator_biasprm": model.actuator_biasprm[aid].tolist(),
            "actuator_forcelimited": bool(model.actuator_forcelimited[aid]),
            "actuator_forcerange": model.actuator_forcerange[aid].tolist(),
            "joint_range_rad": model.jnt_range[jid].tolist(),
            "control_range_rad": model.actuator_ctrlrange[aid].tolist(),
        }
        if hasattr(model, "jnt_actfrclimited"):
            item["joint_actuator_force_limited"] = bool(model.jnt_actfrclimited[jid])
            item["joint_actuator_force_range_Nm"] = model.jnt_actfrcrange[jid].tolist()
        result[side] = item
    return result


def evaluate(model, env_class, seconds, seed, csv_path):
    env = env_class(episode_seconds=seconds)
    rows = []
    try:
        with preserve_training_state(model):
            obs, info = env.reset(seed=seed)
            probe = StanceProbe(env.model)
            settings = hip_settings(env.model, probe)
            initial = probe.sample(env.data)
            initial_width = initial["foot_separation_xy_m"]
            xy0 = env.data.qpos[:2].copy()
            previous_targets = env.data.ctrl[env.lower_body_ids].copy()
            dt = float(env.model.opt.timestep * env.frame_skip)
            total_reward = 0.
            for step in range(1, env.max_episode_steps + 1):
                action, _ = model.predict(obs, deterministic=True)
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Non-finite action during evaluation")
                obs, reward, terminated, truncated, info = env.step(action)
                if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                    raise RuntimeError("Non-finite evaluation observation/reward")
                targets = env.data.ctrl[env.lower_body_ids].copy()
                row = {
                    "time_s": step * dt,
                    "root_displacement_m": float(np.linalg.norm(env.data.qpos[:2] - xy0)),
                    "root_height_m": float(info["root_height"]),
                    "upright": float(info["upright_score"]),
                    "target_delta_rms_rad": float(np.sqrt(np.mean((targets - previous_targets)**2))),
                    **probe.sample(env.data),
                }
                # Forces are from the last physics substep, not recomputed on
                # live state; this leaves the rollout entirely unmodified.
                for side, (aid, jid, qadr) in probe.hips.items():
                    dof = int(env.model.jnt_dofadr[jid])
                    row[f"{side}_hip_actuator_force"] = float(env.data.actuator_force[aid])
                    row[f"{side}_hip_actuator_torque_Nm"] = float(env.data.qfrc_actuator[dof])
                rows.append(row)
                previous_targets = targets
                total_reward += float(reward)
                if terminated or truncated:
                    break
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        def series(key):
            return np.array([r[key] for r in rows])
        final = rows[-1]
        late = [r for r in rows if r["time_s"] >= max(0., final["time_s"] - 10.)]
        changes = series("target_delta_rms_rad")[1:]
        summary = {
            "duration_s": final["time_s"], "steps": len(rows),
            "outcome": "fall threshold reached" if terminated else "time limit reached",
            "original_reward_return": total_reward,
            "initial_foot_separation_m": initial_width,
            "final_foot_separation_m": final["foot_separation_xy_m"],
            "maximum_foot_separation_m": max(initial_width, float(series("foot_separation_xy_m").max())),
            "max_absolute_width_error_m": float(np.max(np.abs(series("foot_separation_xy_m") - initial_width))),
            "final_width_growth_m": final["foot_separation_xy_m"] - initial_width,
            "last_window_duration_s": late[-1]["time_s"] - late[0]["time_s"],
            "last_window_width_slope_m_s": float(np.polyfit([r["time_s"] for r in late], [r["foot_separation_xy_m"] for r in late], 1)[0]) if len(late) > 1 else None,
            "max_root_displacement_m": float(series("root_displacement_m").max()),
            "minimum_root_height_m": float(series("root_height_m").min()),
            "minimum_upright": float(series("upright").min()),
            "target_delta_rms_rad": float(np.sqrt(np.mean(changes**2))) if len(changes) else None,
            "hip_model_settings": settings,
            "trajectory_csv": str(csv_path),
        }
        for side, (aid, jid, qadr) in probe.hips.items():
            forces = series(f"{side}_hip_actuator_force")
            item = {
                "final_actual_rad": final[f"{side}_hip_roll_rad"],
                "final_target_rad": final[f"{side}_hip_roll_target_rad"],
                "minimum_actuator_force": float(forces.min()),
                "maximum_actuator_force": float(forces.max()),
                "max_abs_joint_actuator_torque_Nm": float(np.abs(series(f"{side}_hip_actuator_torque_Nm")).max()),
                "fraction_near_actuator_force_limit": None,
            }
            if env.model.actuator_forcelimited[aid]:
                low, high = env.model.actuator_forcerange[aid]
                margin = .01 * (high-low)
                item["fraction_near_actuator_force_limit"] = float(np.mean((forces <= low+margin) | (forces >= high-margin)))
            summary[side + "_hip"] = item
        return summary
    finally:
        env.close()


def save_report(out, report):
    temp = out / "stance_training_summary.tmp"
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(out / "stance_training_summary.json")
    fields = ("branch", "additional_steps", "duration_s", "outcome", "final_foot_separation_m",
              "max_absolute_width_error_m", "last_window_width_slope_m_s", "max_root_displacement_m", "target_delta_rms_rad", "model_path")
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["evaluations"])


def plot_report(out, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fields = (("duration_s", "Survival duration", "s"),
              ("max_absolute_width_error_m", "Maximum deviation from home foot separation", "m"),
              ("max_root_displacement_m", "Maximum root displacement", "m"),
              ("target_delta_rms_rad", "Target changes between decisions", "RMS rad"))
    ref = report["evaluations"][0]
    for branch in ("control", "stance"):
        rows = [ref] + [r for r in report["evaluations"] if r["branch"] == branch]
        for ax, (key, title, units) in zip(axes.flat, fields):
            ax.plot([r["additional_steps"] for r in rows], [r[key] for r in rows], "o-", label=branch)
            ax.set(title=title, ylabel=units)
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend()
    for ax in axes[-1]:
        ax.set_xlabel("Additional training steps")
    fig.suptitle("Stance reward experiment: check survival before comparing width/drift")
    fig.tight_layout()
    fig.savefig(out / "stance_training_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--additional-steps", type=int, default=100000, help="Additional steps PER branch")
    parser.add_argument("--eval-every", type=int, default=25000)
    parser.add_argument("--eval-seconds", type=float, default=60.)
    parser.add_argument("--stance-weight", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.additional_steps < 1 or args.eval_every < 1:
        parser.error("Step counts must be positive")
    if not np.isfinite(args.eval_seconds) or args.eval_seconds < .04:
        parser.error("Evaluation duration must be finite and >= .04 s")
    if not np.isfinite(args.stance_weight) or args.stance_weight <= 0:
        parser.error("Stance weight must be positive and finite")
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    from t1_balance_env import T1BalanceEnv
    import t1_balance_env
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.model).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    source = Path(t1_balance_env.__file__).resolve()
    hashes = {str(p): digest(p) for p in (checkpoint, source)}
    for weight in (0., args.stance_weight):
        env = StanceReward(T1BalanceEnv(episode_seconds=4.), weight)
        try:
            check_env(env, warn=True)
        finally:
            env.close()
    print("Both stance reward variants passed the environment check.", flush=True)
    if args.check_only:
        print("No training started; no files changed.")
        return
    out = root / "experiments" / ("stance_balance_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    base = out / "starting_policy.zip"
    shutil.copy2(checkpoint, base)
    shutil.copy2(source, out / "environment_snapshot.py")
    shutil.copy2(Path(__file__), out / "experiment_script.py")
    report = {
        "settings": vars(args), "input_sha256": hashes, "training_horizon_s": 4.,
        "versions": {n: importlib.metadata.version(n) for n in ("mujoco", "stable-baselines3", "gymnasium", "numpy", "torch")},
        "reward_formula": "original_reward - weight * min(((foot_separation-home_separation)/0.10)^2, 4)",
        "notes": [
            "Control weight=0, stance weight as specified. Both resume identical saved weights/optimizer and seed.",
            "Home separation is measured after reset, from the two foot-body origins.",
            "Only reward differs; original observations, action scale, 4 s training horizon and all physics parameters are retained.",
            "Evaluations use the original reward, no action filter, same home state, and preserve training RNG/mode.",
            "Force readings are from the last physics substep. Near-limit uses a margin of 1% of actuator force-range width.",
            "Joint-level force clamps are reported separately; actuator-force saturation alone does not cover them.",
            "One-seed nominal-start screening only. Failed short runs may show deceptively small width errors.",
            "No candidate is automatically installed. Additional steps may round up to complete rollouts.",
        ], "evaluations": [],
    }
    reference_model = PPO.load(str(base), device="cpu")
    ref = evaluate(reference_model, T1BalanceEnv, args.eval_seconds, args.seed, out / "reference.csv")
    ref.update(branch="reference", additional_steps=0, model_path=str(base))
    report["evaluations"].append(ref)
    save_report(out, report)
    print(f"Reference: {ref['duration_s']:.2f} s, final foot separation {ref['final_foot_separation_m']:.3f} m", flush=True)
    if ref["outcome"] != "time limit reached":
        raise RuntimeError(f"Reference failed the evaluation horizon; no training started. Results: {out}")
    del reference_model
    for branch, weight in (("control", 0.), ("stance", args.stance_weight)):
        directory = out / branch
        directory.mkdir()
        env = Monitor(StanceReward(T1BalanceEnv(episode_seconds=4.), weight), filename=str(directory / "training"))
        try:
            model = PPO.load(str(base), env=env, device="cpu", seed=args.seed, verbose=0)
            for name, schedule in (("learning rate", model.lr_schedule), ("clip range", model.clip_range)):
                values = [float(schedule(p)) for p in (0., .25, .5, .75, 1.)]
                if not np.allclose(values, values[0], rtol=0, atol=1e-12):
                    raise RuntimeError(f"Expected constant {name} schedule")
            model.set_logger(configure(str(directory), ["csv"]))
            start = int(model.num_timesteps)
            while model.num_timesteps - start < args.additional_steps:
                added = int(model.num_timesteps - start)
                chunk = min(args.eval_every, args.additional_steps - added)
                print(f"\n{branch}: training ~{chunk:,} more steps ({added:,} completed)...", flush=True)
                model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
                added = int(model.num_timesteps - start)
                saved = directory / f"checkpoint_{added:09d}.zip"
                model.save(str(saved))
                metric = evaluate(model, T1BalanceEnv, args.eval_seconds, args.seed, directory / f"evaluation_{added:09d}.csv")
                metric.update(branch=branch, additional_steps=added, total_model_steps=int(model.num_timesteps), model_path=str(saved))
                report["evaluations"].append(metric)
                save_report(out, report)
                print(f"  {metric['duration_s']:.2f} s, {metric['outcome']}; "
                      f"width={metric['final_foot_separation_m']:.3f} m, "
                      f"max width error={metric['max_absolute_width_error_m']:.3f} m, "
                      f"root drift={metric['max_root_displacement_m']:.3f} m", flush=True)
        finally:
            env.close()
    if any(digest(Path(p)) != value for p, value in hashes.items()):
        raise RuntimeError("An original input changed during the experiment (possibly another process)")
    report["original_input_files_unchanged"] = True
    try:
        plot_report(out, report)
    except Exception as exc:
        report["plot_error"] = str(exc)
    save_report(out, report)
    print("\nCOMPARISON: branch | added steps | survival(s) | final width(m) | max width error(m) | root drift(m) | outcome")
    for row in report["evaluations"]:
        print(f"{row['branch']:9} | {row['additional_steps']:11,d} | {row['duration_s']:11.2f} | "
              f"{row['final_foot_separation_m']:.3f} | {row['max_absolute_width_error_m']:.3f} | "
              f"{row['max_root_displacement_m']:.3f} | {row['outcome']}")
    print(f"\nResults: {out}\nNo candidate installed; original files unchanged.")
    print("Share stance_training_summary.json and stance_training_comparison.png.")


if __name__ == "__main__":
    main()
