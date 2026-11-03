"""Measure stance widening for the selected T1 checkpoint; no training/edits.

Place in scripts/ and run: python scripts/check_stance.py
Defaults to the control checkpoint at 50,176 extra steps and 60 s evaluation.
Foot separation uses model foot-body origins, not the inner edges of the soles.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_MODEL = "experiments/smooth_balance_20261027_221853_406565/control/checkpoint_000050176.zip"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StanceProbe:
    def __init__(self, model):
        self.model = model
        self.shadow = mujoco.MjData(model)
        self.feet, self.hips = {}, {}
        self.rotation = np.zeros(9)
        for side in ("left", "right"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot_link")
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side.title()}_Hip_Roll")
            if bid < 0 or aid < 0 or model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
                raise RuntimeError(f"Could not resolve {side} foot and hip-roll joint actuator.")
            jid = int(model.actuator_trnid[aid, 0])
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise RuntimeError("Expected a hinge hip-roll joint.")
            self.feet[side] = bid
            self.hips[side] = (aid, jid, int(model.jnt_qposadr[jid]))

    def sample(self, live):
        # Kinematics only on separate data: no changes to the live integration.
        self.shadow.qpos[:] = live.qpos
        self.shadow.mocap_pos[:] = live.mocap_pos
        self.shadow.mocap_quat[:] = live.mocap_quat
        mujoco.mj_kinematics(self.model, self.shadow)
        positions = {s: self.shadow.xpos[b].copy() for s, b in self.feet.items()}
        delta = positions["left"] - positions["right"]
        mujoco.mju_quat2Mat(self.rotation, live.qpos[3:7])
        lateral = self.rotation.reshape(3, 3)[:2, 1]
        lateral = lateral / max(float(np.linalg.norm(lateral)), 1e-12)
        row = {
            "foot_separation_xy_m": float(np.linalg.norm(delta[:2])),
            "lateral_separation_m": float(abs(np.dot(delta[:2], lateral))),
        }
        for side, pos in positions.items():
            for axis, value in zip("xyz", pos):
                row[f"{side}_foot_{axis}_m"] = float(value)
            aid, jid, qadr = self.hips[side]
            row[f"{side}_hip_roll_rad"] = float(live.qpos[qadr])
            row[f"{side}_hip_roll_target_rad"] = float(live.ctrl[aid])
        return row

    def limits(self):
        return {
            side: {
                "joint_limited": bool(self.model.jnt_limited[jid]),
                "joint_range_rad": self.model.jnt_range[jid].tolist(),
                "control_limited": bool(self.model.actuator_ctrllimited[aid]),
                "control_range_rad": self.model.actuator_ctrlrange[aid].tolist(),
            } for side, (aid, jid, _) in self.hips.items()
        }


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [r["time_s"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    panels = [
        ("Foot-body origin separation", "m", [("foot_separation_xy_m", "horizontal distance"), ("lateral_separation_m", "lateral width")]),
        ("Individual foot displacement along world Y", "m", [("left_foot_dy_m", "left"), ("right_foot_dy_m", "right")]),
        ("Hip-roll angles and commanded targets", "rad", [("left_hip_roll_rad", "left actual"), ("left_hip_roll_target_rad", "left target"), ("right_hip_roll_rad", "right actual"), ("right_hip_roll_target_rad", "right target")]),
        ("Root height", "m", [("root_height_m", "height")]),
    ]
    for ax, (title, unit, lines) in zip(axes.flat, panels):
        for key, label in lines:
            ax.plot(t, [r[key] for r in rows], "--" if "target" in key else "-", label=label, linewidth=1.)
        ax.set(title=title, ylabel=unit)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Simulated time (s)")
    fig.suptitle("Selected checkpoint: stance stability")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Absolute path or path relative to the project root.")
    parser.add_argument("--seconds", type=float, default=60.)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < .04:
        parser.error("--seconds must be finite and >= 0.04")
    from stable_baselines3 import PPO
    from t1_balance_env import T1BalanceEnv
    import t1_balance_env
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.model).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint}")
    env_path = Path(t1_balance_env.__file__).resolve()
    hashes = {str(p): digest(p) for p in (checkpoint, env_path)}
    model = PPO.load(str(checkpoint), device="cpu")
    env = T1BalanceEnv(episode_seconds=args.seconds)
    out = root / "logs" / ("stance_check_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        obs, info = env.reset(seed=args.seed)
        probe = StanceProbe(env.model)
        initial = probe.sample(env.data)
        root_start = env.data.qpos[:2].copy()
        dt = float(env.frame_skip * env.model.opt.timestep)
        def record(time_s, info):
            row = {"time_s": time_s, **probe.sample(env.data)}
            row["root_height_m"] = float(info["root_height"])
            row["upright"] = float(info["upright_score"])
            row["root_displacement_m"] = float(np.linalg.norm(env.data.qpos[:2] - root_start))
            for side in ("left", "right"):
                for axis in "xy":
                    key = f"{side}_foot_{axis}_m"
                    row[f"{side}_foot_d{axis}_m"] = row[key] - initial[key]
            rows.append(row)
            return row
        def print_row(row):
            print(f"{row['time_s']:6.1f} | {row['foot_separation_xy_m']:.3f} | "
                  f"{row['lateral_separation_m']:.3f} | {row['root_height_m']:.3f} | "
                  f"{row['root_displacement_m']:.3f}", flush=True)
        print(f"Selected checkpoint: {checkpoint}", flush=True)
        print("Time(s) | foot distance(m) | lateral width(m) | root height(m) | root drift(m)")
        print_row(record(0., info))
        for step in range(1, env.max_episode_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            if not np.all(np.isfinite(action)):
                raise RuntimeError("Non-finite policy action")
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                raise RuntimeError("Non-finite simulation observation/reward")
            row = record(step * dt, info)
            if step % max(1, round(5. / dt)) == 0 or terminated or truncated:
                print_row(row)
            if terminated or truncated:
                break
        final = rows[-1]
        window = [r for r in rows if r["time_s"] >= max(0., final["time_s"] - 10.)]
        slope = float(np.polyfit([r["time_s"] for r in window], [r["lateral_separation_m"] for r in window], 1)[0])
        summary = {
            "model_path": str(checkpoint), "input_sha256": hashes,
            "seed": args.seed, "duration_s": final["time_s"],
            "outcome": "fall threshold reached" if terminated else "time limit reached",
            "initial_horizontal_foot_separation_m": initial["foot_separation_xy_m"],
            "final_horizontal_foot_separation_m": final["foot_separation_xy_m"],
            "maximum_horizontal_foot_separation_m": max(r["foot_separation_xy_m"] for r in rows),
            "initial_lateral_width_m": initial["lateral_separation_m"],
            "final_lateral_width_m": final["lateral_separation_m"],
            "lateral_width_growth_m": final["lateral_separation_m"] - initial["lateral_separation_m"],
            "last_window_duration_s": window[-1]["time_s"] - window[0]["time_s"],
            "last_window_lateral_width_slope_m_s": slope,
            "max_root_displacement_m": max(r["root_displacement_m"] for r in rows),
            "minimum_root_height_m": min(r["root_height_m"] for r in rows),
            "minimum_upright": min(r["upright"] for r in rows),
            "final_hip_roll": {s: {"actual_rad": final[f"{s}_hip_roll_rad"], "target_rad": final[f"{s}_hip_roll_target_rad"]} for s in ("left", "right")},
            "hip_roll_limits": probe.limits(),
            "notes": [
                "Foot distances use body origins, not sole-edge clearance.",
                "Lateral width is projected on the trunk lateral axis in the horizontal plane.",
                "The last-window slope is descriptive; no automatic success threshold is applied.",
                "One nominal-start rollout, no disturbances; input files are not modified.",
            ],
        }
        with (out / "stance.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        try:
            plot(rows, out / "stance_plot.png")
        except Exception as exc:
            summary["plot_error"] = str(exc)
        if any(digest(Path(p)) != value for p, value in hashes.items()):
            raise RuntimeError("An input file changed during the run (possibly another process).")
        summary["input_files_unchanged"] = True
        (out / "stance_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print("\nSTANCE SUMMARY\n" + json.dumps(summary, indent=2, allow_nan=False))
        print(f"\nSaved results: {out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
