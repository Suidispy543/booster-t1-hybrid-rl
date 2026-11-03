"""Read-only T1 controller diagnostics. Put this file in the project's scripts/.

Run: python scripts/diagnose_balance.py --seconds 30
Creates a new logs/balance_diagnostics_<timestamp>/ directory, never trains/saves
a policy, and never edits the environment. Requires the existing project venv.

Foot origins are ankle-roll body origins, NOT sole clearance. Loaded means total
normal force > 1 N by default; unloaded does not prove that a foot is airborne.
Contact slip is normal-force-weighted tangential relative velocity at loaded
contact points. These are instantaneous estimates sampled at the policy rate,
not integrated slip distance or a definitive diagnosis of walking.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ContactProbe:
    """Recompute contacts on copied integration state, leaving live data alone."""

    def __init__(self, model, load_threshold=1.0):
        self.model = model
        self.data = mujoco.MjData(model)
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mujoco.mj_stateSize(model, self.spec))
        self.jac1 = np.zeros((3, model.nv))
        self.jac2 = np.zeros((3, model.nv))
        self.wrench = np.zeros(6)
        self.threshold = load_threshold
        self.feet = {}
        self.geom_side = {}
        self.floor = {
            g for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
            and model.geom_bodyid[g] == 0
        }
        if not self.floor:
            raise RuntimeError("Expected a world-attached plane floor; none found.")
        for side in ("left", "right"):
            name = f"{side.title()}_Ankle_Roll"
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0 or model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
                raise RuntimeError(f"Expected direct joint actuator {name}.")
            jid = int(model.actuator_trnid[aid, 0])
            bid = int(model.jnt_bodyid[jid])
            self.feet[side] = bid
            for gid in range(model.ngeom):
                ancestor = int(model.geom_bodyid[gid])
                while ancestor and ancestor != bid:
                    ancestor = int(model.body_parentid[ancestor])
                if ancestor == bid:
                    self.geom_side[gid] = side
        if not all(side in self.geom_side.values() for side in self.feet):
            raise RuntimeError("No geometries found under one of the foot bodies.")

    def sample(self, live):
        m, d = self.model, self.data
        mujoco.mj_getState(m, live, self.state, self.spec)
        mujoco.mj_setState(m, d, self.state, self.spec)
        mujoco.mj_forward(m, d)
        force = {s: 0.0 for s in self.feet}
        weighted = {s: 0.0 for s in self.feet}
        count = {s: 0 for s in self.feet}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            foot = g2 if g1 in self.floor else g1 if g2 in self.floor else -1
            side = self.geom_side.get(foot)
            if side is None or c.efc_address < 0:
                continue
            mujoco.mj_contactForce(m, d, i, self.wrench)
            normal_force = max(0.0, float(self.wrench[0]))
            if normal_force <= 1e-9:
                continue
            mujoco.mj_jac(m, d, self.jac1, None, c.pos, int(m.geom_bodyid[g1]))
            mujoco.mj_jac(m, d, self.jac2, None, c.pos, int(m.geom_bodyid[g2]))
            relative_velocity = (self.jac2 - self.jac1) @ d.qvel
            normal = c.frame[:3]
            tangent = relative_velocity - normal * np.dot(normal, relative_velocity)
            force[side] += normal_force
            weighted[side] += normal_force * float(np.linalg.norm(tangent))
            count[side] += 1
        result = {}
        for side, bid in self.feet.items():
            loaded = force[side] > self.threshold
            result[f"{side}_normal_force_N"] = force[side]
            result[f"{side}_loaded"] = int(loaded)
            result[f"{side}_force_contacts"] = count[side]
            result[f"{side}_contact_slip_m_s"] = (
                weighted[side] / force[side] if loaded else float("nan")
            )
            for axis, value in zip("xyz", d.xpos[bid]):
                result[f"{side}_origin_{axis}_m"] = float(value)
        return result

    def mapping(self):
        return {
            side: {
                "body": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid),
                "body_id": bid,
                "geom_ids": [g for g, s in self.geom_side.items() if s == side],
            }
            for side, bid in self.feet.items()
        }


def summarize(rows, total_return, terminated, dt):
    def series(key):
        return np.array([r[key] for r in rows], dtype=float)
    summary = {
        "duration_s": rows[-1]["time_s"],
        "steps": len(rows),
        "outcome": "fall threshold reached" if terminated else "time limit reached",
        "return": total_return,
        "sample_period_s": dt,
        "minimum_height_m": float(series("root_height_m").min()),
        "minimum_upright": float(series("upright").min()),
        "maximum_displacement_m": float(series("root_distance_m").max()),
        "final_displacement_m": rows[-1]["root_distance_m"],
        "rms_horizontal_speed_m_s": rms(series("root_xy_speed_m_s")),
        "rms_root_angular_speed_rad_s": rms(series("root_angular_speed_rad_s")),
        "action_delta_rms_excluding_first": rms(series("action_delta_rms")[1:]) if len(rows) > 1 else None,
        "target_delta_rms_rad_excluding_first": rms(series("target_delta_rms_rad")[1:]) if len(rows) > 1 else None,
        "mean_action_near_limit_fraction": float(series("action_near_limit_fraction").mean()),
    }
    for side in ("left", "right"):
        slip = series(f"{side}_contact_slip_m_s")
        slip = slip[np.isfinite(slip)]
        loaded = series(f"{side}_loaded") > 0
        summary[side] = {
            "loaded_sample_fraction": float(loaded.mean()),
            "load_state_transitions": int(np.count_nonzero(loaded[1:] != loaded[:-1])),
            "loaded_contact_slip_mean_m_s": float(slip.mean()) if len(slip) else None,
            "loaded_contact_slip_p95_m_s": float(np.percentile(slip, 95)) if len(slip) else None,
            "foot_origin_z_range_m": float(np.ptp(series(f"{side}_origin_z_m"))),
        }
    return summary


def run_trial(env_class, policy, seconds, seed, threshold, out, label):
    env = env_class(episode_seconds=seconds)
    rows, total_return = [], 0.0
    try:
        obs, initial = env.reset(seed=seed)
        probe = ContactProbe(env.model, threshold)
        initial_contact = probe.sample(env.data)
        start_xy = env.data.qpos[:2].copy()
        previous_action = np.zeros(env.action_space.shape, dtype=np.float32)
        previous_targets = env.data.ctrl[env.lower_body_ids].copy()
        dt = float(env.frame_skip * env.model.opt.timestep)
        print(f"\n{label}: foot mapping {probe.mapping()}", flush=True)
        for step in range(1, env.max_episode_steps + 1):
            action = previous_action * 0 if policy is None else policy.predict(obs, deterministic=True)[0]
            action = np.clip(np.asarray(action, dtype=np.float32), env.action_space.low, env.action_space.high)
            if not np.all(np.isfinite(action)):
                raise RuntimeError("Non-finite action.")
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
                raise RuntimeError("Non-finite observation/reward.")
            total_return += float(reward)
            xy = env.data.qpos[:2] - start_xy
            targets = env.data.ctrl[env.lower_body_ids].copy()
            row = {
                "time_s": step * dt,
                "root_dx_m": float(xy[0]), "root_dy_m": float(xy[1]),
                "root_distance_m": float(np.linalg.norm(xy)),
                "root_height_m": float(info["root_height"]),
                "upright": float(info["upright_score"]),
                "root_tilt_deg": float(np.degrees(np.arccos(np.clip(info["upright_score"], -1, 1)))),
                "root_xy_speed_m_s": float(np.linalg.norm(env.data.qvel[:2])),
                "root_angular_speed_rad_s": float(np.linalg.norm(env.data.qvel[3:6])),
                "action_delta_rms": rms(action - previous_action),
                "target_delta_rms_rad": rms(targets - previous_targets),
                "action_near_limit_fraction": float(np.mean(np.abs(action) >= 0.98)),
                "reward": float(reward),
                **probe.sample(env.data),
            }
            for side in ("left", "right"):
                for axis in "xy":
                    key = f"{side}_origin_{axis}_m"
                    row[f"{side}_origin_d{axis}_m"] = row[key] - initial_contact[key]
            for i, aid in enumerate(env.lower_body_ids):
                name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
                row[f"action_{name}"] = float(action[i])
                row[f"target_rad_{name}"] = float(targets[i])
            rows.append(row)
            previous_action = action.copy()
            previous_targets = targets
            if step % max(1, round(5 / dt)) == 0:
                print(f"  {step * dt:.1f} s: displacement={row['root_distance_m']:.3f} m", flush=True)
            if terminated or truncated:
                break
        summary = summarize(rows, total_return, terminated, dt)
        summary["foot_mapping"] = probe.mapping()
        summary["initial_root_height_m"] = initial["root_height"]
        with (out / f"{label}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return rows, summary
    finally:
        env.close()


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [r["time_s"] for r in rows]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    panels = [
        ("Root displacement", "m", [("root_dx_m", "X"), ("root_dy_m", "Y")]),
        ("Root tilt from vertical", "degrees", [("root_tilt_deg", "tilt")]),
        ("Target changes between policy steps", "RMS rad", [("target_delta_rms_rad", "targets")]),
        ("Foot-ground tangential contact speed (loaded)", "m/s", [("left_contact_slip_m_s", "left"), ("right_contact_slip_m_s", "right")]),
        ("Foot normal contact force", "N", [("left_normal_force_N", "left"), ("right_normal_force_N", "right")]),
        ("Foot body-origin displacement in Y", "m", [("left_origin_dy_m", "left"), ("right_origin_dy_m", "right")]),
    ]
    for ax, (title, unit, lines) in zip(axes.flat, panels):
        for key, label in lines:
            ax.plot(t, [r[key] for r in rows], label=label, linewidth=0.9)
        ax.set(title=title, ylabel=unit)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Simulated time (s)")
    fig.suptitle("Saved PPO policy: balance diagnostics", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-threshold", type=float, default=1.0, help="Foot normal force threshold in N.")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 0.02:
        parser.error("--seconds must be finite and >= 0.02")
    if not np.isfinite(args.load_threshold) or args.load_threshold < 0:
        parser.error("--load-threshold must be finite and >= 0")
    from stable_baselines3 import PPO
    from t1_balance_env import T1BalanceEnv
    import t1_balance_env
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "models/t1_balance_ppo.zip"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Put this script in your project scripts/ folder. Missing: {checkpoint}")
    env_path = Path(t1_balance_env.__file__).resolve()
    before = {str(p): sha256(p) for p in (checkpoint, env_path)}
    policy = PPO.load(str(checkpoint), device="cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = root / "logs" / f"balance_diagnostics_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "settings": vars(args), "sha256": before,
        "versions": {n: importlib.metadata.version(n) for n in ("mujoco", "stable-baselines3", "numpy", "gymnasium")},
        "notes": [
            "One deterministic home-state rollout per controller; not a robustness test.",
            "Contact estimates are sampled at the policy rate on copied simulation data.",
            "Loaded: foot normal force exceeds threshold. Unloaded does not establish airborne.",
            "Foot origin heights are not sole clearance; contact slip is not cumulative displacement.",
            "Compare full rollouts cautiously: baseline ends much earlier than PPO.",
            "Action deltas measure variation, not necessarily harmful oscillation.",
        ],
    }
    for label, controller in (("baseline", None), ("ppo", policy)):
        rows, summary = run_trial(T1BalanceEnv, controller, args.seconds, args.seed, args.load_threshold, out, label)
        report[label] = summary
        if label == "ppo":
            early = [r for r in rows if r["time_s"] <= 4.0 + 1e-9]
            late = [r for r in rows if r["time_s"] > 4.0 + 1e-9]
            for segment, subset in (("first_4s", early), ("after_4s", late)):
                if subset:
                    summary[segment + "_rms_xy_speed_m_s"] = rms([r["root_xy_speed_m_s"] for r in subset])
            try:
                plot(rows, out / "ppo_diagnostics.png")
            except Exception as exc:
                report["plot_error"] = str(exc)
                print("Plot failed; CSV and JSON results remain available:", exc)
        print(f"\n{label.upper()} SUMMARY\n" + json.dumps(summary, indent=2, allow_nan=False), flush=True)
    if any(sha256(Path(p)) != digest for p, digest in before.items()):
        raise RuntimeError("Model/environment changed during diagnostics (possibly another process).")
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"\nSaved results: {out}\nModel and environment file hashes are unchanged.")


if __name__ == "__main__":
    main()
