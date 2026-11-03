from pathlib import Path
import numpy as np
import mujoco
import base_env
from scipy.optimize import least_squares
from stable_baselines3 import PPO

SCENE = Path("../assets/mujoco_menagerie/booster_t1/scene_pick_place.xml").resolve()
CHECKPOINT = Path(
    "runs/stance_alpha0.5_20261028_215029_608658/checkpoint_000025600.zip"
)
ARM_NAMES = [
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]
ARM_TARGET = np.array([-0.300, 1.382, 1.459, 0.664])
CROUCH = np.array([
    0.03, 0, 0, 0.10, -0.03, 0,
    0.03, 0, 0, 0.10, -0.03, 0
])

def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3.0 - 2.0*x)

def main():
    base_env.MODEL_PATH = SCENE
    from balance_env import BalanceEnv

    env = BalanceEnv(
        episode_seconds=8.0,
        reset_noise=0.0,
        alpha=0.5,
        stance_weight=0.5,
        drift_weight=0.25,
        rate_weight=0.2,
        joint_velocity_weight=0.5,
    )

    try:
        policy = PPO.load(str(CHECKPOINT), device="cpu")
        model, data = env.model, env.data

        arm_ids = np.array([
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            for name in ARM_NAMES
        ])
        hand_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_link"
        )

        joint_ids = model.actuator_trnid[arm_ids, 0].astype(int)
        qpos_ids = model.jnt_qposadr[joint_ids].astype(int)

        obs, _ = env.reset(seed=42)
        home_legs = env.home_ctrl[env.lower_body_ids].copy()
        home_arm = env.home_ctrl[arm_ids].copy()

        dt = model.opt.timestep * env.frame_skip
        steps = int(3.5 / dt)

        print("Moving robot to the tested crouched pickup posture...")

        # Reproduce the previously tested PPO crouch + arm-reach phase.
        for step in range(steps):
            t = step * dt
            blend = smooth((t - 1.0) / 2.5)

            env.home_ctrl[env.lower_body_ids] = (
                home_legs + blend * CROUCH
            )
            env.home_ctrl[arm_ids] = (
                home_arm + blend * (ARM_TARGET - home_arm)
            )

            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                raise RuntimeError("Robot did not remain stable during setup.")

        # Hold the pickup pose briefly to let the arm settle.
        for _ in range(int(0.4 / dt)):
            env.home_ctrl[env.lower_body_ids] = home_legs + CROUCH
            env.home_ctrl[arm_ids] = ARM_TARGET
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                raise RuntimeError("Robot lost balance while settling.")

        hand_start = data.xpos[hand_id].copy()
        q_start = data.qpos[qpos_ids].copy()
        saved_qpos = data.qpos.copy()
        saved_qvel = data.qvel.copy()

        lo = np.array([model.jnt_range[j, 0] for j in joint_ids])
        hi = np.array([model.jnt_range[j, 1] for j in joint_ids])

        # Keep elbow pitch and yaw on the positive, non-flipped branch.
        lo[2] = max(lo[2], 0.15)
        lo[3] = max(lo[3], 0.15)
        q_start = np.clip(q_start, lo + 1e-5, hi - 1e-5)

        print("Pickup hand position:", np.round(hand_start, 4))
        print("Actual arm angles:", np.round(q_start, 4))
        print("Positive elbow limits:", (round(lo[2], 3), round(hi[2], 3)))
        print("\nScanning candidate carrying positions...")

        candidates = []

        for dz in (0.02, 0.04, 0.06, 0.08):
            for dx in (-0.06, -0.03, 0.0, 0.03, 0.06):
                for dy in (-0.04, 0.0, 0.04):
                    target = hand_start + np.array([dx, dy, dz])

                    def objective(q):
                        data.qpos[:] = saved_qpos
                        data.qvel[:] = saved_qvel
                        data.qpos[qpos_ids] = q
                        mujoco.mj_forward(model, data)

                        position_error = data.xpos[hand_id] - target
                        regularisation = 0.015 * (q - q_start)
                        return np.concatenate(
                            [position_error, regularisation]
                        )

                    result = least_squares(
                        objective,
                        q_start,
                        bounds=(lo, hi),
                        max_nfev=100,
                    )

                    data.qpos[:] = saved_qpos
                    data.qvel[:] = saved_qvel
                    data.qpos[qpos_ids] = result.x
                    mujoco.mj_forward(model, data)

                    residual = float(np.linalg.norm(
                        data.xpos[hand_id] - target
                    ))

                    candidates.append({
                        "dx": dx, "dy": dy, "dz": dz,
                        "residual": residual,
                        "angles": result.x.copy(),
                    })

        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        mujoco.mj_forward(model, data)

        feasible = [
            c for c in candidates if c["residual"] <= 0.010
        ]
        feasible.sort(
            key=lambda c: (
                -c["dz"],
                c["dx"]**2 + c["dy"]**2,
                c["residual"],
            )
        )

        print("\n========== REACHABLE CARRY POSES ==========")
        print("Acceptance: position error <= 1 cm")
        if not feasible:
            print("No candidate passed the 1 cm threshold.")
            best = sorted(candidates, key=lambda c: c["residual"])[:8]
            print("\nClosest candidates:")
        else:
            print(f"Feasible candidates: {len(feasible)}")
            best = feasible[:10]

        for c in best:
            print(
                f"offset=({c['dx']:+.2f}, {c['dy']:+.2f}, "
                f"{c['dz']:+.2f}) m | "
                f"error={c['residual']:.4f} m | "
                f"angles={np.round(c['angles'], 3)}"
            )

        if feasible:
            selected = feasible[0]
            print("\nSELECTED CARRY TARGET")
            print("Target hand:", np.round(
                hand_start + np.array([
                    selected["dx"], selected["dy"], selected["dz"]
                ]), 4
            ))
            print("Joint angles:", np.round(selected["angles"], 4))
            print("Vertical lift:", selected["dz"], "m")
            print("IK error:", round(selected["residual"], 5), "m")

    finally:
        env.close()

if __name__ == "__main__":
    main()
