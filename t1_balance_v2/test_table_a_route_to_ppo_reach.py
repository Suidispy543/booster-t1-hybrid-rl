from pathlib import Path

import numpy as np
import mujoco
import torch
import yaml
import base_env

from scipy.optimize import least_squares
from stable_baselines3 import PPO

SCENE = Path("../assets/mujoco_menagerie/booster_t1/scene_pick_place.xml").resolve()
REF = Path("booster_gym_reference")
CONFIG = yaml.safe_load((REF / "envs/T1.yaml").read_text())

TEACHER_PATH = REF / "deploy/models/T1.pt"
PPO_PATH = Path(
    "runs/stance_alpha0.5_20261028_215029_608658/"
    "checkpoint_000025600.zip"
)

WALK_SECONDS = 9.0
TURN_SECONDS = 8.0
WALK_X = 0.10
WALK_Y = 0.02
TURN_YAW = -0.15

PICKUP_OBJECT = np.array([0.56, -0.20, 0.635])
PICKUP_HAND = PICKUP_OBJECT + np.array([0.0, 0.0, 0.04])

ARM_NAMES = [
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]

CROUCH = np.array([
    0.03, 0, 0, 0.10, -0.03, 0,
    0.03, 0, 0, 0.10, -0.03, 0
])


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3.0 - 2.0*x)


def quat_rotate_inverse(q, v):
    qw, qv = q[-1], q[:3]
    return (
        v * (2.0 * qw**2 - 1.0)
        - np.cross(qv, v) * (2.0 * qw)
        + qv * (2.0 * np.dot(qv, v))
    )


def solve_positive_elbow_ik(model, data, arm_ids, hand_id, target):
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()

    joint_ids = model.actuator_trnid[arm_ids, 0].astype(int)
    qpos_ids = model.jnt_qposadr[joint_ids].astype(int)

    lo = np.array([model.jnt_range[j, 0] for j in joint_ids])
    hi = np.array([model.jnt_range[j, 1] for j in joint_ids])

    lo[2] = max(lo[2], 0.15)
    lo[3] = max(lo[3], 0.15)

    start = np.clip(
        saved_qpos[qpos_ids], lo + 1e-5, hi - 1e-5
    )

    def error(q):
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        return data.xpos[hand_id] - target

    rng = np.random.default_rng(42)
    starts = [start]
    starts += [
        rng.uniform(lo + 1e-4, hi - 1e-4)
        for _ in range(11)
    ]

    best = None
    for initial in starts:
        result = least_squares(
            error, initial, bounds=(lo, hi), max_nfev=400
        )
        residual = float(np.linalg.norm(error(result.x)))
        if best is None or residual < best[0]:
            best = (residual, result.x.copy())

    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    mujoco.mj_forward(model, data)

    return best[1], best[0]


def main():
    for path in (SCENE, TEACHER_PATH, PPO_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    base_env.MODEL_PATH = SCENE

    from balance_env import BalanceEnv

    teacher = torch.jit.load(
        str(TEACHER_PATH), map_location="cpu"
    ).eval()

    ppo = PPO.load(str(PPO_PATH), device="cpu")

    env = BalanceEnv(
        episode_seconds=30.0,
        reset_noise=0.0,
        alpha=1.0,
        stance_weight=0.0,
        drift_weight=0.0,
        rate_weight=0.0,
        joint_velocity_weight=0.0
    )

    try:
        model, data = env.model, env.data

        # Save the original PPO controller settings.
        ppo_home = env.home_ctrl.copy()
        ppo_gain = model.actuator_gainprm.copy()
        ppo_bias = model.actuator_biasprm.copy()
        ppo_action_scale = env.action_scale

        norm = CONFIG["normalization"]
        decimation = int(CONFIG["control"]["decimation"])
        dt = float(model.opt.timestep)
        policy_dt = dt * decimation
        frequency = float(
            np.mean(CONFIG["commands"]["gait_frequency"])
        )
        teacher_scale = float(CONFIG["control"]["action_scale"])

        leg_ids = env.lower_body_ids.copy()
        leg_joints = model.actuator_trnid[leg_ids, 0].astype(int)
        leg_qpos = model.jnt_qposadr[leg_joints].astype(int)
        leg_qvel = model.jnt_dofadr[leg_joints].astype(int)

        defaults = CONFIG["init_state"]["default_joint_angles"]
        stiffness = CONFIG["control"]["stiffness"]
        damping = CONFIG["control"]["damping"]
        default_pos = np.zeros(12)

        # Configure teacher-compatible PD gains.
        for i, aid in enumerate(leg_ids):
            name = (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid)
                ) or ""
            )

            value = float(defaults["default"])
            for key, val in defaults.items():
                if key != "default" and key in name:
                    value = float(val)
                    break

            gain_key = next(
                (key for key in stiffness if key in name), None
            )
            if gain_key is None:
                raise RuntimeError(f"Missing teacher gains for {name}")

            default_pos[i] = value
            env.home_ctrl[aid] = value

            kp = float(stiffness[gain_key])
            kd = float(damping[gain_key])

            model.actuator_gainprm[aid, 0] = kp
            model.actuator_biasprm[aid, 1] = -kp
            model.actuator_biasprm[aid, 2] = -kd

        env.action_scale = teacher_scale
        env.alpha = 1.0

        arm_ids = np.array([
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            for name in ARM_NAMES
        ])

        hand_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_link"
        )
        trunk_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "Trunk"
        )

        obs, _ = env.reset(seed=42)

        phase = 0.0
        previous_action = np.zeros(12, dtype=np.float32)
        max_tilt = 0.0

        def teacher_phase(seconds, vx, vy, yaw, label):
            nonlocal phase, previous_action, max_tilt

            count = round(seconds / policy_dt)
            print("\n" + label)

            for i in range(count):
                quat = data.sensor("orientation").data[
                    [1, 2, 3, 0]
                ].astype(np.float32)

                angular = data.sensor(
                    "angular-velocity"
                ).data.astype(np.float32)

                gravity = quat_rotate_inverse(
                    quat,
                    np.array([0., 0., -1.], dtype=np.float32)
                )

                pos = data.qpos[leg_qpos].astype(np.float32)
                vel = data.qvel[leg_qvel].astype(np.float32)

                teacher_obs = np.zeros(47, dtype=np.float32)
                teacher_obs[0:3] = gravity * norm["gravity"]
                teacher_obs[3:6] = angular * norm["ang_vel"]
                teacher_obs[6] = vx * norm["lin_vel"]
                teacher_obs[7] = vy * norm["lin_vel"]
                teacher_obs[8] = yaw * norm["ang_vel"]
                teacher_obs[9] = np.cos(2*np.pi*phase)
                teacher_obs[10] = np.sin(2*np.pi*phase)
                teacher_obs[11:23] = (
                    pos - default_pos
                ) * norm["dof_pos"]
                teacher_obs[23:35] = vel * norm["dof_vel"]
                teacher_obs[35:47] = previous_action

                with torch.no_grad():
                    action = teacher(
                        torch.from_numpy(teacher_obs).unsqueeze(0)
                    ).squeeze(0).cpu().numpy().astype(np.float32)

                action = np.clip(
                    action, -norm["clip_actions"], norm["clip_actions"]
                )

                obs, _, terminated, truncated, _ = env.step(action)
                previous_action = action.copy()
                phase = (phase + dt * frequency * decimation) % 1.0

                R = data.xmat[trunk_id].reshape(3, 3)
                tilt = float(np.degrees(np.arccos(
                    np.clip(R[2, 2], -1.0, 1.0)
                )))
                max_tilt = max(max_tilt, tilt)

                if (i + 1) % round(1.0 / policy_dt) == 0:
                    yaw_now = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                    print(
                        f"t={(i+1)*policy_dt:.1f}s | "
                        f"position={np.round(data.qpos[:3], 3)} | "
                        f"yaw={yaw_now:+.1f}deg | tilt={tilt:.2f}deg"
                    )

                if terminated or truncated or tilt > 30.0:
                    raise RuntimeError(
                        f"Teacher stopped during {label}"
                    )

        # 1. Walk towards table A.
        print("TABLE A ROUTE + PPO PICKUP REACH TEST")

        teacher_phase(
            WALK_SECONDS, WALK_X, WALK_Y, 0.0,
            "PHASE 1: WALK TOWARDS TABLE A"
        )

        # 2. Turn towards table A.
        teacher_phase(
            TURN_SECONDS, 0.0, 0.0, TURN_YAW,
            "PHASE 2: TURN TOWARDS TABLE A"
        )

        # 3. Calculate IK at the actual post-walking posture.
        target_arm, ik_error = solve_positive_elbow_ik(
            model, data, arm_ids, hand_id, PICKUP_HAND
        )

        print("\n========== PICKUP IK ==========")
        print("Current hand:", np.round(data.xpos[hand_id], 3))
        print("Target hand:", np.round(PICKUP_HAND, 3))
        print("IK residual:", round(ik_error, 4), "m")
        print("Arm targets:", np.round(target_arm, 3))

        if ik_error > 0.08:
            raise RuntimeError("Pickup target is outside the reachable workspace.")

        # 4. Switch to the trained PPO controller without resetting MuJoCo.
        print("\n========== SWITCHING TO PPO ==========")

        model.actuator_gainprm[:] = ppo_gain
        model.actuator_biasprm[:] = ppo_bias
        env.home_ctrl[:] = ppo_home
        env.action_scale = ppo_action_scale
        env.alpha = 0.5

        env.origin_xy = data.qpos[:2].copy()
        env.home_width = env.width()
        env.previous_action.fill(0.0)

        obs = env._observation()

        if obs.shape != ppo.observation_space.shape:
            raise RuntimeError("PPO observation shape mismatch.")

        print("State preserved:", np.round(data.qpos[:3], 3))
        print("PPO observation shape:", obs.shape)

        # 5. PPO stabilises, bends slightly and reaches.
        print("\n========== PHASE 3: PPO BEND + REACH ==========")

        initial_xy = data.qpos[:2].copy()
        home_legs = ppo_home[leg_ids].copy()
        home_arm = ppo_home[arm_ids].copy()

        total_seconds = 5.0
        steps = round(total_seconds / policy_dt)

        for step in range(steps):
            t = step * policy_dt

            # Smooth crouch and reach over the first 2.5 seconds.
            blend = smooth(t / 2.5)

            env.home_ctrl[leg_ids] = (
                home_legs + blend * CROUCH
            )
            env.home_ctrl[arm_ids] = (
                home_arm + blend * (target_arm - home_arm)
            )

            action, _ = ppo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            R = data.xmat[trunk_id].reshape(3, 3)
            tilt = float(np.degrees(np.arccos(
                np.clip(R[2, 2], -1.0, 1.0)
            )))
            max_tilt = max(max_tilt, tilt)

            if step % round(1.0 / policy_dt) == 0:
                print(
                    f"t={t:.1f}s | height={data.qpos[2]:.3f}m | "
                    f"tilt={tilt:.2f}deg | "
                    f"drift={np.linalg.norm(data.qpos[:2]-initial_xy):.3f}m | "
                    f"hand={np.round(data.xpos[hand_id], 3)}"
                )

            if terminated or truncated or tilt > 20.0:
                raise RuntimeError("PPO manipulation phase stopped.")

        final_hand = data.xpos[hand_id].copy()
        hand_error = float(np.linalg.norm(final_hand - PICKUP_HAND))

        print("\n========== TABLE A PICKUP REACH RESULTS ==========")
        print("PPO handoff succeeded: True")
        print("Maximum torso tilt:", round(max_tilt, 2), "degrees")
        print("Final hand position:", np.round(final_hand, 3))
        print("Target hand position:", np.round(PICKUP_HAND, 3))
        print("Hand-to-target error:", round(hand_error, 4), "m")
        print("Robot final position:", np.round(data.qpos[:3], 3))

        if hand_error <= 0.08:
            print("\nRESULT: TABLE A APPROACH, HANDOFF AND REACH PASSED.")
        else:
            print("\nRESULT: Route passed, but arm reach needs refinement.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
