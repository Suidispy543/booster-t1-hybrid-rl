from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml
import base_env
from scipy.optimize import least_squares

SCENE = Path("../assets/mujoco_menagerie/booster_t1/scene_pick_place.xml").resolve()
REF = Path("booster_gym_reference")
CONFIG = yaml.safe_load((REF / "envs/T1.yaml").read_text())
TEACHER_PATH = REF / "deploy/models/T1.pt"

# Initial route calibration.
WALK_SECONDS = 8.0
TURN_SECONDS = 10.5
COMMAND_X = 0.10
COMMAND_Y = 0.02
COMMAND_YAW = -0.15

# Fixed pickup position from scene geometry.
PICKUP_OBJECT = np.array([0.56, -0.20, 0.635])
PICKUP_HAND = PICKUP_OBJECT + np.array([0.0, 0.0, 0.04])

ARM_NAMES = [
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]


def quat_rotate_inverse(q, v):
    qw, qv = q[-1], q[:3]
    return (
        v * (2.0 * qw**2 - 1.0)
        - np.cross(qv, v) * (2.0 * qw)
        + qv * (2.0 * np.dot(qv, v))
    )


def yaw_of(data, trunk_id):
    R = data.xmat[trunk_id].reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def solve_positive_elbow_ik(model, data, arm_ids, hand_id, target):
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()

    joint_ids = model.actuator_trnid[arm_ids, 0].astype(int)
    qpos_ids = model.jnt_qposadr[joint_ids].astype(int)

    lo = np.array([model.jnt_range[j, 0] for j in joint_ids])
    hi = np.array([model.jnt_range[j, 1] for j in joint_ids])

    # Keep the elbow/yaw on the non-flipped branch.
    lo[2] = max(lo[2], 0.15)
    lo[3] = max(lo[3], 0.15)

    start = np.clip(saved_qpos[qpos_ids], lo + 1e-5, hi - 1e-5)

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
    if not SCENE.exists() or not TEACHER_PATH.exists():
        raise FileNotFoundError("Scene or teacher model missing.")

    base_env.MODEL_PATH = SCENE
    from balance_env import BalanceEnv

    teacher = torch.jit.load(
        str(TEACHER_PATH), map_location="cpu"
    ).eval()

    env = BalanceEnv(
        episode_seconds=22.0,
        reset_noise=0.0,
        alpha=1.0,
        stance_weight=0.0,
        drift_weight=0.0,
        rate_weight=0.0,
        joint_velocity_weight=0.0,
    )

    try:
        model, data = env.model, env.data
        norm = CONFIG["normalization"]

        decimation = int(CONFIG["control"]["decimation"])
        if env.frame_skip != decimation:
            raise RuntimeError("Teacher decimation mismatch.")

        dt = float(model.opt.timestep)
        policy_dt = dt * decimation
        frequency = float(
            np.mean(CONFIG["commands"]["gait_frequency"])
        )

        leg_ids = env.lower_body_ids.copy()
        leg_joints = model.actuator_trnid[leg_ids, 0].astype(int)
        leg_qpos = model.jnt_qposadr[leg_joints].astype(int)
        leg_qvel = model.jnt_dofadr[leg_joints].astype(int)

        defaults = CONFIG["init_state"]["default_joint_angles"]
        stiffness = CONFIG["control"]["stiffness"]
        damping = CONFIG["control"]["damping"]
        default_pos = np.zeros(12)

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

            kp = float(stiffness[gain_key])
            kd = float(damping[gain_key])

            default_pos[i] = value
            env.home_ctrl[aid] = value
            model.actuator_gainprm[aid, 0] = kp
            model.actuator_biasprm[aid, 1] = -kp
            model.actuator_biasprm[aid, 2] = -kd

        env.action_scale = float(CONFIG["control"]["action_scale"])

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
        previous_action = np.zeros(12, dtype=np.float32)
        phase = 0.0

        initial_xy = data.qpos[:2].copy()
        max_tilt = 0.0

        def run_teacher_phase(seconds, vx, vy, yaw, label):
            nonlocal obs, previous_action, phase, max_tilt

            count = round(seconds / policy_dt)
            print(f"\n{label}")

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

                observation = np.zeros(47, dtype=np.float32)
                observation[0:3] = gravity * norm["gravity"]
                observation[3:6] = angular * norm["ang_vel"]
                observation[6] = vx * norm["lin_vel"]
                observation[7] = vy * norm["lin_vel"]
                observation[8] = yaw * norm["ang_vel"]
                observation[9] = np.cos(2*np.pi*phase)
                observation[10] = np.sin(2*np.pi*phase)
                observation[11:23] = (
                    pos - default_pos
                ) * norm["dof_pos"]
                observation[23:35] = vel * norm["dof_vel"]
                observation[35:47] = previous_action

                with torch.no_grad():
                    action = teacher(
                        torch.from_numpy(observation).unsqueeze(0)
                    ).squeeze(0).cpu().numpy().astype(np.float32)

                action = np.clip(
                    action, -norm["clip_actions"], norm["clip_actions"]
                )

                _, _, terminated, truncated, _ = env.step(action)
                previous_action = action.copy()
                phase = (phase + dt * frequency * decimation) % 1.0

                tilt = float(np.degrees(np.arccos(
                    np.clip(
                        data.xmat[trunk_id].reshape(3, 3)[2, 2],
                        -1.0, 1.0
                    )
                )))
                max_tilt = max(max_tilt, tilt)

                if (i + 1) % round(1.0 / policy_dt) == 0:
                    print(
                        f"  t={(i+1)*policy_dt:.1f}s | "
                        f"pos={np.round(data.qpos[:3], 3)} | "
                        f"yaw={np.degrees(yaw_of(data, trunk_id)):+.1f}deg | "
                        f"tilt={tilt:.2f}deg"
                    )

                if terminated or truncated or tilt > 30.0:
                    raise RuntimeError(
                        f"Teacher phase stopped: {label}, step {i}"
                    )

        print("TABLE A APPROACH + ORIENTATION TEST")
        print("Initial robot position:", np.round(initial_xy, 3))
        print("Target pickup object:", PICKUP_OBJECT)
        print("Walking lateral correction:", COMMAND_Y)

        run_teacher_phase(
            WALK_SECONDS, COMMAND_X, COMMAND_Y, 0.0,
            "PHASE 1: WALK TOWARDS TABLE A"
        )

        run_teacher_phase(
            TURN_SECONDS, 0.0, 0.0, COMMAND_YAW,
            "PHASE 2: TURN TOWARDS TABLE A"
        )

        trunk_position = data.xpos[trunk_id].copy()
        current_yaw = yaw_of(data, trunk_id)
        R = data.xmat[trunk_id].reshape(3, 3).copy()

        target_relative = R.T @ (PICKUP_HAND - trunk_position)

        print("\n========== TABLE A ALIGNMENT ==========")
        print("Final trunk position:", np.round(trunk_position, 3))
        print("Final trunk yaw:", round(np.degrees(current_yaw), 2), "deg")
        print("Pickup hand target:", np.round(PICKUP_HAND, 3))
        print("Target relative to trunk:", np.round(target_relative, 3))

        arm_target, residual = solve_positive_elbow_ik(
            model, data, arm_ids, hand_id, PICKUP_HAND
        )

        print("\n========== PICKUP IK ==========")
        print("IK residual:", round(residual, 4), "m")
        print("Positive-elbow joint targets:", np.round(arm_target, 3))
        print("Current hand:", np.round(data.xpos[hand_id], 3))
        print("Target hand:", np.round(PICKUP_HAND, 3))
        print("Maximum torso tilt during route:", round(max_tilt, 2), "deg")

        if residual <= 0.08:
            print("\nRESULT: Table A pickup point is reachable.")
        else:
            print("\nRESULT: Table A pickup point needs approach adjustment.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
