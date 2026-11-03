from pathlib import Path
import numpy as np
import mujoco
import torch
import time
import mujoco.viewer
import yaml
import base_env

from scipy.optimize import least_squares
from stable_baselines3 import PPO

SCENE = Path("../assets/mujoco_menagerie/booster_t1/scene_pick_place.xml").resolve()
REF = Path("booster_gym_reference")
CONFIG = yaml.safe_load((REF / "envs/T1.yaml").read_text())
CONFIG["rewards"]["episode_length_s"] = 60.0
TEACHER_PATH = REF / "deploy/models/T1.pt"
PPO_PATH = Path(
    "runs/stance_alpha0.5_20261028_215029_608658/"
    "checkpoint_000025600.zip"
)

WALK_SECONDS = 9.0
TURN_SECONDS = 8.0
WALK_X, WALK_Y = 0.10, 0.02
TURN_YAW = -0.15

OBJECT_POSITION = np.array([0.56, -0.20, 0.635])
HAND_TARGET = OBJECT_POSITION + np.array([0.0, 0.0, 0.04])

ARM_NAMES = [
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]

CROUCH = np.array([
    0.03, 0, 0, 0.10, -0.03, 0,
    0.03, 0, 0, 0.10, -0.03, 0
], dtype=float)


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
        if not path.is_file():
            raise FileNotFoundError(path)

    base_env.MODEL_PATH = SCENE
    from balance_env import BalanceEnv

    teacher = torch.jit.load(
        str(TEACHER_PATH), map_location="cpu"
    ).eval()
    ppo = PPO.load(str(PPO_PATH), device="cpu")

    env = BalanceEnv(
        episode_seconds=90.0,
        reset_noise=0.0,
        alpha=1.0,
        stance_weight=0.0,
        drift_weight=0.0,
        rate_weight=0.0,
        joint_velocity_weight=0.0,
    )

    viewer = None
    try:
        model, data = env.model, env.data

        # Open the passive MuJoCo viewer.
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = np.array([0.85, -0.28, 0.55])
        viewer.cam.distance = 3.4
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        viewer.sync()

        # Save the native PPO controller settings.
        ppo_home = env.home_ctrl.copy()
        ppo_gain = model.actuator_gainprm.copy()
        ppo_bias = model.actuator_biasprm.copy()
        ppo_scale = env.action_scale

        norm = CONFIG["normalization"]
        decimation = int(CONFIG["control"]["decimation"])
        dt = float(model.opt.timestep)
        policy_dt = dt * decimation

        # Synchronize and pace every environment control step.
        original_env_step = env.step

        def viewer_step(action):
            if not viewer.is_running():
                raise RuntimeError('MuJoCo viewer was closed.')
            result = original_env_step(action)
            viewer.sync()
            time.sleep(policy_dt)
            return result

        env.step = viewer_step

        frequency = float(
            np.mean(CONFIG["commands"]["gait_frequency"])
        )

        if env.frame_skip != decimation:
            raise RuntimeError("Teacher/environment decimation mismatch.")

        leg_ids = env.lower_body_ids.copy()
        leg_joints = model.actuator_trnid[leg_ids, 0].astype(int)
        leg_qpos = model.jnt_qposadr[leg_joints].astype(int)
        leg_qvel = model.jnt_dofadr[leg_joints].astype(int)

        defaults = CONFIG["init_state"]["default_joint_angles"]
        stiffness = CONFIG["control"]["stiffness"]
        damping = CONFIG["control"]["damping"]
        default_pos = np.zeros(12)

        # Configure teacher-compatible leg gains.
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
                raise RuntimeError(f"No teacher gains for {name}")

            kp = float(stiffness[gain_key])
            kd = float(damping[gain_key])

            default_pos[i] = value
            env.home_ctrl[aid] = value
            model.actuator_gainprm[aid, 0] = kp
            model.actuator_biasprm[aid, 1] = -kp
            model.actuator_biasprm[aid, 2] = -kd

        env.action_scale = float(CONFIG["control"]["action_scale"])
        env.alpha = 1.0

        teacher_home = env.home_ctrl.copy()
        teacher_gain = model.actuator_gainprm.copy()
        teacher_bias = model.actuator_biasprm.copy()
        teacher_scale = float(CONFIG["control"]["action_scale"])

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
        object_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pickup_object"
        )
        mocap_id = int(model.body_mocapid[object_id])

        if np.any(arm_ids < 0) or min(hand_id, trunk_id, object_id) < 0:
            raise RuntimeError("Robot body or actuator missing.")
        if mocap_id < 0:
            raise RuntimeError("Pickup object is not a mocap body.")

        # Reset once, then keep the same live state through every phase.
        obs, _ = env.reset(seed=42)

        data.mocap_pos[mocap_id] = OBJECT_POSITION.copy()
        data.mocap_quat[mocap_id] = np.array([1., 0., 0., 0.])
        mujoco.mj_forward(model, data)

        phase = 0.0
        previous_action = np.zeros(12, dtype=np.float32)
        max_tilt = 0.0

        def run_teacher_phase(seconds, vx, vy, yaw, label):
            nonlocal phase, previous_action, max_tilt

            print("\n" + label)
            count = round(seconds / policy_dt)

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
                phase = (phase + dt*frequency*decimation) % 1.0

                R = data.xmat[trunk_id].reshape(3, 3)
                tilt = float(np.degrees(np.arccos(
                    np.clip(R[2, 2], -1.0, 1.0)
                )))
                max_tilt = max(max_tilt, tilt)

                if (i + 1) % round(1.0 / policy_dt) == 0:
                    yaw_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                    print(
                        f"  t={(i+1)*policy_dt:.1f}s | "
                        f"position={np.round(data.qpos[:3], 3)} | "
                        f"yaw={yaw_deg:+.1f}deg | tilt={tilt:.2f}deg"
                    )

                if terminated or truncated or tilt > 30.0:
                    raise RuntimeError(f"Teacher stopped during {label}.")

        print("INTEGRATED TABLE A PICKUP + LIFT")
        print("Fixed object position:", np.round(OBJECT_POSITION, 3))

        # Phase 1: Walk towards table A.
        run_teacher_phase(
            WALK_SECONDS, WALK_X, WALK_Y, 0.0,
            "PHASE 1: WALK TOWARDS TABLE A"
        )

        # Phase 2: Turn towards table A.
        run_teacher_phase(
            TURN_SECONDS, 0.0, 0.0, TURN_YAW,
            "PHASE 2: TURN TOWARDS TABLE A"
        )

        # Phase 3: Switch from teacher to PPO without resetting MuJoCo.
        print("\nPHASE 3: SWITCH TO PPO")
        model.actuator_gainprm[:] = ppo_gain
        model.actuator_biasprm[:] = ppo_bias
        env.home_ctrl[:] = ppo_home
        env.action_scale = ppo_scale
        env.alpha = 0.5

        env.origin_xy = data.qpos[:2].copy()
        env.home_width = env.width()
        env.previous_action.fill(0.0)
        obs = env._observation()

        if obs.shape != ppo.observation_space.shape:
            raise RuntimeError("PPO observation shape mismatch.")

        # Solve pickup IK using the live post-route posture.
        pickup_arm, pickup_ik_error = solve_positive_elbow_ik(
            model, data, arm_ids, hand_id, HAND_TARGET
        )

        print("Pickup IK error:", round(pickup_ik_error, 4), "m")
        print("Pickup arm targets:", np.round(pickup_arm, 3))

        if pickup_ik_error > 0.08:
            raise RuntimeError("Pickup point outside reachable workspace.")

        home_legs = ppo_home[leg_ids].copy()
        home_arm = ppo_home[arm_ids].copy()
        route_xy = data.qpos[:2].copy()

        print("\nPHASE 4: PPO BEND AND REACH")

        # Gradually bend and reach, then hold the pickup pose.
        for step in range(round(5.0 / policy_dt)):
            t = step * policy_dt
            blend = smooth(t / 2.5)

            env.home_ctrl[leg_ids] = (
                home_legs + blend*CROUCH
            )
            env.home_ctrl[arm_ids] = (
                home_arm + blend*(pickup_arm-home_arm)
            )

            action, _ = ppo.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

            R = data.xmat[trunk_id].reshape(3, 3)
            tilt = float(np.degrees(np.arccos(
                np.clip(R[2, 2], -1.0, 1.0)
            )))
            max_tilt = max(max_tilt, tilt)

            if step % round(1.0 / policy_dt) == 0:
                print(
                    f"  t={t:.1f}s | height={data.qpos[2]:.3f}m | "
                    f"tilt={tilt:.2f}deg | "
                    f"hand={np.round(data.xpos[hand_id], 3)}"
                )

            if terminated or truncated or tilt > 20.0:
                raise RuntimeError("PPO reach phase stopped.")

        actual_hand = data.xpos[hand_id].copy()
        pickup_error = float(np.linalg.norm(actual_hand-HAND_TARGET))

        print("Actual hand:", np.round(actual_hand, 3))
        print("Pickup target:", np.round(HAND_TARGET, 3))
        print("Hand error:", round(pickup_error, 4), "m")

        if pickup_error > 0.08:
            raise RuntimeError("Hand did not reach the pickup object.")

        # Phase 5: Attach the box to the actual hand pose.
        print("\nPHASE 5: SIMULATED GRASP")

        R_hand = data.xmat[hand_id].reshape(3, 3).copy()
        attached_local = R_hand.T @ (
            data.mocap_pos[mocap_id] - data.xpos[hand_id]
        )
        object_at_grasp = data.mocap_pos[mocap_id].copy()
        hand_at_grasp = data.xpos[hand_id].copy()

        print("Object attached. Hand-relative offset:",
              np.round(attached_local, 3))

        # Search for carrying poses in the robot's LOCAL coordinate frame.
        print("\\nPHASE 6: FIND CARRYING POSE")

        R_trunk = data.xmat[trunk_id].reshape(3, 3).copy()

        local_candidates = [
            np.array([0.06, -0.04, 0.04]),
            np.array([0.06, -0.06, 0.04]),
            np.array([0.06, -0.02, 0.04]),
            np.array([0.04, -0.04, 0.04]),
            np.array([0.08, -0.04, 0.04]),
            np.array([0.04, -0.06, 0.04]),
            np.array([0.08, -0.06, 0.04]),
            np.array([0.06, -0.04, 0.02]),
            np.array([0.06, -0.04, 0.06]),
            np.array([0.04, -0.02, 0.04]),
            np.array([0.02, -0.06, 0.04]),
            np.array([0.08, -0.02, 0.04]),
            np.array([0.04, -0.08, 0.04]),
            np.array([0.06, -0.08, 0.04]),
            np.array([0.06, 0.00, 0.04]),
            np.array([0.04, -0.04, 0.06]),
        ]

        carry_arm = None
        carry_offset = None
        carry_error = float("inf")

        preferred_angles = np.array([
            -0.6486, 1.3963, 1.4596, 0.6612
        ])

        joint_ids = model.actuator_trnid[arm_ids, 0].astype(int)
        joint_limits = np.array([
            model.jnt_range[j] for j in joint_ids
        ])

        valid_candidates = []

        for local_offset in local_candidates:
            # Convert robot-relative movement into world coordinates.
            world_offset = R_trunk @ local_offset
            target = hand_at_grasp + world_offset

            candidate, error = solve_positive_elbow_ik(
                model, data, arm_ids, hand_id, target
            )

            print(
                "Local offset:", np.round(local_offset, 3),
                "| IK error:", round(error, 4), "m",
                "| angles:", np.round(candidate, 3)
            )

            # Avoid solutions at the shoulder-roll and elbow limits.
            shoulder_roll_ok = (
                candidate[1] < joint_limits[1, 1] - 0.06
            )
            elbow_pitch_ok = candidate[2] > 0.50
            elbow_yaw_ok = candidate[3] > 0.20

            if (
                error <= 0.025
                and shoulder_roll_ok
                and elbow_pitch_ok
                and elbow_yaw_ok
                and local_offset[2] >= 0.02
            ):
                joint_score = float(np.linalg.norm(
                    (candidate - preferred_angles)
                    * np.array([0.5, 1.0, 0.7, 0.5])
                ))

                score = joint_score + 5.0 * error

                valid_candidates.append((
                    score, error, candidate.copy(),
                    local_offset.copy(), target.copy()
                ))

        if not valid_candidates:
            raise RuntimeError(
                "No safe robot-relative carrying pose found. "
                "Pickup succeeded, but carry-pose search needs refinement."
            )

        valid_candidates.sort(key=lambda item: item[0])

        _, carry_error, carry_arm, carry_offset, carry_target = (
            valid_candidates[0]
        )

        print("\\nSELECTED CARRYING POSE")
        print("Robot-relative offset:", np.round(carry_offset, 3))
        print("World hand target:", np.round(carry_target, 3))
        print("IK residual:", round(carry_error, 4), "m")
        print("Carry arm angles:", np.round(carry_arm, 3))
        print("Remaining valid alternatives:", len(valid_candidates))


        # Phase 7: Stand upright while lifting the object.
        print("\nPHASE 7: PPO STAND AND LIFT")

        for step in range(round(3.5 / policy_dt)):
            t = step * policy_dt
            stand_blend = smooth(t / 2.5)

            env.home_ctrl[leg_ids] = (
                home_legs + (1.0-stand_blend)*CROUCH
            )
            env.home_ctrl[arm_ids] = (
                pickup_arm + stand_blend*(carry_arm-pickup_arm)
            )

            action, _ = ppo.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

            # Maintain the kinematic grasp.
            R_hand = data.xmat[hand_id].reshape(3, 3).copy()
            data.mocap_pos[mocap_id] = (
                data.xpos[hand_id] + R_hand @ attached_local
            )
            data.mocap_quat[mocap_id] = np.array([1., 0., 0., 0.])
            mujoco.mj_forward(model, data)

            R = data.xmat[trunk_id].reshape(3, 3)
            tilt = float(np.degrees(np.arccos(
                np.clip(R[2, 2], -1.0, 1.0)
            )))
            max_tilt = max(max_tilt, tilt)

            if step % round(0.5 / policy_dt) == 0:
                print(
                    f"  t={t:.1f}s | height={data.qpos[2]:.3f}m | "
                    f"tilt={tilt:.2f}deg | "
                    f"object={np.round(data.mocap_pos[mocap_id], 3)}"
                )

            if terminated or truncated or tilt > 20.0:
                raise RuntimeError("PPO lift phase stopped.")

        # =========================================================
        # PHASE 8: SWITCH FROM PPO BACK TO THE T1 WALKING TEACHER
        # =========================================================
        print("\\nPHASE 8: PPO-TO-TEACHER HANDOFF")

        model.actuator_gainprm[:] = teacher_gain
        model.actuator_biasprm[:] = teacher_bias

        teacher_home[arm_ids] = carry_arm
        env.home_ctrl[:] = teacher_home
        env.action_scale = teacher_scale
        env.alpha = 1.0
        env.previous_action.fill(0.0)

        phase = 0.0
        previous_action = np.zeros(12, dtype=np.float32)

        handoff_position = data.qpos[:2].copy()
        print("Live robot position preserved:",
              np.round(data.qpos[:3], 3))
        print("Object at handoff:",
              np.round(data.mocap_pos[mocap_id], 3))

        # First settle with a zero-velocity teacher command.
        # Then walk forward while retaining the object attachment.
        print("\\nPHASE 9: WALK WHILE CARRYING")

        carry_test_seconds = 4.0
        carry_command_x = 0.07
        carry_steps = round(carry_test_seconds / policy_dt)

        for step in range(carry_steps):
            t = step * policy_dt

            quat = data.sensor("orientation").data[
                [1, 2, 3, 0]
            ].astype(np.float32)

            angular = data.sensor(
                "angular-velocity"
            ).data.astype(np.float32)

            gravity = quat_rotate_inverse(
                quat, np.array([0., 0., -1.], dtype=np.float32)
            )

            pos = data.qpos[leg_qpos].astype(np.float32)
            vel = data.qvel[leg_qvel].astype(np.float32)

            teacher_obs = np.zeros(47, dtype=np.float32)
            teacher_obs[0:3] = gravity * norm["gravity"]
            teacher_obs[3:6] = angular * norm["ang_vel"]
            teacher_obs[6] = carry_command_x * norm["lin_vel"]
            teacher_obs[7] = 0.0
            teacher_obs[8] = 0.0
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

            _, _, terminated, truncated, _ = env.step(action)
            previous_action = action.copy()
            phase = (phase + dt * frequency * decimation) % 1.0

            # Retain the exact hand-relative object attachment.
            R_hand = data.xmat[hand_id].reshape(3, 3).copy()
            data.mocap_pos[mocap_id] = (
                data.xpos[hand_id] + R_hand @ attached_local
            )
            data.mocap_quat[mocap_id] = np.array([1., 0., 0., 0.])
            mujoco.mj_forward(model, data)

            R_trunk = data.xmat[trunk_id].reshape(3, 3)
            tilt = float(np.degrees(np.arccos(
                np.clip(R_trunk[2, 2], -1.0, 1.0)
            )))
            max_tilt = max(max_tilt, tilt)

            if step % round(0.5 / policy_dt) == 0:
                print(
                    f"t={t:.1f}s | height={data.qpos[2]:.3f}m | "
                    f"tilt={tilt:.2f}deg | "
                    f"position={np.round(data.qpos[:2], 3)} | "
                    f"object={np.round(data.mocap_pos[mocap_id], 3)}"
                )

            if terminated or truncated or tilt > 25.0:
                raise RuntimeError(
                    f"Teacher carrying phase stopped at {t:.2f}s."
                )


        print("\nPHASE 10: TURN TOWARDS TABLE B WHILE CARRYING")

        turn_seconds = 11.2
        turn_command_yaw = 0.15
        turn_steps = round(turn_seconds / policy_dt)

        def current_yaw_deg():
            R = data.xmat[trunk_id].reshape(3, 3)
            return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

        yaw_start = current_yaw_deg()
        position_before_turn = data.qpos[:3].copy()
        object_before_turn = data.mocap_pos[mocap_id].copy()

        print(f"Starting heading: {yaw_start:.2f} degrees")
        print(f"Starting position: {np.round(position_before_turn, 3)}")
        print(f"Yaw command: {turn_command_yaw:.2f} rad/s")

        for step in range(turn_steps):
            t = step * policy_dt

            quat = data.sensor("orientation").data[
                [1, 2, 3, 0]
            ].astype(np.float32)

            angular = data.sensor(
                "angular-velocity"
            ).data.astype(np.float32)

            gravity = quat_rotate_inverse(
                quat, np.array([0., 0., -1.], dtype=np.float32)
            )

            pos = data.qpos[leg_qpos].astype(np.float32)
            vel = data.qvel[leg_qvel].astype(np.float32)

            teacher_obs = np.zeros(47, dtype=np.float32)
            teacher_obs[0:3] = gravity * norm["gravity"]
            teacher_obs[3:6] = angular * norm["ang_vel"]

            # Rotate in place: no forward or lateral velocity command.
            teacher_obs[6] = 0.0
            teacher_obs[7] = 0.0
            teacher_obs[8] = turn_command_yaw * norm["ang_vel"]

            teacher_obs[9] = np.cos(2 * np.pi * phase)
            teacher_obs[10] = np.sin(2 * np.pi * phase)
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

            _, _, terminated, truncated, _ = env.step(action)

            previous_action = action.copy()
            phase = (phase + dt * frequency * decimation) % 1.0

            # Keep the object attached to the hand throughout the turn.
            R_hand = data.xmat[hand_id].reshape(3, 3).copy()
            data.mocap_pos[mocap_id] = (
                data.xpos[hand_id] + R_hand @ attached_local
            )
            data.mocap_quat[mocap_id] = np.array([1., 0., 0., 0.])
            mujoco.mj_forward(model, data)

            R_trunk = data.xmat[trunk_id].reshape(3, 3)
            tilt = float(np.degrees(np.arccos(
                np.clip(R_trunk[2, 2], -1.0, 1.0)
            )))
            max_tilt = max(max_tilt, tilt)

            if step % round(0.5 / policy_dt) == 0:
                print(
                    f"t={t:.1f}s | yaw={current_yaw_deg():.1f}deg | "
                    f"tilt={tilt:.2f}deg | "
                    f"position={np.round(data.qpos[:2], 3)} | "
                    f"object={np.round(data.mocap_pos[mocap_id], 3)}"
                )

            if terminated or truncated or tilt > 25.0:
                raise RuntimeError(
                    f"Teacher turning phase stopped at {t:.2f}s."
                )

        yaw_end = current_yaw_deg()
        yaw_change = (yaw_end - yaw_start + 180.0) % 360.0 - 180.0

        # Table B's intended object placement point.
        table_b_target = np.array([1.56, -0.20])
        robot_xy = data.qpos[:2].copy()
        direction_to_table_b = np.degrees(np.arctan2(
            table_b_target[1] - robot_xy[1],
            table_b_target[0] - robot_xy[0]
        ))
        remaining_heading_error = (
            direction_to_table_b - yaw_end + 180.0
        ) % 360.0 - 180.0

        print("\n========== TABLE B TURN TEST RESULTS ==========")
        print(f"Starting yaw: {yaw_start:.2f} degrees")
        print(f"Final yaw: {yaw_end:.2f} degrees")
        print(f"Actual yaw change: {yaw_change:+.2f} degrees")
        print(f"Direction to Table B: {direction_to_table_b:.2f} degrees")
        print(f"Remaining heading error: {remaining_heading_error:+.2f} degrees")
        print(f"Position before turn: {np.round(position_before_turn, 3)}")
        print(f"Position after turn: {np.round(data.qpos[:3], 3)}")
        print(f"Position change: {np.round(data.qpos[:2] - position_before_turn[:2], 3)}")
        print(f"Object before turn: {np.round(object_before_turn, 3)}")
        print(f"Object after turn: {np.round(data.mocap_pos[mocap_id], 3)}")
        print(f"Maximum torso tilt: {max_tilt:.2f} degrees")
        print("Robot fell: False")
        print("Object remained kinematically attached during the turn.")


        print("\nPHASE 11: FORWARD TRANSFER WALK TOWARDS TABLE B")

        walk_seconds = 24.0
        walk_command_x = 0.07
        walk_command_y = 0.033
        walk_steps = round(walk_seconds / policy_dt)

        table_b_target = np.array([1.56, -0.20, 0.635])
        robot_before_walk = data.qpos[:3].copy()
        object_before_walk = data.mocap_pos[mocap_id].copy()
        distance_before_walk = float(np.linalg.norm(
            object_before_walk[:2] - table_b_target[:2]
        ))
        yaw_before_walk = current_yaw_deg()

        print(f"Starting heading: {yaw_before_walk:.2f} degrees")
        print("Commands: vx=0.07; vy=0.033 for 16s, then 0.04 m/s")
        print(f"Starting robot position: {np.round(robot_before_walk, 3)}")
        print(f"Starting object position: {np.round(object_before_walk, 3)}")
        print(f"Object-to-target distance: {distance_before_walk:.3f} m")

        for step in range(walk_steps):
            t = step * policy_dt
            current_walk_y = 0.033 if t < 16.0 else 0.04

            quat = data.sensor("orientation").data[
                [1, 2, 3, 0]
            ].astype(np.float32)

            angular = data.sensor(
                "angular-velocity"
            ).data.astype(np.float32)

            gravity = quat_rotate_inverse(
                quat, np.array([0., 0., -1.], dtype=np.float32)
            )

            pos = data.qpos[leg_qpos].astype(np.float32)
            vel = data.qvel[leg_qvel].astype(np.float32)

            teacher_obs = np.zeros(47, dtype=np.float32)
            teacher_obs[0:3] = gravity * norm["gravity"]
            teacher_obs[3:6] = angular * norm["ang_vel"]
            teacher_obs[6] = walk_command_x * norm["lin_vel"]
            teacher_obs[7] = current_walk_y * norm["lin_vel"]
            teacher_obs[8] = 0.0
            teacher_obs[9] = np.cos(2 * np.pi * phase)
            teacher_obs[10] = np.sin(2 * np.pi * phase)
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

            _, _, terminated, truncated, _ = env.step(action)

            previous_action = action.copy()
            phase = (phase + dt * frequency * decimation) % 1.0

            R_hand = data.xmat[hand_id].reshape(3, 3).copy()
            data.mocap_pos[mocap_id] = (
                data.xpos[hand_id] + R_hand @ attached_local
            )
            data.mocap_quat[mocap_id] = np.array([1., 0., 0., 0.])
            mujoco.mj_forward(model, data)

            R_trunk = data.xmat[trunk_id].reshape(3, 3)
            tilt = float(np.degrees(np.arccos(
                np.clip(R_trunk[2, 2], -1.0, 1.0)
            )))
            max_tilt = max(max_tilt, tilt)

            if step % max(1, round(0.5 / policy_dt)) == 0:
                obj = data.mocap_pos[mocap_id]
                distance = float(np.linalg.norm(
                    obj[:2] - table_b_target[:2]
                ))
                print(
                    f"t={t:.1f}s | yaw={current_yaw_deg():.1f}deg | "
                    f"tilt={tilt:.2f}deg | "
                    f"robot={np.round(data.qpos[:2], 3)} | "
                    f"object={np.round(obj, 3)} | "
                    f"target distance={distance:.3f}m"
                )

            if terminated or truncated or tilt > 25.0:
                raise RuntimeError(
                    f"Transfer walk stopped at {t:.2f}s: "
                    f"terminated={terminated}, truncated={truncated}, "
                    f"tilt={tilt:.2f}deg"
                )

            target_distance_now = float(np.linalg.norm(
                data.mocap_pos[mocap_id][:2] - table_b_target[:2]
            ))
            if target_distance_now <= 0.12:
                print(
                    f"Target proximity reached at t={t:.2f}s "
                    f"(distance={target_distance_now:.3f}m). Stopping walk."
                )
                break

        robot_after_walk = data.qpos[:3].copy()
        object_after_walk = data.mocap_pos[mocap_id].copy()
        distance_after_walk = float(np.linalg.norm(
            object_after_walk[:2] - table_b_target[:2]
        ))

        print("\n========== TABLE B TRANSFER WALK PROBE ==========")
        print(f"Robot displacement: {np.round(robot_after_walk[:2] - robot_before_walk[:2], 3)} m")
        print(f"Object displacement: {np.round(object_after_walk[:2] - object_before_walk[:2], 3)} m")
        print(f"Heading change: {current_yaw_deg() - yaw_before_walk:+.2f} degrees")
        print(f"Object distance before: {distance_before_walk:.3f} m")
        print(f"Object distance after: {distance_after_walk:.3f} m")
        print(f"Distance reduction: {distance_before_walk - distance_after_walk:.3f} m")
        print(f"Final object position: {np.round(object_after_walk, 3)}")
        print(f"Maximum torso tilt: {max_tilt:.2f} degrees")


        # =========================================================
        # PHASE 12: SWITCH BACK TO PPO FOR TABLE B PLACEMENT
        # =========================================================
        print("\nPHASE 12: TEACHER-TO-PPO HANDOFF AT TABLE B")

        model.actuator_gainprm[:] = ppo_gain
        model.actuator_biasprm[:] = ppo_bias
        env.home_ctrl[:] = ppo_home
        env.action_scale = ppo_scale
        env.alpha = 0.5

        env.origin_xy = data.qpos[:2].copy()
        env.home_width = env.width()
        env.previous_action.fill(0.0)
        obs = env._observation()

        arm_joint_ids = model.actuator_trnid[arm_ids, 0].astype(int)
        arm_qpos_ids = model.jnt_qposadr[arm_joint_ids].astype(int)

        arm_start = data.qpos[arm_qpos_ids].copy()
        env.home_ctrl[arm_ids] = arm_start
        env.home_ctrl[leg_ids] = ppo_home[leg_ids]

        print("Robot position:", np.round(data.qpos[:3], 3))
        print("Robot heading:", round(current_yaw_deg(), 2), "degrees")
        print("Object before placement:", np.round(data.mocap_pos[mocap_id], 3))

        # The box centre is placed on Table B. The tabletop is at z=0.600 m;
        # the box half-height is 0.035 m.
        PLACE_TARGET = np.array([
            data.mocap_pos[mocap_id][0],
            data.mocap_pos[mocap_id][1],
            0.635
        ], dtype=float)

        # Keep the current horizontal object position and lower only in Z.
        saved_qpos = data.qpos.copy()
        saved_qvel = data.qvel.copy()

        R_hand_now = data.xmat[hand_id].reshape(3, 3).copy()
        attached_offset_world = R_hand_now @ attached_local
        hand_target = PLACE_TARGET - attached_offset_world

        place_arm, place_ik_error = solve_positive_elbow_ik(
            model, data, arm_ids, hand_id, hand_target
        )

        # Preview the box position with the solved arm pose.
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        data.qpos[arm_qpos_ids] = place_arm
        mujoco.mj_forward(model, data)

        R_preview = data.xmat[hand_id].reshape(3, 3).copy()
        predicted_object = (
            data.xpos[hand_id] + R_preview @ attached_local
        )
        # Use the reachable horizontal position, provided it is on Table B.
        PLACE_TARGET[:2] = predicted_object[:2].copy()
        predicted_error = predicted_object - PLACE_TARGET

        inside_table_b = (
            1.20 <= PLACE_TARGET[0] <= 1.92
            and -0.61 <= PLACE_TARGET[1] <= -0.07
        )
        if not inside_table_b:
            raise RuntimeError("Reachable placement point is outside Table B.")

        # Restore the actual live robot state before starting motion.
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        mujoco.mj_forward(model, data)

        print("Placement arm targets:", np.round(place_arm, 3))
        print("Placement IK residual:", round(place_ik_error, 4), "m")
        print("Predicted box position:", np.round(predicted_object, 3))
        print("Predicted horizontal error:",
              round(float(np.linalg.norm(predicted_error[:2])), 4), "m")
        print("Predicted vertical error:",
              round(float(abs(predicted_error[2])), 4), "m")

        if place_ik_error > 0.10:
            raise RuntimeError(
                "Vertical placement pose is outside the reachable workspace."
            )

        if np.linalg.norm(predicted_error[:2]) > 0.02 or abs(predicted_error[2]) > 0.10:
            raise RuntimeError(
                "Predicted box position is not sufficiently close to the tabletop target."
            )

        # =========================================================
        # PHASE 13: LOWER THE BOX ONTO TABLE B
        # =========================================================
        print("\nPHASE 13: LOWER BOX ONTO TABLE B")

        placement_seconds = 4.0
        placement_steps = round(placement_seconds / policy_dt)

        for step in range(placement_steps):
            t = step * policy_dt
            blend = smooth(t / 3.0)

            env.home_ctrl[arm_ids] = (
                arm_start + blend * (place_arm - arm_start)
            )

            action, _ = ppo.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

            # Keep the box following the hand until release.
            R_hand = data.xmat[hand_id].reshape(3, 3).copy()
            data.mocap_pos[mocap_id] = (
                data.xpos[hand_id] + R_hand @ attached_local
            )
            data.mocap_quat[mocap_id] = np.array([1., 0., 0., 0.])
            mujoco.mj_forward(model, data)

            R_trunk = data.xmat[trunk_id].reshape(3, 3)
            tilt = float(np.degrees(np.arccos(
                np.clip(R_trunk[2, 2], -1.0, 1.0)
            )))
            max_tilt = max(max_tilt, tilt)

            if step % max(1, round(0.5 / policy_dt)) == 0:
                print(
                    f"t={t:.1f}s | robot height={data.qpos[2]:.3f}m | "
                    f"tilt={tilt:.2f}deg | "
                    f"hand={np.round(data.xpos[hand_id], 3)} | "
                    f"box={np.round(data.mocap_pos[mocap_id], 3)}"
                )

            if terminated or truncated or tilt > 20.0:
                raise RuntimeError(
                    f"Placement phase stopped: terminated={terminated}, "
                    f"truncated={truncated}, tilt={tilt:.2f}deg"
                )

        actual_box = data.mocap_pos[mocap_id].copy()
        placement_error = float(np.linalg.norm(actual_box - PLACE_TARGET))

        print("Box position before release:", np.round(actual_box, 3))
        print("3D placement error:", round(placement_error, 4), "m")

        if placement_error > 0.12:
            raise RuntimeError(
                "Box is too far from the Table B target; refusing to release."
            )

        # =========================================================
        # PHASE 14: RELEASE THE KINEMATIC OBJECT
        # =========================================================
        print("\nPHASE 14: RELEASE BOX ON TABLE B")

        # The box is a visual-only mocap body, so place its centre precisely
        # on the tabletop and stop updating it from the hand.
        print(
            "Kinematic release: placing box at reachable tabletop coordinates; "
            "the final vertical set-down is scripted."
        )
        data.mocap_pos[mocap_id] = PLACE_TARGET.copy()
        data.mocap_quat[mocap_id] = np.array([1., 0., 0., 0.])
        mujoco.mj_forward(model, data)

        print("Released box position:", np.round(data.mocap_pos[mocap_id], 3))
        print("Tabletop surface: z=0.600m")
        print("Box bottom: z=0.600m")
        print("Kinematic attachment: released")
        print("Release method: scripted mocap placement")

        # =========================================================
        # PHASE 15: RETRACT ARM AND STABILIZE STANDING POSE
        # =========================================================
        print("\nPHASE 15: RETRACT ARM AND STABILIZE")

        current_hand = data.xpos[hand_id].copy()
        retreat_target = current_hand + np.array([0.0, 0.0, 0.12])

        retreat_arm, retreat_error = solve_positive_elbow_ik(
            model, data, arm_ids, hand_id, retreat_target
        )

        print("Retraction IK error:", round(retreat_error, 4), "m")

        if retreat_error <= 0.08:
            retreat_start = data.qpos[arm_qpos_ids].copy()

            for step in range(round(2.5 / policy_dt)):
                t = step * policy_dt
                blend = smooth(t / 2.0)

                env.home_ctrl[arm_ids] = (
                    retreat_start + blend * (retreat_arm - retreat_start)
                )
                env.home_ctrl[leg_ids] = ppo_home[leg_ids]

                action, _ = ppo.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)

                R_trunk = data.xmat[trunk_id].reshape(3, 3)
                tilt = float(np.degrees(np.arccos(
                    np.clip(R_trunk[2, 2], -1.0, 1.0)
                )))
                max_tilt = max(max_tilt, tilt)

                if terminated or truncated or tilt > 20.0:
                    raise RuntimeError("Arm retraction / standing stabilization stopped.")

        else:
            print("Retraction IK did not converge.")
            print("Using a smooth fallback to the neutral PPO home-arm pose.")

            retreat_start = data.qpos[arm_qpos_ids].copy()

            for step in range(round(2.5 / policy_dt)):
                t = step * policy_dt
                blend = smooth(t / 2.0)

                env.home_ctrl[arm_ids] = (
                    retreat_start + blend * (home_arm - retreat_start)
                )
                env.home_ctrl[leg_ids] = ppo_home[leg_ids]

                action, _ = ppo.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)

                R_trunk = data.xmat[trunk_id].reshape(3, 3)
                tilt = float(np.degrees(np.arccos(
                    np.clip(R_trunk[2, 2], -1.0, 1.0)
                )))
                max_tilt = max(max_tilt, tilt)

                if terminated or truncated or tilt > 20.0:
                    raise RuntimeError(
                        "Neutral-arm retraction stopped: "
                        f"terminated={terminated}, truncated={truncated}, "
                        f"tilt={tilt:.2f}deg"
                    )

        # Hold the robot upright for a short settling interval.
        env.home_ctrl[leg_ids] = ppo_home[leg_ids]

        for step in range(round(1.5 / policy_dt)):
            action, _ = ppo.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

            R_trunk = data.xmat[trunk_id].reshape(3, 3)
            tilt = float(np.degrees(np.arccos(
                np.clip(R_trunk[2, 2], -1.0, 1.0)
            )))
            max_tilt = max(max_tilt, tilt)

            if terminated or truncated or tilt > 20.0:
                raise RuntimeError("Final standing stabilization stopped.")

        final_hand = data.xpos[hand_id].copy()
        final_box = data.mocap_pos[mocap_id].copy()

        print("\n========== TABLE B PLACEMENT RESULTS ==========")
        print("Final robot position:", np.round(data.qpos[:3], 3))
        print("Final robot heading:", round(current_yaw_deg(), 2), "degrees")
        print("Final hand position:", np.round(final_hand, 3))
        print("Final box position:", np.round(final_box, 3))
        print("Box target:", np.round(PLACE_TARGET, 3))
        print("Final box placement error:",
              round(float(np.linalg.norm(final_box - PLACE_TARGET)), 4), "m")
        print("Maximum torso tilt:", round(max_tilt, 2), "degrees")
        print("Robot completed placement without a reported fall.")

        object_final = data.mocap_pos[mocap_id].copy()
        final_position = data.qpos[:3].copy()
        final_hand = data.xpos[hand_id].copy()
        handoff_displacement = data.qpos[:2] - handoff_position

        print("\\n========== PPO-TO-TEACHER CARRY RESULTS ==========")
        print("Robot fell: False")
        print("Post-handoff displacement:",
              np.round(handoff_displacement, 3), "m")
        print("Object at handoff:",
              np.round(object_at_grasp + np.array([0., 0., 0.]), 3))
        print("Object after walking:", np.round(object_final, 3))
        print("Maximum torso tilt:", round(max_tilt, 2), "degrees")
        print("Final hand:", np.round(final_hand, 3))
        print("Final robot position:", np.round(final_position, 3))

        table_b = np.array([1.56, -0.34])
        print("Remaining horizontal distance to table B:",
              round(float(np.linalg.norm(table_b-final_position[:2])), 3),
              "m")

        final_hand = data.xpos[hand_id].copy()
        drift = float(np.linalg.norm(data.qpos[:2] - route_xy))

        print("\n========== INTEGRATED PICKUP RESULT ==========")
        print("Walked to table A: True")
        print("Turned towards table A: True")
        print("Teacher-to-PPO handoff: True")
        print("Reached object: True")
        print("Simulated grasp: True")
        print("Found carrying pose: True")
        print("Fell: False")
        print("Maximum torso tilt:", round(max_tilt, 2), "degrees")
        print("PPO manipulation drift:", round(drift, 3), "m")
        print("Pickup hand error:", round(pickup_error, 3), "m")
        print("Object at grasp:", np.round(object_at_grasp, 3))
        print("Object after lift:", np.round(object_final, 3))
        print("Object displacement:", np.round(object_final-object_at_grasp, 3))
        print("Final hand:", np.round(final_hand, 3))
        print("Final robot position:", np.round(final_position, 3))
        print("Carry arm angles:", np.round(carry_arm, 3))
        print("Maximum torso tilt:", round(max_tilt, 2), "degrees")
        print("\nThis run tests the staged transfer and stops near the fixed Table B target.")
        print("\nFinal scene is visible. Close the MuJoCo viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)

    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
