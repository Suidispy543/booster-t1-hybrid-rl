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
        episode_seconds=40.0,
        reset_noise=0.0,
        alpha=1.0,
        stance_weight=0.0,
        drift_weight=0.0,
        rate_weight=0.0,
        joint_velocity_weight=0.0,
    )

    try:
        model, data = env.model, env.data

        # Save the native PPO controller settings.
        ppo_home = env.home_ctrl.copy()
        ppo_gain = model.actuator_gainprm.copy()
        ppo_bias = model.actuator_biasprm.copy()
        ppo_scale = env.action_scale

        norm = CONFIG["normalization"]
        decimation = int(CONFIG["control"]["decimation"])
        dt = float(model.opt.timestep)
        policy_dt = dt * decimation
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
        print("\nThis test ends at table A. It does not yet walk to table B.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
