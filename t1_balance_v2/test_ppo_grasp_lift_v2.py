from pathlib import Path
import numpy as np
import mujoco
import base_env
from scipy.optimize import least_squares
from stable_baselines3 import PPO

SCENE = Path("../assets/mujoco_menagerie/booster_t1/scene_pick_place.xml").resolve()
CHECKPOINT = Path(
    "runs/stance_alpha0.5_20261028_215029_608658/"
    "checkpoint_000025600.zip"
)

base_env.MODEL_PATH = SCENE
from balance_env import BalanceEnv

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
], dtype=float)

# Box centre: approximately 3.5 cm below the hand.
OBJECT_START = np.array([0.102, -0.183, 0.634])


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3.0 - 2.0*x)


def solve_arm_ik(model, data, arm_ids, hand_id, target):
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()

    joint_ids = model.actuator_trnid[arm_ids, 0].astype(int)
    qpos_ids = model.jnt_qposadr[joint_ids].astype(int)

    lo = np.array([model.jnt_range[j, 0] for j in joint_ids])
    hi = np.array([model.jnt_range[j, 1] for j in joint_ids])

    start = np.clip(
        saved_qpos[qpos_ids],
        lo + 1e-5,
        hi - 1e-5
    )

    def error(q):
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        return data.xpos[hand_id] - target

    best = None
    rng = np.random.default_rng(17)
    starts = [start]

    for _ in range(4):
        starts.append(rng.uniform(lo + 1e-4, hi - 1e-4))

    for initial in starts:
        result = least_squares(
            error,
            initial,
            bounds=(lo, hi),
            max_nfev=400
        )
        residual = float(np.linalg.norm(error(result.x)))

        if best is None or residual < best[0]:
            best = (residual, result.x.copy())

    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    mujoco.mj_forward(model, data)

    return best[1], best[0]


def main():
    if not SCENE.exists():
        raise FileNotFoundError(SCENE)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    env = BalanceEnv(
        episode_seconds=9.0,
        reset_noise=0.0,
        alpha=0.5,
        stance_weight=0.5,
        drift_weight=0.25,
        rate_weight=0.2,
        joint_velocity_weight=0.5
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
        trunk_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "Trunk"
        )
        object_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pickup_object"
        )

        mocap_id = int(model.body_mocapid[object_id])

        if np.any(arm_ids < 0) or min(hand_id, trunk_id, object_id) < 0:
            raise RuntimeError("Required robot body or actuator missing.")
        if mocap_id < 0:
            raise RuntimeError("Pickup object is not a mocap body.")

        obs, _ = env.reset(seed=42)

        home_legs = env.home_ctrl[env.lower_body_ids].copy()
        home_arm = env.home_ctrl[arm_ids].copy()
        arm_joint_ids = model.actuator_trnid[arm_ids, 0].astype(int)
        arm_qpos_ids = model.jnt_qposadr[arm_joint_ids].astype(int)
        initial_xy = data.qpos[:2].copy()

        print("\\nArm joint limits:")
        for name, jid in zip(ARM_NAMES, arm_joint_ids):
            print(name, np.round(model.jnt_range[jid], 3))

        data.mocap_pos[mocap_id] = OBJECT_START.copy()
        data.mocap_quat[mocap_id] = np.array([1., 0., 0., 0.])
        mujoco.mj_forward(model, data)

        dt = model.opt.timestep * env.frame_skip
        steps = int(8.0 / dt)

        attached_local = None
        carry_arm_target = None
        grasp_object_start = None
        grasp_hand_start = None
        grasp_trunk_start = None
        grasped = False

        max_tilt = 0.0
        min_height = float(data.qpos[2])
        max_drift = 0.0
        fell = False

        print("PPO GRASP AND ACTUAL ARM LIFT TEST V2")
        print("0-1s: stabilise")
        print("1-3.5s: bend and reach")
        print("3.5s: attach box")
        print("3.5-6s: stand and lift arm")
        print("6-8s: hold carrying pose")

        next_report = 0.0

        for step in range(steps):
            t = step * dt

            # Leg crouch and stand trajectory.
            if t < 1.0:
                crouch_blend = 0.0
            elif t < 3.5:
                crouch_blend = smooth((t - 1.0) / 2.5)
            elif t < 4.0:
                crouch_blend = 1.0
            elif t < 6.0:
                crouch_blend = 1.0 - smooth((t - 4.0) / 2.0)
            else:
                crouch_blend = 0.0

            # Arm reaches the pickup pose before grasp.
            reach_blend = smooth((t - 1.0) / 2.5)
            reach_blend = min(reach_blend, 1.0)

            env.home_ctrl[env.lower_body_ids] = (
                home_legs + crouch_blend * CROUCH
            )

            if not grasped:
                env.home_ctrl[arm_ids] = (
                    home_arm + reach_blend * (ARM_TARGET - home_arm)
                )
            elif carry_arm_target is not None:
                lift_blend = smooth((t - 3.5) / 2.0)
                env.home_ctrl[arm_ids] = (
                    ARM_TARGET
                    + lift_blend * (carry_arm_target - ARM_TARGET)
                )

            # PPO maintains lower-body balance.
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Attach object once the arm has reached the pickup pose.
            if not grasped and t >= 3.5:
                R_hand = data.xmat[hand_id].reshape(3, 3).copy()

                attached_local = R_hand.T @ (
                    data.mocap_pos[mocap_id] - data.xpos[hand_id]
                )

                grasp_object_start = data.mocap_pos[mocap_id].copy()
                grasp_hand_start = data.xpos[hand_id].copy()
                grasp_trunk_start = data.xpos[trunk_id].copy()

                print("\nOBJECT ATTACHED")
                print("Hand at grasp:", np.round(grasp_hand_start, 3))
                print("Object at grasp:", np.round(grasp_object_start, 3))

                # Use the positive-elbow carry pose found by the
                # positive-elbow workspace scan.
                carry_arm_target = np.array([
                    -0.6486, 1.3963, 1.4596, 0.6612
                ])

                desired_hand = grasp_hand_start + np.array([
                    0.06, 0.0, 0.04
                ])

                arm_joint_ids = model.actuator_trnid[
                    arm_ids, 0
                ].astype(int)
                arm_qpos_ids = model.jnt_qposadr[
                    arm_joint_ids
                ].astype(int)

                saved_qpos = data.qpos.copy()
                saved_qvel = data.qvel.copy()

                data.qpos[arm_qpos_ids] = carry_arm_target
                mujoco.mj_forward(model, data)

                predicted_hand = data.xpos[hand_id].copy()
                carry_pose_error = float(np.linalg.norm(
                    predicted_hand - desired_hand
                ))

                data.qpos[:] = saved_qpos
                data.qvel[:] = saved_qvel
                mujoco.mj_forward(model, data)

                print("Positive-elbow carry target:",
                      np.round(carry_arm_target, 4))
                print("Predicted hand position:",
                      np.round(predicted_hand, 4))
                print("Desired hand position:",
                      np.round(desired_hand, 4))
                print("Carry-pose error:",
                      round(carry_pose_error, 4), "m")

                if carry_pose_error > 0.025:
                    raise RuntimeError(
                        "Scanned carry pose does not match the current "
                        "pickup posture closely enough."
                    )

                grasped = True

            # Follow the hand with the kinematic object.
            if grasped:
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

            height = float(data.qpos[2])
            drift = float(np.linalg.norm(data.qpos[:2] - initial_xy))

            max_tilt = max(max_tilt, tilt)
            min_height = min(min_height, height)
            max_drift = max(max_drift, drift)

            if t >= next_report:
                print(
                    f"t={t:.1f}s | crouch={crouch_blend:.2f} | "
                    f"height={height:.3f}m | tilt={tilt:.2f}deg | "
                    f"drift={drift:.3f}m | "
                    f"hand={np.round(data.xpos[hand_id], 3)} | "
                    f"object={np.round(data.mocap_pos[mocap_id], 3)}"
                )
                next_report += 0.5

            if terminated:
                fell = True
                print("PPO episode terminated at", round(t + dt, 2), "s")
                break

            if truncated:
                break

            if tilt > 25.0 or height < 0.45:
                fell = True
                print("Safety stop triggered.")
                break

        final_object = data.mocap_pos[mocap_id].copy()
        final_hand = data.xpos[hand_id].copy()

        print("\n========== GRASP AND LIFT RESULTS ==========")
        print("Grasp attachment created:", grasped)
        print("Raised IK pose found:", carry_arm_target is not None)
        print("Fell / safety stop:", fell)
        print("Maximum torso tilt:", round(max_tilt, 2), "degrees")
        print("Minimum root height:", round(min_height, 3), "m")
        print("Maximum horizontal drift:", round(max_drift, 3), "m")
        print("Initial object position:", np.round(OBJECT_START, 3))
        print("Final object position:", np.round(final_object, 3))

        if grasp_object_start is not None:
            displacement = final_object - grasp_object_start
            print("Object displacement:", np.round(displacement, 3))
            print("Object vertical lift:",
                  round(float(displacement[2]), 3), "m")

        print("Final hand position:", np.round(final_hand, 3))
        print("Selected carry joint targets:",
              None if carry_arm_target is None
              else np.round(carry_arm_target, 3))
        print("Actual final arm joint angles:",
              np.round(data.qpos[arm_qpos_ids], 3))
        print("Final arm control targets:",
              np.round(env.home_ctrl[arm_ids], 3))
        print("Trunk at grasp:",
              None if grasp_trunk_start is None
              else np.round(grasp_trunk_start, 3))
        print("Trunk at end:", np.round(data.xpos[trunk_id], 3))
        if grasp_trunk_start is not None:
            print("Hand relative to trunk at grasp:",
                  np.round(grasp_hand_start-grasp_trunk_start, 3))
            print("Hand relative to trunk at end:",
                  np.round(final_hand-data.xpos[trunk_id], 3))

    finally:
        env.close()


if __name__ == "__main__":
    main()
