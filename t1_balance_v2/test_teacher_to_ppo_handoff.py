from pathlib import Path
import numpy as np
import mujoco
import torch
import yaml
import base_env
from stable_baselines3 import PPO

SCENE = Path("../assets/mujoco_menagerie/booster_t1/scene_pick_place.xml").resolve()
REF = Path("booster_gym_reference")
CONFIG = yaml.safe_load((REF / "envs/T1.yaml").read_text())
TEACHER_PATH = REF / "deploy/models/T1.pt"
PPO_PATH = Path(
    "runs/stance_alpha0.5_20261028_215029_608658/"
    "checkpoint_000025600.zip"
)

WALK_SECONDS = 3.0
PPO_SECONDS = 5.0
WALK_SPEED = 0.07

ARM_NAMES = [
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]
PICKUP_ARM = np.array([-0.300, 1.382, 1.459, 0.664])


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


def tilt_degrees(data, trunk_id):
    R = data.xmat[trunk_id].reshape(3, 3)
    return float(np.degrees(np.arccos(
        np.clip(R[2, 2], -1.0, 1.0)
    )))


def main():
    if not SCENE.exists():
        raise FileNotFoundError(SCENE)
    if not TEACHER_PATH.exists():
        raise FileNotFoundError(TEACHER_PATH)
    if not PPO_PATH.exists():
        raise FileNotFoundError(PPO_PATH)

    base_env.MODEL_PATH = SCENE
    from balance_env import BalanceEnv

    teacher = torch.jit.load(
        str(TEACHER_PATH), map_location="cpu"
    ).eval()
    ppo = PPO.load(str(PPO_PATH), device="cpu")

    env = BalanceEnv(
        episode_seconds=12.0,
        reset_noise=0.0,
        alpha=1.0,
        stance_weight=0.0,
        drift_weight=0.0,
        rate_weight=0.0,
        joint_velocity_weight=0.0,
    )

    try:
        model, data = env.model, env.data
        native_home = env.home_ctrl.copy()
        native_gain = model.actuator_gainprm.copy()
        native_bias = model.actuator_biasprm.copy()
        native_action_scale = env.action_scale

        leg_ids = env.lower_body_ids.copy()
        leg_joints = model.actuator_trnid[leg_ids, 0].astype(int)
        leg_qpos = model.jnt_qposadr[leg_joints].astype(int)
        leg_qvel = model.jnt_dofadr[leg_joints].astype(int)

        arm_ids = np.array([
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            for name in ARM_NAMES
        ])
        trunk_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "Trunk"
        )
        hand_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_link"
        )

        if np.any(arm_ids < 0) or min(trunk_id, hand_id) < 0:
            raise RuntimeError("Required actuator or body missing.")

        arm_home = native_home[arm_ids].copy()

        # Configure reference teacher gains and joint defaults.
        defaults = CONFIG["init_state"]["default_joint_angles"]
        stiffness = CONFIG["control"]["stiffness"]
        damping = CONFIG["control"]["damping"]
        norm = CONFIG["normalization"]
        teacher_scale = float(CONFIG["control"]["action_scale"])
        decimation = int(CONFIG["control"]["decimation"])
        frequency = float(np.mean(
            CONFIG["commands"]["gait_frequency"]
        ))
        dt = model.opt.timestep * decimation

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
                raise RuntimeError(f"No teacher PD gains for {name}")

            kp = float(stiffness[gain_key])
            kd = float(damping[gain_key])
            default_pos[i] = value
            env.home_ctrl[aid] = value

            model.actuator_gainprm[aid, 0] = kp
            model.actuator_biasprm[aid, 1] = -kp
            model.actuator_biasprm[aid, 2] = -kd

        env.action_scale = teacher_scale
        env.alpha = 1.0

        obs, _ = env.reset(seed=42)
        phase = 0.0
        teacher_previous = np.zeros(12, dtype=np.float32)

        initial_xy = data.qpos[:2].copy()
        max_tilt = 0.0
        fell = False

        print("========== PHASE 1: TEACHER WALK ==========")

        for step in range(round(WALK_SECONDS / dt)):
            t = step * dt

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
            observation[6] = WALK_SPEED * norm["lin_vel"]
            observation[7] = 0.0
            observation[8] = 0.0
            observation[9] = np.cos(2*np.pi*phase)
            observation[10] = np.sin(2*np.pi*phase)
            observation[11:23] = (
                pos - default_pos
            ) * norm["dof_pos"]
            observation[23:35] = vel * norm["dof_vel"]
            observation[35:47] = teacher_previous

            with torch.no_grad():
                action = teacher(
                    torch.from_numpy(observation).unsqueeze(0)
                ).squeeze(0).cpu().numpy().astype(np.float32)

            action = np.clip(
                action, -norm["clip_actions"], norm["clip_actions"]
            )
            obs, _, terminated, truncated, _ = env.step(action)
            teacher_previous = action.copy()
            phase = (phase + dt * frequency) % 1.0

            current_tilt = tilt_degrees(data, trunk_id)
            max_tilt = max(max_tilt, current_tilt)

            if step % 50 == 0:
                print(
                    f"t={t:.1f}s | height={data.qpos[2]:.3f}m | "
                    f"tilt={current_tilt:.2f}deg | "
                    f"position={np.round(data.qpos[:2], 3)}"
                )

            if terminated or truncated or current_tilt > 30.0:
                fell = bool(terminated or current_tilt > 30.0)
                print("Teacher phase stopped early.")
                break

        if fell:
            raise RuntimeError("Teacher phase failed; PPO handoff skipped.")

        switch_xy = data.qpos[:2].copy()
        hand_before = data.xpos[hand_id].copy()

        print("\n========== CONTROLLER HANDOFF ==========")

        # Restore the exact model control parameters used by the PPO policy.
        model.actuator_gainprm[:] = native_gain
        model.actuator_biasprm[:] = native_bias
        env.home_ctrl[:] = native_home
        env.action_scale = native_action_scale
        env.alpha = 0.5

        # Start PPO's drift and stance references at the current live state.
        env.origin_xy = data.qpos[:2].copy()
        env.home_width = env.width()
        env.previous_action.fill(0.0)

        # Build the exact 72-element BalanceEnv observation from live state.
        obs = env._observation()

        if obs.shape != ppo.observation_space.shape:
            raise RuntimeError(
                f"PPO observation mismatch: {obs.shape}"
            )

        print("Robot state preserved; no simulation reset.")
        print("Switch position:", np.round(switch_xy, 3))
        print("Hand before PPO:", np.round(hand_before, 3))
        print("PPO observation:", obs.shape)

        print("\n========== PHASE 2: PPO STABILISE + REACH ==========")

        for step in range(round(PPO_SECONDS / dt)):
            t = step * dt

            # Reach gradually, then hold the pickup arm pose.
            blend = smooth(t / 2.0)
            env.home_ctrl[arm_ids] = (
                arm_home + blend * (PICKUP_ARM - arm_home)
            )

            action, _ = ppo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            current_tilt = tilt_degrees(data, trunk_id)
            max_tilt = max(max_tilt, current_tilt)
            drift = float(np.linalg.norm(data.qpos[:2] - switch_xy))

            if step % 50 == 0:
                print(
                    f"t={t:.1f}s | height={data.qpos[2]:.3f}m | "
                    f"tilt={current_tilt:.2f}deg | "
                    f"drift_after_switch={drift:.3f}m | "
                    f"hand={np.round(data.xpos[hand_id], 3)}"
                )

            if terminated or truncated:
                fell = bool(terminated)
                print("PPO phase ended early.")
                break

            if current_tilt > 20.0 or data.qpos[2] < 0.45:
                fell = True
                print("PPO safety stop triggered.")
                break

        final_xy = data.qpos[:2].copy()
        final_hand = data.xpos[hand_id].copy()

        print("\n========== HANDOFF TEST RESULTS ==========")
        print("Fall / safety stop:", fell)
        print("Teacher walk displacement:",
              np.round(switch_xy - initial_xy, 3))
        print("PPO-phase displacement:",
              np.round(final_xy - switch_xy, 3))
        print("Maximum torso tilt:", round(max_tilt, 2), "degrees")
        print("Final height:", round(float(data.qpos[2]), 3), "m")
        print("Hand before PPO:", np.round(hand_before, 3))
        print("Hand after PPO:", np.round(final_hand, 3))
        print("PPO observation/action:",
              ppo.observation_space.shape, ppo.action_space.shape)

    finally:
        env.close()


if __name__ == "__main__":
    main()
