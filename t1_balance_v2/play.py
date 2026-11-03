"""macOS: mjpython t1_balance_v2/play.py --model PATH_TO_V2_CHECKPOINT"""
import argparse
import time
import numpy as np

import mujoco.viewer
from balance_env import BalanceEnv
from evaluate import HERE,load_run



def angles_deg(R):
    R = np.asarray(R, dtype=float).reshape(3, 3)
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    pitch = np.degrees(
        np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
    )
    return float(roll), float(pitch)


def joint_deg(env, name):
    jid = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    if jid < 0:
        raise ValueError(f"Joint not found: {name}")
    return float(np.degrees(
        env.data.qpos[env.model.jnt_qposadr[jid]]
    ))


def target_deg(env, name):
    aid = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
    )
    if aid < 0:
        raise ValueError(f"Actuator not found: {name}")
    return float(np.degrees(env.data.ctrl[aid]))

def actuator_force_info(env, name):
    aid = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
    )
    jid = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, name
    )

    if aid < 0 or jid < 0:
        return f"{name}: actuator or joint not found"

    dof = int(env.model.jnt_dofadr[jid])
    force = float(env.data.actuator_force[aid])
    joint_force = float(env.data.qfrc_actuator[dof])

    actuator_limited = bool(env.model.actuator_forcelimited[aid])
    actuator_range = env.model.actuator_forcerange[aid].tolist()

    joint_limited_flags = getattr(
        env.model, "jnt_actfrclimited", None
    )
    joint_ranges = getattr(env.model, "jnt_actfrcrange", None)

    if joint_limited_flags is not None and joint_ranges is not None:
        joint_limited = bool(joint_limited_flags[jid])
        joint_range = joint_ranges[jid].tolist()
    else:
        joint_limited = "unavailable"
        joint_range = "unavailable"

    return (
        f"{name}: actuator_force={force:+.2f}, "
        f"joint_actuator_force={joint_force:+.2f}, "
        f"actuator_force_limited={actuator_limited} "
        f"range={actuator_range}, "
        f"joint_force_limited={joint_limited} "
        f"range={joint_range}"
    )

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=str(HERE/'selected/model.zip'))
    p.add_argument('--seconds',type=float,default=180.0)
    p.add_argument('--loop',action='store_true')
    args=p.parse_args()
    if args.seconds<0.04 or not __import__('math').isfinite(args.seconds):
        p.error('Require finite --seconds >= 0.04')
    model,config,path=load_run(args.model)
    env=BalanceEnv(episode_seconds=args.seconds,reset_noise=0.,**config)
    try:
        obs, _ = env.reset(seed=42)
        dt = env.model.opt.timestep * env.frame_skip

        R0 = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(R0, env.data.qpos[3:7])
        R0 = R0.reshape(3, 3)
        roll0, pitch0 = angles_deg(R0)

        print(
            f"RESET root roll/pitch="
            f"{roll0:+.2f}/{pitch0:+.2f}deg "
            f"upright={R0[2, 2]:.4f}"
        )

        for name in (
            "Left_Hip_Roll", "Right_Hip_Roll",
            "Left_Ankle_Roll", "Right_Ankle_Roll",
            "Left_Hip_Pitch", "Right_Hip_Pitch",
            "Left_Ankle_Pitch", "Right_Ankle_Pitch",
        ):
            print(
                f"RESET {name}: "
                f"actual={joint_deg(env, name):+.2f}deg "
                f"target={target_deg(env, name):+.2f}deg"
            )
        print(f'Model: {path}\nAction filter alpha={env.alpha}; NoSlip={env.model.opt.noslip_iterations}; duration={args.seconds:g}s.')
        with mujoco.viewer.launch_passive(env.model,env.data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:]=[0.,0.,.55];viewer.cam.distance=1.8;viewer.cam.azimuth=135;viewer.cam.elevation=-15
            while viewer.is_running():
                started=time.perf_counter()
                with viewer.lock():
                    action,_=model.predict(obs,deterministic=True)
                    obs,_,term,trunc,info=env.step(action)
                viewer.sync()
                if env.steps % 250 == 0 or term or trunc:
                    root_matrix = np.zeros(9, dtype=np.float64)
                    mujoco.mju_quat2Mat(
                        root_matrix, env.data.qpos[3:7]
                    )
                    root_R = root_matrix.reshape(3, 3)

                    root_roll, root_pitch = angles_deg(root_R)
                    left_roll, left_pitch = angles_deg(
                        env.data.xmat[env.feet[0]]
                    )
                    right_roll, right_pitch = angles_deg(
                        env.data.xmat[env.feet[1]]
                    )

                    print(
                        f"t={env.data.time:.2f}s "
                        f"width={info['foot_width_m']:.3f}m "
                        f"drift={info['root_displacement_m']:.3f}m "
                        f"height={info['root_height']:.3f}m"
                    )
                    print(
                        f"  ROOT roll/pitch="
                        f"{root_roll:+.2f}/{root_pitch:+.2f}deg "
                        f"upright={root_R[2, 2]:.4f}"
                    )
                    print(
                        f"  FEET world roll/pitch: "
                        f"L={left_roll:+.2f}/{left_pitch:+.2f}deg "
                        f"R={right_roll:+.2f}/{right_pitch:+.2f}deg"
                    )

                    for side in ("Left", "Right"):
                        for joint in ("Hip_Roll", "Ankle_Roll",
                                      "Hip_Pitch", "Ankle_Pitch"):
                            name = f"{side}_{joint}"
                            actual = joint_deg(env, name)
                            target = target_deg(env, name)
                            print(
                                f"  {name}: actual={actual:+.2f}deg "
                                f"target={target:+.2f}deg"
                            )
                    for name in (
                        "Left_Ankle_Roll",
                        "Right_Ankle_Roll",
                        "Left_Ankle_Pitch",
                        "Right_Ankle_Pitch",
                        "Left_Hip_Roll",
                        "Right_Hip_Roll",
                    ):
                        print("  FORCE " + actuator_force_info(env, name))
                time.sleep(max(0.,dt-(time.perf_counter()-started)))
                if term or trunc:
                    print('Fall threshold reached' if term else 'Time limit reached',flush=True)
                    if not args.loop:break
                    with viewer.lock():obs,_=env.reset(seed=42)
    finally:env.close()


if __name__=='__main__':main()
