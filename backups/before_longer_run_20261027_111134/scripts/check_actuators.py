import mujoco
import numpy as np

path = "assets/mujoco_menagerie/booster_t1/scene.xml"
model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
data.ctrl[:] = model.key_ctrl[0]
mujoco.mj_forward(model, data)

def report(label):
    print(f"\n--- {label}: t={data.time:.2f}s ---")
    print(f"Root height: {data.qpos[2]:.3f} m")
    print("Actuator | Joint | Target | Actual | Error | Force")

    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        qpos_index = model.jnt_qposadr[joint_id]
        target = data.ctrl[i]
        actual = data.qpos[qpos_index]

        print(
            f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):24s}"
            f" {joint_name:24s}"
            f" {target:7.3f} {actual:7.3f}"
            f" {target - actual:7.3f} {data.actuator_force[i]:8.2f}"
        )

report("Initial")

checkpoints = [1.0, 1.5, 2.0]
for target_time in checkpoints:
    while data.time < target_time:
        mujoco.mj_step(model, data)
    report(f"Checkpoint {target_time:.1f}s")

print("\nDiagnostic complete.")
