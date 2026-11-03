import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(
    "assets/mujoco_menagerie/booster_t1/scene.xml"
)
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
data.ctrl[:] = model.key_ctrl[0]

print("Simulating the T1 home pose for 3 seconds...\n")

next_report = 0.0

while data.time < 3.0:
    mujoco.mj_step(model, data)

    if data.time + 1e-9 >= next_report:
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
        rotation = rotation.reshape(3, 3)

        contact_geoms = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            contact_geoms.add(model.geom(contact.geom1).name or str(contact.geom1))
            contact_geoms.add(model.geom(contact.geom2).name or str(contact.geom2))

        print(
            f"t={data.time:.2f}s | "
            f"z={data.qpos[2]:.3f} m | "
            f"upright={rotation[2, 2]:.3f} | "
            f"contacts={sorted(contact_geoms)}"
        )
        next_report += 0.25

print("\nTest complete.")
