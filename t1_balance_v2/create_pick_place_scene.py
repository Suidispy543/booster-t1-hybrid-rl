from pathlib import Path
import mujoco

asset_dir = (
    Path("../assets/mujoco_menagerie/booster_t1")
).resolve()

source = asset_dir / "scene.xml"
destination = asset_dir / "scene_pick_place.xml"

if not source.is_file():
    raise FileNotFoundError(source)

if destination.exists():
    raise FileExistsError(
        f"{destination} already exists; refusing to overwrite it."
    )

xml = source.read_text()

addition = """
    <!-- Pickup table A -->
    <body name="pickup_table_A" pos="0.56 -0.34 0.575">
      <geom name="table_A_top" type="box"
            size="0.36 0.27 0.025"
            rgba="0.32 0.34 0.38 1"
            contype="0" conaffinity="0"/>
      <geom name="table_A_leg_1" type="box"
            pos="-0.32 -0.23 -0.30"
            size="0.025 0.025 0.275"
            rgba="0.22 0.24 0.28 1"
            contype="0" conaffinity="0"/>
      <geom name="table_A_leg_2" type="box"
            pos="0.32 -0.23 -0.30"
            size="0.025 0.025 0.275"
            rgba="0.22 0.24 0.28 1"
            contype="0" conaffinity="0"/>
      <geom name="table_A_leg_3" type="box"
            pos="-0.32 0.23 -0.30"
            size="0.025 0.025 0.275"
            rgba="0.22 0.24 0.28 1"
            contype="0" conaffinity="0"/>
      <geom name="table_A_leg_4" type="box"
            pos="0.32 0.23 -0.30"
            size="0.025 0.025 0.275"
            rgba="0.22 0.24 0.28 1"
            contype="0" conaffinity="0"/>
    </body>

    <!-- Destination table B -->
    <body name="pickup_table_B" pos="1.56 -0.34 0.575">
      <geom name="table_B_top" type="box"
            size="0.36 0.27 0.025"
            rgba="0.32 0.34 0.38 1"
            contype="0" conaffinity="0"/>
      <geom name="table_B_leg_1" type="box"
            pos="-0.32 -0.23 -0.30"
            size="0.025 0.025 0.275"
            rgba="0.22 0.24 0.28 1"
            contype="0" conaffinity="0"/>
      <geom name="table_B_leg_2" type="box"
            pos="0.32 -0.23 -0.30"
            size="0.025 0.025 0.275"
            rgba="0.22 0.24 0.28 1"
            contype="0" conaffinity="0"/>
      <geom name="table_B_leg_3" type="box"
            pos="-0.32 0.23 -0.30"
            size="0.025 0.025 0.275"
            rgba="0.22 0.24 0.28 1"
            contype="0" conaffinity="0"/>
      <geom name="table_B_leg_4" type="box"
            pos="0.32 0.23 -0.30"
            size="0.025 0.025 0.275"
            rgba="0.22 0.24 0.28 1"
            contype="0" conaffinity="0"/>

      <!-- Green destination marker -->
      <geom name="drop_zone_B" type="cylinder"
            pos="0 0 0.027"
            size="0.075 0.002"
            rgba="0.1 0.8 0.25 0.9"
            contype="0" conaffinity="0"/>
    </body>

    <!-- Kinematic box: follows the right hand when attached -->
    <body name="pickup_object" mocap="true"
          pos="0.56 -0.20 0.635">
      <geom name="pickup_object_geom" type="box"
            size="0.025 0.025 0.035"
            rgba="0.9 0.12 0.08 1"
            contype="0" conaffinity="0"/>
    </body>
"""

if xml.count("</worldbody>") != 1:
    raise RuntimeError("Expected exactly one worldbody closing tag.")

xml = xml.replace("</worldbody>", addition + "\n  </worldbody>", 1)
destination.write_text(xml)

# Compile the new scene before using it.
try:
    model = mujoco.MjModel.from_xml_path(str(destination))
    obj_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pickup_object"
    )
    mocap_id = int(model.body_mocapid[obj_id])

    if mocap_id < 0:
        raise RuntimeError("Pickup object was not compiled as a mocap body.")

    print("\n===== PICK-AND-PLACE SCENE CREATED =====")
    print("Scene:", destination)
    print("Original scene.xml remains unchanged.")
    print("Robot actuators:", model.nu)
    print("Robot qpos:", model.nq)
    print("Mocap bodies:", model.nmocap)
    print("Pickup object mocap ID:", mocap_id)
    print("Pickup object geom:", mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "pickup_object_geom"
    ))
    print("Scene compilation: PASSED")

except Exception:
    destination.unlink(missing_ok=True)
    raise
