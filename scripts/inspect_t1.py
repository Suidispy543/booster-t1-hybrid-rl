"""Load the official Booster T1 MJCF and report its joints and actuators."""
from pathlib import Path
import sys

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "mujoco_menagerie" / "booster_t1" / "scene.xml"


def name_or_fallback(model, object_type, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name else f"unnamed_{object_id}"


def main() -> int:
    if not MODEL_PATH.is_file():
        print("T1 model file not found:", MODEL_PATH)
        print("Follow the README section 'Fetch the Booster T1 MuJoCo model'.")
        return 1

    try:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    except Exception as exc:  # include MuJoCo's XML/asset error for diagnosis
        print(f"Failed to load the T1 model: {exc}")
        return 2

    print("=== Booster T1 MuJoCo model loaded successfully ===")
    print(f"Model file : {MODEL_PATH}")
    print(f"MuJoCo     : {mujoco.__version__}")
    print(f"Bodies     : {model.nbody}")
    print(f"Joints     : {model.njnt}")
    print(f"qpos size  : {model.nq}")
    print(f"qvel size  : {model.nv}")
    print(f"Actuators  : {model.nu}")
    print(f"Sensors    : {model.nsensor}")
    print("\nJoint names:")
    for idx in range(model.njnt):
        print(f"  {idx:02d}: {name_or_fallback(model, mujoco.mjtObj.mjOBJ_JOINT, idx)}")
    print("\nActuator names:")
    for idx in range(model.nu):
        print(f"  {idx:02d}: {name_or_fallback(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)}")

    print("\nNext: run `python scripts/view_t1.py` to open the viewer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
