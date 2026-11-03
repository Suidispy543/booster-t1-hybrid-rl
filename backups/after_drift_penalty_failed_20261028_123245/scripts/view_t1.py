"""Open the Booster T1 MuJoCo model in the passive interactive viewer."""
from pathlib import Path
import time

import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "mujoco_menagerie" / "booster_t1" / "scene.xml"


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"T1 model not found at {MODEL_PATH}. Follow the README setup steps first."
        )

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Use a model-provided keyframe (commonly the home pose) if one exists.
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        # Apply the actuator targets paired with the home keyframe. Without
        # this, ctrl remains at zero and the joints drift away from home.
        data.ctrl[:] = model.key_ctrl[0]
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print("Viewer opened. Close the viewer window or press Ctrl+C to stop.")
    print("This is a model/physics smoke test, not a trained walking policy.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            delay = model.opt.timestep - (time.perf_counter() - start)
            if delay > 0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
