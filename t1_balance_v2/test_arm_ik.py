import mujoco
import numpy as np
from scipy.optimize import least_squares
from base_env import MODEL_PATH

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

if model.nkey < 1:
    raise RuntimeError("No home keyframe found in the T1 model.")

base_qpos = model.key_qpos[0].copy()

hand = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_link"
)
names = [
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]
joints = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    for name in names
]

if hand < 0 or any(j < 0 for j in joints):
    raise RuntimeError("Required hand body or arm joint not found.")

qadr = np.array([model.jnt_qposadr[j] for j in joints], dtype=int)

for joint in joints:
    if not model.jnt_limited[joint]:
        raise RuntimeError("Expected bounded arm joint.")

lo = np.array([model.jnt_range[j, 0] for j in joints])
hi = np.array([model.jnt_range[j, 1] for j in joints])


def solve_target(target, seed_count=30):
    def position_error(angles):
        data.qpos[:] = base_qpos
        data.qpos[qadr] = angles
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        return data.xpos[hand].copy() - target

    rng = np.random.default_rng(42)
    starts = [np.clip(base_qpos[qadr], lo + 1e-5, hi - 1e-5)]
    starts += [
        rng.uniform(lo + 1e-4, hi - 1e-4)
        for _ in range(seed_count - 1)
    ]

    best = None
    for start in starts:
        result = least_squares(
            position_error, start, bounds=(lo, hi), max_nfev=500
        )
        error = np.linalg.norm(position_error(result.x))
        if best is None or error < best[0]:
            best = (error, result.x.copy())

    error, angles = best
    data.qpos[:] = base_qpos
    data.qpos[qadr] = angles
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    print("\nTarget:", np.round(target, 3))
    print("Best hand position:", np.round(data.xpos[hand], 3))
    print(f"Position error: {error:.4f} m")
    print("Joint angles (rad):")
    for name, value in zip(names, angles):
        print(f"  {name}: {value:+.3f}")

    return error


print("===== RIGHT ARM REACHABILITY TEST =====")
e1 = solve_target(np.array([0.28, -0.22, 0.57]))
e2 = solve_target(np.array([0.13, -0.20, 0.65]))

print("\n===== SUMMARY =====")
print(f"Target 1 error: {e1:.4f} m")
print(f"Target 2 error: {e2:.4f} m")
print("A small error indicates positional reachability, not collision-free grasping.")
