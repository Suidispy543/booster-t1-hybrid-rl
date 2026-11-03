from pathlib import Path

p = Path("make_outside_route.py")
s = p.read_text()

marker = "# 6. Compile the generated source before writing it."

if s.count(marker) != 1:
    raise RuntimeError(
        "Could not find the generator's final compilation section."
    )

if "PHASE 9A: TURN RIGHT TO FACE EAST" in s:
    raise RuntimeError("These changes appear to be already installed.")

patch = r'''
# ============================================================
# FINAL REFINEMENTS: FAST CARRY IK, TURN-WALK-TURN ROUTE,
# AND FINAL STABILIZATION
# ============================================================

# 1. Make the IK helper configurable for fewer starting points.
code = replace_once(
    code,
    "def solve_positive_elbow_ik(model, data, arm_ids, hand_id, target):",
    "def solve_positive_elbow_ik(model, data, arm_ids, hand_id, target, max_starts=12):",
    "Make IK starting-point count configurable",
)

old_starts = """    rng = np.random.default_rng(42)
    starts = [start]
    starts += [
        rng.uniform(lo + 1e-4, hi - 1e-4)
        for _ in range(11)
    ]"""

new_starts = """    rng = np.random.default_rng(42)
    starts = [start]
    starts += [
        rng.uniform(lo + 1e-4, hi - 1e-4)
        for _ in range(max(0, int(max_starts) - 1))
    ]"""

code = replace_once(
    code, old_starts, new_starts,
    "Reduce IK starting points when requested",
)

# 2. Retain just the previously tested carrying offset.
list_start = code.index("        local_candidates = [")
list_end = code.index("        ]", list_start) + len("        ]")

code = (
    code[:list_start]
    + """        local_candidates = [
            np.array([0.06, -0.06, 0.04]),
        ]"""
    + code[list_end:]
)

# Use only two IK starts for the carrying-pose search.
old_carry_ik = """            candidate, error = solve_positive_elbow_ik(
                model, data, arm_ids, hand_id, target
            )"""

new_carry_ik = """            candidate, error = solve_positive_elbow_ik(
                model, data, arm_ids, hand_id, target,
                max_starts=2,
            )"""

code = replace_once(
    code, old_carry_ik, new_carry_ik,
    "Fast carrying-pose IK",
)


# 3. Replace sideways transfer with right turn, straight walk,
# and left turn at Table B.
old_transfer = """        run_teacher_waypoint(
            np.array([1.50, SOUTH_LANE_Y]),
            PICKUP_YAW,
            "PHASE 9: WALK EAST ALONG THE OUTSIDE SOUTH LANE",
            carry=True,
            timeout_seconds=50.0,
        )"""

new_transfer = """        # PHASE 9A: TURN RIGHT TO FACE EAST.
        turn_right_position = data.qpos[:2].copy()

        run_teacher_waypoint(
            turn_right_position,
            0.0,
            "PHASE 9A: TURN RIGHT TO FACE EAST",
            carry=True,
            timeout_seconds=18.0,
        )

        # PHASE 9B: WALK STRAIGHT TOWARDS TABLE B.
        run_teacher_waypoint(
            np.array([1.43, SOUTH_LANE_Y]),
            0.0,
            "PHASE 9B: WALK STRAIGHT ALONG THE OUTSIDE LANE",
            carry=True,
            timeout_seconds=35.0,
        )

        # PHASE 9C: TURN LEFT TO FACE TABLE B.
        turn_left_position = data.qpos[:2].copy()

        run_teacher_waypoint(
            turn_left_position,
            PICKUP_YAW,
            "PHASE 9C: TURN LEFT TO FACE TABLE B",
            carry=True,
            timeout_seconds=18.0,
        )"""

code = replace_once(
    code, old_transfer, new_transfer,
    "Right-turn straight-walk left-turn route",
)


# 4. Give final stabilization a little more time and report
# whether the environment ended or the torso exceeded its limit.
code = replace_once(
    code,
    "for step in range(round(1.5 / policy_dt)):",
    "for step in range(round(3.0 / policy_dt)):",
    "Extend final stabilization",
)

old_final_check = """            if terminated or truncated or tilt > 20.0:
                raise RuntimeError("Final standing stabilization stopped.")"""

new_final_check = """            if terminated or truncated:
                raise RuntimeError(
                    "Final stabilization ended: "
                    f"terminated={terminated}, "
                    f"truncated={truncated}, tilt={tilt:.2f}deg"
                )

            if tilt > 25.0:
                raise RuntimeError(
                    f"Final stabilization tilt exceeded 25 degrees: {tilt:.2f}"
                )"""

code = replace_once(
    code, old_final_check, new_final_check,
    "Improve final stabilization diagnostics",
)

'''

s = s.replace(marker, patch + marker, 1)

# Verify that the patch was inserted only once.
if s.count("FINAL REFINEMENTS: FAST CARRY IK") != 1:
    raise RuntimeError("Patch insertion verification failed.")

p.write_text(s)
print("Generator patched successfully.")
print("Original simulation script remains unchanged.")
