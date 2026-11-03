from pathlib import Path


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected 1 match, found {count}. "
            "No ambiguous replacement was made."
        )
    return text.replace(old, new, 1)


def patch_source(source):
    if "hold_position=False" in source:
        return source

    start = source.index("        def run_teacher_phase(")
    end = source.index("\n\n        def run_teacher_waypoint(", start)
    helper = source[start:end]

    helper = replace_once(
        helper,
        "def run_teacher_phase(seconds, vx, vy, yaw, label, carry_offset=None, arm_target=None):",
        "def run_teacher_phase(seconds, vx, vy, yaw, label, carry_offset=None, arm_target=None, hold_position=False):",
        "Add position-hold option",
    )

    helper = replace_once(
        helper,
        "            for i in range(count):",
        """            hold_xy = data.qpos[:2].copy() if hold_position else None
            R_hold = data.xmat[trunk_id].reshape(3, 3)
            hold_yaw = float(np.arctan2(R_hold[1, 0], R_hold[0, 0]))

            for i in range(count):""",
        "Capture the base position and heading",
    )

    helper = replace_once(
        helper,
        "                teacher_obs = np.zeros(47, dtype=np.float32)",
        """                # Closed-loop base position and heading hold.
                if hold_position:
                    delta_world = hold_xy - data.qpos[:2]
                    distance = float(np.linalg.norm(delta_world))

                    if distance < 0.025:
                        world_vx = 0.0
                        world_vy = 0.0
                    else:
                        speed = min(0.07, 1.2 * distance)
                        world_vx = delta_world[0] / distance * speed
                        world_vy = delta_world[1] / distance * speed

                    R_now = data.xmat[trunk_id].reshape(3, 3)
                    yaw_now = float(np.arctan2(R_now[1, 0], R_now[0, 0]))
                    yaw_error = (
                        hold_yaw - yaw_now + np.pi
                    ) % (2.0 * np.pi) - np.pi

                    yaw_command = float(np.clip(
                        1.6 * yaw_error, -0.14, 0.14
                    ))
                    if abs(yaw_error) < 0.018:
                        yaw_command = 0.0

                    c = np.cos(yaw_now)
                    s = np.sin(yaw_now)
                    vx_command = c * world_vx + s * world_vy
                    vy_command = -s * world_vx + c * world_vy
                else:
                    vx_command = vx
                    vy_command = vy
                    yaw_command = yaw

                teacher_obs = np.zeros(47, dtype=np.float32)""",
        "Add closed-loop velocity commands",
    )

    helper = replace_once(
        helper,
        "teacher_obs[6] = vx * norm[\"lin_vel\"]",
        "teacher_obs[6] = vx_command * norm[\"lin_vel\"]",
        "Use corrected forward velocity",
    )
    helper = replace_once(
        helper,
        "teacher_obs[7] = vy * norm[\"lin_vel\"]",
        "teacher_obs[7] = vy_command * norm[\"lin_vel\"]",
        "Use corrected lateral velocity",
    )
    helper = replace_once(
        helper,
        "teacher_obs[8] = yaw * norm[\"ang_vel\"]",
        "teacher_obs[8] = yaw_command * norm[\"ang_vel\"]",
        "Use corrected yaw command",
    )

    source = source[:start] + helper + source[end:]

    # Hold the base during all stationary carrying and placement phases.
    carry_call = (
        "            carry_offset=attached_local,\n"
        "        )"
    )
    carry_count = source.count(carry_call)
    if carry_count < 1:
        raise RuntimeError("Could not find the stationary carrying calls.")

    source = source.replace(
        carry_call,
        "            carry_offset=attached_local,\n"
        "            hold_position=True,\n"
        "        )",
    )

    source = replace_once(
        source,
        "            arm_target=final_arm_target,\n        )",
        "            arm_target=final_arm_target,\n"
        "            hold_position=True,\n"
        "        )",
        "Hold position during arm retraction",
    )

    source = replace_once(
        source,
        '            "PHASE 15B: FINAL TEACHER BALANCE HOLD",\n'
        "        )",
        '            "PHASE 15B: FINAL TEACHER BALANCE HOLD",\n'
        "            hold_position=True,\n"
        "        )",
        "Hold position during final stabilization",
    )

    compile(source, "integrated_table_a_to_b_outside_route.py", "exec")
    return source


if __name__ == "__main__":
    print("This module is imported by make_outside_route.py.")
