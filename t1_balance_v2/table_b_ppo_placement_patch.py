import re


PHASE_BLOCK = r'''
        # =========================================================
        # TABLE_B_TEACHER_POSITION_HOLD_PLACEMENT
        print("\nPHASE 12: KEEP THE WALKING TEACHER ACTIVE AT TABLE B")

        # Restore the teacher's original controller settings.
        model.actuator_gainprm[:] = teacher_gain
        model.actuator_biasprm[:] = teacher_bias
        env.home_ctrl[:] = teacher_home
        env.action_scale = teacher_scale
        env.alpha = 1.0

        # Preserve the current right-arm carrying posture.
        arm_joint_ids = model.actuator_trnid[arm_ids, 0].astype(int)
        arm_qpos_ids = model.jnt_qposadr[arm_joint_ids].astype(int)
        arm_start = data.qpos[arm_qpos_ids].copy()
        env.home_ctrl[arm_ids] = arm_start

        # Hold the base position and heading while carrying the box.
        run_teacher_phase(
            2.5, 0.0, 0.0, 0.0,
            "PHASE 12A: TEACHER POSITION- AND HEADING-HOLD",
            carry_offset=attached_local,
            hold_position=True,
        )

        # Desired area is inside Table B, away from its south edge.

        REQUESTED_PLACE_TARGET = np.array([1.680, -0.535, 0.635], dtype=float)

        
        # Move to a closer, aligned stance in front of the green marker.
        # The robot remains south of Table B throughout the approach.
        R_place = data.xmat[trunk_id].reshape(3, 3)
        place_yaw = float(np.arctan2(R_place[1, 0], R_place[0, 0]))
        forward = np.array([np.cos(place_yaw), np.sin(place_yaw)])

        desired_stance_xy = (
            REQUESTED_PLACE_TARGET[:2] - 0.46 * forward
        )
        desired_stance_xy[1] = min(desired_stance_xy[1], -0.82)

        stance_shift = float(np.linalg.norm(
            desired_stance_xy - data.qpos[:2]
        ))

        print("Green marker centre:", np.round(REQUESTED_PLACE_TARGET, 4))
        print("Table B approach stance:", np.round(desired_stance_xy, 4))
        print("Required base shift:", round(stance_shift, 4), "m")

        if stance_shift > 0.06:
            run_teacher_waypoint(
                desired_stance_xy,
                place_yaw,
                "PHASE 12B: ALIGN WITH TABLE B GREEN MARK",
                carry=True,
                timeout_seconds=20.0,
            )

        # Estimate the carried box footprint for edge-clearance checks.
        box_geom_ids = [
            gid for gid in range(model.ngeom)
            if int(model.geom_bodyid[gid]) == int(object_id)
            and int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX)
        ]

        box_half_x = max(
            (float(model.geom_size[gid, 0]) for gid in box_geom_ids),
            default=0.035,
        )
        box_half_y = max(
            (float(model.geom_size[gid, 1]) for gid in box_geom_ids),
            default=0.035,
        )
        table_edge_clearance = 0.012

        saved_qpos = data.qpos.copy()
        saved_qvel = data.qvel.copy()

        R_hand_now = data.xmat[hand_id].reshape(3, 3).copy()
        attached_offset_world = R_hand_now @ attached_local
        hand_target = REQUESTED_PLACE_TARGET - attached_offset_world

        place_arm, place_ik_error = solve_positive_elbow_ik(
            model, data, arm_ids, hand_id, hand_target
        )

        # Preview the actual reachable box position with the solved arm pose.
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        data.qpos[arm_qpos_ids] = place_arm
        mujoco.mj_forward(model, data)

        R_preview = data.xmat[hand_id].reshape(3, 3).copy()
        predicted_object = (
            palm_position(data, hand_id) + R_preview @ attached_local
        )
        predicted_error = predicted_object - REQUESTED_PLACE_TARGET

                # Validate the final box footprint on Table B.
        box_geom_ids = [
            gid for gid in range(model.ngeom)
            if int(model.geom_bodyid[gid]) == int(object_id)
            and int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX)
        ]

        if not box_geom_ids:
            raise RuntimeError("Could not determine box footprint.")

        box_half_xy = np.max(
            model.geom_size[box_geom_ids, :2], axis=0
        )
        edge_margin = 0.012

        safe_x = (
            1.20 + box_half_xy[0] + edge_margin
            <= REQUESTED_PLACE_TARGET[0]
            <= 1.92 - box_half_xy[0] - edge_margin
        )
        safe_y = (
            -0.61 + box_half_xy[1] + edge_margin
            <= REQUESTED_PLACE_TARGET[1]
            <= -0.07 - box_half_xy[1] - edge_margin
        )
        safe_z = abs(REQUESTED_PLACE_TARGET[2] - 0.635) <= 0.005

        preview_horizontal_error = float(
            np.linalg.norm(predicted_error[:2])
        )
        preview_vertical_error = float(abs(predicted_error[2]))

        preview_close = (
            preview_horizontal_error <= 0.09
            and preview_vertical_error <= 0.08
        )

        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        mujoco.mj_forward(model, data)

        print("Placement arm targets:", np.round(place_arm, 3))
        print("Placement IK residual:", round(place_ik_error, 4), "m")
        print("Requested box target:",
              np.round(REQUESTED_PLACE_TARGET, 3))
        print("Reachable box preview:",
              np.round(predicted_object, 3))
        print("Horizontal difference from request:",
              round(float(np.linalg.norm(predicted_error[:2])), 4), "m")
        print("Vertical difference from request:",
              round(float(abs(predicted_error[2])), 4), "m")

        if place_ik_error > 0.10:
            raise RuntimeError(
                "Placement IK residual exceeds 0.10 m. "
                "The box will not be released."
            )

        if not (safe_x and safe_y and safe_z and preview_close):
            raise RuntimeError(
                "The arm's reachable placement is outside the safe "
                "interior of Table B. The box will not be released. "
                f"Preview={np.round(predicted_object, 3)}"
            )

        # Use the reachable XY only after verifying the box has adequate
        # clearance from all Table B edges. The final vertical set-down
        # remains scripted because the object is a mocap body.
        PLACE_TARGET = REQUESTED_PLACE_TARGET.copy()

        print("Validated interior release target:",
              np.round(PLACE_TARGET, 3))

        # =========================================================
        # Release at the exact green-marker centre after IK validation.
        PLACE_TARGET = REQUESTED_PLACE_TARGET.copy()
        print("Exact green-marker box target:", np.round(PLACE_TARGET, 4))

        # PHASE 13: LOWER THE BOX WITH PPO HOLDING THE LEGS STATIONARY
        # =========================================================
        print("\nPHASE 13: PPO-STABILIZED RIGHT-ARM PLACEMENT")

        run_teacher_phase(
            4.5, 0.0, 0.0, 0.0,
            "PHASE 13: LOWER BOX ONTO TABLE B",
            carry_offset=attached_local,
            arm_target=place_arm,
            hold_position=True,
        )

        actual_box = data.mocap_pos[mocap_id].copy()
        horizontal_error = float(np.linalg.norm(
            actual_box[:2] - PLACE_TARGET[:2]
        ))
        vertical_error = float(abs(
            actual_box[2] - PLACE_TARGET[2]
        ))

        print("Box position before release:", np.round(actual_box, 3))
        print("Horizontal placement error:",
              round(horizontal_error, 4), "m")
        print("Vertical placement error:",
              round(vertical_error, 4), "m")

        actual_safe = bool(safe_x and safe_y and safe_z)

        if not actual_safe or horizontal_error > 0.12 or vertical_error > 0.12:
            raise RuntimeError(
                "Box did not remain in the validated Table B placement area. "
                "Refusing release. "
                f"Actual={np.round(actual_box, 3)}"
            )

        # =========================================================
        # PHASE 14: RELEASE THE KINEMATIC OBJECT
        # =========================================================
        print("\nPHASE 14: RELEASE BOX ON TABLE B")

        data.mocap_pos[mocap_id] = PLACE_TARGET.copy()
        data.mocap_quat[mocap_id] = np.array([1., 0., 0., 0.])
        mujoco.mj_forward(model, data)

        print("Released box position:", np.round(data.mocap_pos[mocap_id], 3))
        print("Tabletop surface: z=0.600m")
        print("Box bottom: z=0.600m")
        print("Kinematic attachment: released")
        print("Release method: scripted mocap placement")

        # =========================================================
        # =========================================================

        # =========================================================

        # =========================================================

        # =========================================================
        # PHASE 15: LIFT ARM AND RETURN TO NORMAL HOME POSITION
        # =========================================================
        print("\nPHASE 15: RETRACT ARM TO NORMAL POSITION")

        current_hand = palm_position(data, hand_id).copy()
        retreat_target = current_hand + np.array([0.0, 0.0, 0.12])

        retreat_arm, retreat_error = solve_positive_elbow_ik(
            model, data, arm_ids, hand_id, retreat_target
        )

        print("Arm-clearance IK error:", round(retreat_error, 4), "m")

        if retreat_error > 0.10:
            raise RuntimeError(
                "Could not validate a safe arm-clearance pose."
            )

        run_teacher_phase(
            2.0, 0.0, 0.0, 0.0,
            "PHASE 15A: LIFT ARM CLEAR OF TABLE B",
            arm_target=retreat_arm,
            hold_position=True,
        )

        run_teacher_phase(
            3.5, 0.0, 0.0, 0.0,
            "PHASE 15B: RETURN RIGHT ARM TO NORMAL HOME POSITION",
            arm_target=home_arm,
            hold_position=True,
        )

        run_teacher_phase(
            2.0, 0.0, 0.0, 0.0,
            "PHASE 15C: FINAL POSITION AND HEADING HOLD",
            hold_position=True,
        )

        print(
            "Right-arm home-pose joint error:",
            round(float(np.linalg.norm(
                data.qpos[arm_qpos_ids] - home_arm
            )), 4),
            "rad"
        )

        final_hand = palm_position(data, hand_id).copy()
        final_box = data.mocap_pos[mocap_id].copy()

        print("\n========== TABLE B PLACEMENT RESULTS ==========")
        print("Final robot position:", np.round(data.qpos[:3], 3))
        print("Final robot heading:", round(current_yaw_deg(), 2), "degrees")
        print("Final hand position:", np.round(final_hand, 3))
        print("Final box position:", np.round(final_box, 3))
        print("Box target:", np.round(PLACE_TARGET, 3))
        print("Final box placement error:",
              round(float(np.linalg.norm(final_box - PLACE_TARGET)), 4), "m")
        print("Maximum torso tilt:", round(max_tilt, 2), "degrees")
        print("Walking-teacher position hold: completed")
'''


def patch_source(source):
    marker = "TABLE_B_TEACHER_POSITION_HOLD_PLACEMENT"

    if marker in source:
        return source

    # Remove the extra zero-velocity teacher phases. The teacher is for
    # actual locomotion and turning, not for standing still.
    labels = [
        "PHASE 8B: SETTLE AFTER LIFT",
        "PHASE 9A: SETTLE AFTER RIGHT TURN",
        "PHASE 9B: SETTLE AFTER WALK",
        "PHASE 9C: SETTLE AFTER LEFT TURN",
    ]

    for label in labels:
        pattern = (
            r'(?m)^[ \t]*run_teacher_phase\(\s*'
            r'(?:0\.8|1\.5),\s*0\.0,\s*0\.0,\s*0\.0,\s*'
            + re.escape('"' + label + '"')
            + r',\s*carry_offset=attached_local,\s*'
            r'(?:hold_position=True,\s*)?\)\s*\n'
        )
        source = re.sub(pattern, "", source)

    # Move the Table B stance slightly west and south to improve clearance.
    old_route = "np.array([1.42, SOUTH_LANE_Y])"
    new_route = "np.array([1.38, SOUTH_LANE_Y - 0.04])"

    if old_route in source:
        source = source.replace(old_route, new_route, 1)
    elif new_route not in source:
        raise RuntimeError(
            "Could not locate the Table B waypoint. "
            "Expected the current [1.42, SOUTH_LANE_Y] route."
        )

    start_marker = (
        "        # PHASE 12: KEEP THE WALKING TEACHER ACTIVE AT TABLE B"
    )
    end_marker = "        object_final = data.mocap_pos[mocap_id].copy()"

    if source.count(start_marker) != 1:
        raise RuntimeError(
            "Could not uniquely locate the existing Table B teacher-placement "
            "block. No replacement was made."
        )

    start = source.index(start_marker)
    end = source.index(end_marker, start)

    source = source[:start] + PHASE_BLOCK + "\n" + source[end:]

    compile(source, "integrated_table_a_to_b_outside_route.py", "exec")
    return source
