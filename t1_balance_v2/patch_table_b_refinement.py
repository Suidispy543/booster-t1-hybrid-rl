from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENERATOR = ROOT / "make_outside_route.py"
OUTPUT = ROOT / "integrated_table_a_to_b_outside_route.py"

FORWARD_REACH_CROUCH = """placement_crouch = np.array([
            -0.04, 0.0, 0.0, 0.12, -0.04, 0.0,
            -0.04, 0.0, 0.0, 0.12, -0.04, 0.0,
        ], dtype=float)"""

OLD_PLACE_TARGET = """PLACE_TARGET = np.array([
            data.mocap_pos[mocap_id][0],
            data.mocap_pos[mocap_id][1],
            0.635
        ], dtype=float)"""

NEW_PLACE_TARGET = (
    "PLACE_TARGET = np.array([1.56, -0.43, 0.635], dtype=float)"
)

OLD_AUTO_SHIFT = """        # Use the reachable horizontal position, provided it is on Table B.
        PLACE_TARGET[:2] = predicted_object[:2].copy()
        predicted_error = predicted_object - PLACE_TARGET"""

NEW_AUTO_SHIFT = """        # Preserve the requested interior Table B target.
        # Do not silently move the target back to the table edge.
        predicted_error = predicted_object - PLACE_TARGET"""


def replace_once(text, old, new, label):
    count = text.count(old)

    if count == 1:
        return text.replace(old, new, 1)

    if count == 0 and new in text:
        return text

    raise RuntimeError(
        f"{label}: expected one match, found {count}. "
        "No ambiguous edit was made."
    )


def refine_generated_source(source):
    source = replace_once(
        source,
        "np.array([0.06, -0.06, 0.04])",
        "np.array([0.06, -0.06, 0.08])",
        "Higher carry pose",
    )

    source = replace_once(
        source,
        "placement_crouch = CROUCH.copy()",
        FORWARD_REACH_CROUCH,
        "Forward-biased placement posture",
    )

    source = replace_once(
        source,
        OLD_PLACE_TARGET,
        NEW_PLACE_TARGET,
        "Interior Table B placement target",
    )

    source = replace_once(
        source,
        OLD_AUTO_SHIFT,
        NEW_AUTO_SHIFT,
        "Preserve requested placement coordinates",
    )

    compile(source, str(OUTPUT), "exec")
    return source


# Insert the same refinements into the generator so future
# generations retain these changes.
GENERATOR_INSERTION = r'''
# FINAL TABLE B REFINEMENTS

code = replace_once(
    code,
    "np.array([0.06, -0.06, 0.04])",
    "np.array([0.06, -0.06, 0.08])",
    "Higher carry pose",
)

code = replace_once(
    code,
    "placement_crouch = CROUCH.copy()",
    """placement_crouch = np.array([
            -0.04, 0.0, 0.0, 0.12, -0.04, 0.0,
            -0.04, 0.0, 0.0, 0.12, -0.04, 0.0,
        ], dtype=float)""",
    "Forward-biased placement posture",
)

code = replace_once(
    code,
    """PLACE_TARGET = np.array([
            data.mocap_pos[mocap_id][0],
            data.mocap_pos[mocap_id][1],
            0.635
        ], dtype=float)""",
    "PLACE_TARGET = np.array([1.56, -0.43, 0.635], dtype=float)",
    "Interior Table B placement target",
)

code = replace_once(
    code,
    """        # Use the reachable horizontal position, provided it is on Table B.
        PLACE_TARGET[:2] = predicted_object[:2].copy()
        predicted_error = predicted_object - PLACE_TARGET""",
    """        # Preserve the requested interior Table B target.
        # Do not silently move the target back to the table edge.
        predicted_error = predicted_object - PLACE_TARGET""",
    "Preserve requested placement coordinates",
)

# END FINAL TABLE B REFINEMENTS

'''


def backup(path):
    backup_path = path.with_suffix(
        path.suffix + ".pre_table_b_refinement.bak"
    )

    if not backup_path.exists():
        backup_path.write_text(path.read_text())


def patch_generator(source):
    marker = 'compile(code, str(OUTPUT), "exec")'

    if "# FINAL TABLE B REFINEMENTS" in source:
        return source

    if source.count(marker) != 1:
        raise RuntimeError(
            "Could not uniquely locate the generator's final compile line."
        )

    patched = source.replace(
        marker,
        GENERATOR_INSERTION + marker,
        1,
    )

    compile(patched, str(GENERATOR), "exec")
    return patched


def main():
    for path in (GENERATOR, OUTPUT):
        if not path.is_file():
            raise FileNotFoundError(
                f"Could not find {path.name}. "
                "Run this patcher from the t1_balance_v2 directory."
            )

    generator_text = GENERATOR.read_text()
    output_text = OUTPUT.read_text()

    patched_generator = patch_generator(generator_text)
    patched_output = refine_generated_source(output_text)

    # Back up both files before overwriting either one.
    backup(GENERATOR)
    backup(OUTPUT)

    GENERATOR.write_text(patched_generator)
    OUTPUT.write_text(patched_output)

    print("Table B refinements applied.")
    print("Generator:", GENERATOR.name)
    print("Simulation:", OUTPUT.name)
    print("Backups created with .pre_table_b_refinement.bak")
    print("Python syntax check: PASSED")


if __name__ == "__main__":
    main()
