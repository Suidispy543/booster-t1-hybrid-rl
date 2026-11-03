import numpy as np

import train_walk_v4 as v4
import train_walk_v5 as v5


def evaluate_case(label, module, native_home_ctrl=None):
    EnvClass = getattr(module, "WalkEnvV4", None)
    if EnvClass is None:
        EnvClass = getattr(module, "WalkEnvV5")
    env = EnvClass(
        episode_seconds=20.0,
        reset_noise=0.0,
        target_speed=0.10,
        gait_frequency=0.8,
    )

    try:
        # Disable the reference gait to isolate the standing controller.
        env._reference_gait = lambda phase: np.zeros(
            12, dtype=np.float32
        )

        # For the teacher-gain case, restore the original home targets
        # so that only the motor gains differ.
        if native_home_ctrl is not None:
            env.home_ctrl[:] = native_home_ctrl

        model = module.initialize_policy(
            env, learning_rate=3e-5, exploration_std=0.25
        )

        obs, _ = env.reset(seed=42)
        start_xy = env.data.qpos[:2].copy()
        max_tilt = 0.0
        duration = 0.0
        fell = False

        dt = env.model.opt.timestep * env.frame_skip
        max_steps = int(20.0 / dt)

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            tilt = float(info.get("root_tilt_deg", 0.0))
            max_tilt = max(max_tilt, tilt)
            duration += dt

            if terminated:
                fell = True
                break
            if truncated:
                break

        displacement = env.data.qpos[:2] - start_xy

        print(f"\n===== {label} =====")
        print(f"Duration: {duration:.2f} s")
        print(f"Fell: {fell}")
        print(f"Forward displacement: {displacement[0]:+.3f} m")
        print(f"Lateral displacement: {displacement[1]:+.3f} m")
        print(f"Maximum torso tilt: {max_tilt:.2f} degrees")

        return env.home_ctrl.copy()

    finally:
        env.close()


if __name__ == "__main__":
    native_targets = evaluate_case(
        "Original XML gains, original targets, no gait",
        v4,
    )

    evaluate_case(
        "Teacher gains, original targets, no gait",
        v5,
        native_home_ctrl=native_targets,
    )
