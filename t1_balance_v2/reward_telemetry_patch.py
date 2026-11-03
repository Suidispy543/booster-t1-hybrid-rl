import re
import textwrap


MARKER = "# RL_REWARD_TELEMETRY_V1"


def patch_source(source):
    if MARKER in source:
        return source

    # Restore the reward weights used when training the selected PPO policy.
    weight_pattern = re.compile(
        r"(env = BalanceEnv\(\s*"
        r"episode_seconds=180\.0,\s*"
        r"reset_noise=0\.0,\s*"
        r"alpha=1\.0,\s*"
        r"stance_weight=)[^,\n]+"
        r"(,\s*drift_weight=)[^,\n]+"
        r"(,\s*rate_weight=)[^,\n]+"
        r"(,\s*joint_velocity_weight=)[^,\n]+"
        r"(,)",
        re.S,
    )

    source, weight_count = weight_pattern.subn(
        r"\g<1>0.5\g<2>0.25\g<3>0.2\g<4>0.5\g<5>",
        source,
        count=1,
    )

    if weight_count != 1:
        raise RuntimeError(
            "Could not restore the PPO training reward weights."
        )

    # Label actions coming from the PPO policy.
    ppo_pattern = re.compile(
        r"(?m)^([ \t]*)action,\s*_\s*=\s*"
        r"ppo\.predict\(obs,\s*deterministic=True\)"
    )

    source, ppo_count = ppo_pattern.subn(
        r'\1env._reward_controller = "PPO"\n'
        r'\1action, _ = ppo.predict(obs, deterministic=True)',
        source,
    )

    if ppo_count < 1:
        raise RuntimeError("Could not find PPO prediction calls.")

    # Label actions coming from the pretrained locomotion teacher.
    teacher_pattern = re.compile(
        r"(?m)^([ \t]*)action = teacher\("
    )

    source, teacher_count = teacher_pattern.subn(
        r'\1env._reward_controller = "TEACHER"\n'
        r'\1action = teacher(',
        source,
    )

    if teacher_count < 1:
        raise RuntimeError("Could not find walking-teacher action calls.")

    # Replace the viewer's environment-step wrapper with reward telemetry.
    wrapper_pattern = re.compile(
        r"(?ms)^(?P<indent>[ \t]*)original_env_step = env\.step\n"
        r".*?"
        r"^(?P=indent)env\.step = visible_env_step[ \t]*$"
    )

    match = wrapper_pattern.search(source)

    if not match:
        raise RuntimeError(
            "Could not find the original visible_env_step wrapper."
        )

    indent = match.group("indent")

    new_wrapper = r'''
        # RL_REWARD_TELEMETRY_V1
        # Log the actual reward returned by BalanceEnv.
        reward_telemetry = {
            "steps": 0,
            "by_controller": {
                "PPO": 0.0,
                "TEACHER": 0.0,
                "OTHER": 0.0,
            },
            "window_controller": None,
            "window_sum": 0.0,
            "window_steps": 0,
        }

        reward_log_every = max(1, round(0.5 / policy_dt))

        env._reward_controller = "TEACHER"

        print("\n========== LIVE REWARD MONITOR ==========")
        print(
            "Reward weights:",
            f"stance={env.stance_weight:.2f},",
            f"drift={env.drift_weight:.2f},",
            f"action_change={env.rate_weight:.2f},",
            f"joint_velocity={env.joint_velocity_weight:.2f}",
        )
        print(
            f"Logging interval: {reward_log_every} decisions "
            f"({reward_log_every * policy_dt:.2f}s)"
        )
        print("=========================================\n")

        original_env_step = env.step

        def visible_env_step(action):
            if not viewer.is_running():
                raise RuntimeError("MuJoCo viewer was closed.")

            result = original_env_step(action)

            _, reward, terminated, truncated, info = result
            reward = float(reward)

            controller = getattr(
                env, "_reward_controller", "OTHER"
            )

            if controller not in reward_telemetry["by_controller"]:
                controller = "OTHER"

            # Start a fresh short window when the controller changes.
            if reward_telemetry["window_controller"] != controller:
                reward_telemetry["window_controller"] = controller
                reward_telemetry["window_sum"] = 0.0
                reward_telemetry["window_steps"] = 0

            reward_telemetry["steps"] += 1
            reward_telemetry["window_steps"] += 1
            reward_telemetry["window_sum"] += reward
            reward_telemetry["by_controller"][controller] += reward

            if (
                reward_telemetry["window_steps"] >= reward_log_every
                or terminated
                or truncated
            ):
                window_sum = reward_telemetry["window_sum"]
                window_steps = reward_telemetry["window_steps"]
                window_mean = window_sum / max(1, window_steps)

                controller_sum = (
                    reward_telemetry["by_controller"][controller]
                )

                costs = info.get("reward_costs", {})
                cost_names = (
                    "stance",
                    "drift",
                    "action_change_extra",
                    "joint_velocity",
                    "root_angular_extra",
                )

                cost_text = " | ".join(
                    f"{name}={float(costs.get(name, 0.0)):.5f}"
                    for name in cost_names
                )

                base_reward = float(
                    info.get("original_reward", reward)
                )

                policy_action = np.asarray(action, dtype=float)
                applied_action = np.asarray(
                    env.previous_action, dtype=float
                )

                policy_rms = float(
                    np.sqrt(np.mean(np.square(policy_action)))
                )
                applied_rms = float(
                    np.sqrt(np.mean(np.square(applied_action)))
                )

                print(
                    f"[REWARD][{controller}] "
                    f"t={data.time:.2f}s | "
                    f"step={reward_telemetry['steps']} | "
                    f"r_t={reward:+.4f} | "
                    f"window_sum={window_sum:+.4f} | "
                    f"window_mean={window_mean:+.4f} | "
                    f"controller_reward_sum={controller_sum:+.3f}",
                    flush=True,
                )

                print(
                    f"  base_reward={base_reward:+.4f} | "
                    f"{cost_text}",
                    flush=True,
                )

                print(
                    f"  upright={float(info.get('upright_score', np.nan)):.4f} | "
                    f"height={float(info.get('root_height', data.qpos[2])):.3f}m | "
                    f"foot_width={float(info.get('foot_width_m', np.nan)):.3f}m | "
                    f"drift={float(info.get('root_displacement_m', np.nan)):.3f}m",
                    flush=True,
                )

                print(
                    f"  alpha={env.alpha:.2f} | "
                    f"policy_action_RMS={policy_rms:.4f} | "
                    f"applied_action_RMS={applied_rms:.4f}",
                    flush=True,
                )

                reward_telemetry["window_sum"] = 0.0
                reward_telemetry["window_steps"] = 0

                if terminated:
                    print("[REWARD] Environment termination: FALL")
                elif truncated:
                    print("[REWARD] Environment time limit reached")

            viewer.sync()
            time.sleep(policy_dt)
            return result

        env.step = visible_env_step
'''

    replacement = textwrap.indent(
        textwrap.dedent(new_wrapper).strip("\n"),
        indent,
    )

    source = (
        source[:match.start()]
        + replacement
        + source[match.end():]
    )

    compile(source, "integrated_table_a_to_b_outside_route.py", "exec")

    return source
