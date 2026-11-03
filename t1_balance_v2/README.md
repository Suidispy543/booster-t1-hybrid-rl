# T1 balance v2 — trained candidate with original MuJoCo physics

This is a separate, trained candidate for your existing T1 project. It greatly
reduces the stance checkpoint's shaking and drift in the tests below. The feet
still widen during settling, from 21.25 cm to approximately 27.9 cm. Continued
widening becomes negligible near the end of the nominal 180-second test.

NoSlip is **disabled**. The supplied scene, contact parameters, friction,
actuator gains, torque limits, joint limits, timestep and frame skip are unchanged.
There is no foot anchoring or state correction during playback.

## Start here in VS Code

Extract the ZIP and place `t1_balance_v2` directly inside
`~/Downloads/t1_mujoco_ppo_starter`, beside `scripts`, `assets` and `models`.
Keep your existing files. The package includes its own base-environment snapshot
and selected policy, and loads your existing Booster T1 assets.

In VS Code's integrated terminal:

```bash
cd ~/Downloads/t1_mujoco_ppo_starter
source .venv/bin/activate
mjpython t1_balance_v2/play.py --seconds 180
```

Use this new viewer: the new checkpoint expects **72 observations**, so it is
not compatible with your original 57-observation `scripts/play_ppo.py`.
The viewer loads the included selected model and its matching action-filter
configuration automatically. It closes after one episode; add `--loop` to reset
and repeat. Close the viewer window at any time to stop.

Then verify on your Mac:

```bash
python t1_balance_v2/evaluate.py --seconds 180 --suite
```

This runs a nominal 180-second episode plus nine small-disturbance trials of
60 seconds each. It writes `summary.json` and trajectory CSVs to a new dated
folder under `t1_balance_v2/evaluations`. Omit `--suite` for the nominal run only.
No retraining is needed to view or evaluate the included model.

## Measured results

Fresh matched 60-second evaluations, deterministic CPU predictions:

| Metric | Original quiet policy | Original stance policy | Selected v2 policy |
|---|---:|---:|---:|
| Survival | 60 s | 60 s | 60 s |
| Final foot-body separation | 0.6686 m | 0.2440 m | 0.2776 m |
| Maximum root displacement | 0.0724 m | 0.4672 m | 0.0407 m |
| Leg-joint velocity RMS | 0.0849 rad/s | 0.9215 rad/s | 0.0372 rad/s |
| Target-change RMS | 0.0189 rad | 0.2249 rad | 0.00505 rad |

The selected v2 policy has about 96% lower leg-joint velocity RMS than the
original stance checkpoint in these matched nominal trials. This comparison
includes initial settling. It is a comparison of complete controller setups,
not a claim that any single training change explains the improvement.

The selected policy's nominal **180-second** evaluation:

- Stayed upright for all 9,000 policy decisions.
- Maximum root displacement: **0.04069 m**.
- Final horizontal foot-body separation: **0.27891 m**.
- Final-ten-second foot-width slope: **8.19e-8 m/s**.
- After-five-second leg-joint velocity RMS: **0.000481 rad/s**.
- Minimum upright score: **0.99011**.

All **nine** small-disturbance trials stayed upright for 60 seconds. Across
those trials, maximum root displacement stayed below 0.046 m and final foot
separation stayed between 0.2772 and 0.2776 m. The four velocity-increment trials
include recovery transients in their after-five-second motion statistics.

These tests cover five initial leg-velocity perturbations, uniform in
[-0.02, 0.02] rad/s with seeds 100–104, and four root-velocity increments:
plus/minus 0.05 m/s along X/Y at five seconds. They are state perturbations,
not calibrated physical pushes. They do not establish resistance to large
pushes, uneven terrain, model mismatch, all initial states, or real hardware.
Only one training seed was used.

## What changed

1. Training episodes increased from 4 seconds to 30 seconds in phase 1 and
   60 seconds in phase 2.
2. The observation includes the original 57 values, root XY displacement from
   reset (two values, scaled by 10), previous applied actions (12), and deviation
   from home foot separation (one, scaled by 10).
3. Rewards penalize stance error, root drift, action changes, actual leg-joint
   velocity and root angular velocity together. Phase 2 increases the stance
   and joint-motion weights.
4. The selected branch uses
   `applied = 0.5 * raw + 0.5 * previous_applied` during **both training and
   evaluation**. The previous applied action is in the observation. Filtering
   was not simply added to the old viewer and presented as a learned fix.
5. Small initial leg-velocity variation, +/-0.005 rad/s, is present during
   training. Nominal evaluation resets exactly to home.
6. At the start of phase 2, the policy's left hip-roll mean bias was reduced by
   0.4 normalized action units and the right increased by 0.4. This was an
   explicit initialization adjustment, chosen after a small sensitivity study,
   followed by PPO training. It changes the policy's command preference; it
   does not change actuator gain, available torque, or joint limits. The offset
   is already part of the saved weights; playback adds no further offset.

The phase-2 reward starts with the original reward, then subtracts:

```text
2.00 * min(((foot_width - home_width) / 0.10)^2, 9)
0.25 * min((root_displacement / 0.10)^2, 9)
(0.20 - 0.005) * mean((applied - previous_applied)^2)
1.00 * mean(leg_joint_velocity^2)
(0.30 - 0.05) * sum(root_angular_velocity^2)
```

The subtracted original coefficients prevent double-counting the existing
action-change and angular-motion penalties. Original termination thresholds
are retained: root height below 0.38 m or upright score below 0.40.

## Training and selection history

The initial source was your stance checkpoint:
`experiments/stance_balance_20261027_224155_950849/stance/checkpoint_000075264.zip`.

Phase 1 transferred its policy/value weights into the 72-observation network,
with zero initial weights for the added input features. It used a new optimizer,
initial Gaussian action standard deviation 0.35, learning rate 1e-4, entropy
coefficient zero, and 30-second episodes. The source checkpoint's deterministic
actions were checked against the transferred network before training.

Phase 2 resumed the phase-1 checkpoint at 100,352 v2 steps, retained its optimizer
and learned exploration standard deviations, applied the explicit hip-bias
initialization, and trained with 60-second episodes and learning rate 5e-5.
The selected checkpoint is at another 100,352 steps: **200,704 additional v2
training steps in its lineage**. Other screening branches consumed additional
training and are retained in `runs/`; this count is not the total experiment
budget or the original policy's lifetime training count.

The selected file is an unchanged copy of:
`runs/stance_alpha0.5_20261028_121845_504877/checkpoint_000100352.zip`.

Model SHA256:
`ecb6e4ba7f3c610c8c49a84da851ca33cc49cc72cfc94803edfceb941dc6b51c`

Candidate selection considered survival, width, drift and motion together;
the latest checkpoint was not automatically selected. The weaker direct-action
and unadjusted continuation branches remain in the reports for comparison.

## Reproduce the selected training sequence (optional)

The original source policies and assets must remain in your project. First:

```bash
python t1_balance_v2/experiment.py --check-only
python t1_balance_v2/experiment.py --source stance --alpha 0.5 --steps 100000
```

Then use the newly printed run directory for `FIRST_RUN` below:

```bash
python t1_balance_v2/experiment.py \
  --source stance \
  --resume FIRST_RUN/checkpoint_000100352.zip \
  --alpha 0.5 --steps 100000 --episode-seconds 60 \
  --stance-weight 2 --velocity-weight 1 \
  --learning-rate 0.00005 --hip-bias-inward 0.4
```

PPO rounds training to complete rollouts. Each invocation creates a new
timestamped run folder. It never overwrites your original model or promotes a
new checkpoint automatically. For another v2 checkpoint, pass its path through
`--model` to `play.py` or `evaluate.py`; keep its neighboring `summary.json`.

## Evidence and limits

- `selected/config.json`: matching environment settings and provenance.
- `evaluations/20261028_122200_281065/`: selected model's full nominal/perturbation
  results and CSV trajectories.
- `validation/comparison.png`: original-policy and selected-policy comparison.
- `validation/selected_preview.mp4`: first eight and final five seconds of a
  180-second run, at real-time playback speed, with the time jump labeled.
- `validation/physics_rate_check.json`: a separate ten-second check sampling
  actual joint motion at each 500 Hz physics step.
- `runs/`: checkpoint screening results, policy files, training logs and source
  snapshots. `research_hip_bias/` records the initialization sensitivity study.

The main trajectory measurements are sampled at 50 Hz, use foot-body origins
rather than sole-edge clearance, and exclude the first home-to-action transition
from target-change RMS. The separate 500 Hz check addresses motion between policy
decisions over its ten-second diagnostic. The main minimum-height metrics start
after the first simulation step, not at the exact reset instant.

Tests used Linux, MuJoCo 3.17.0, Stable-Baselines3 2.9.2, Gymnasium 1.4.0,
PyTorch 2.14.1+cpu, NumPy 2.3.5 and one PyTorch thread. Your Mac can produce
different trajectories. Confirm using the supplied evaluator; the macOS passive
viewer itself has not been run in this Linux environment.

For background on separate evaluation, seed variability and reward design:
https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
