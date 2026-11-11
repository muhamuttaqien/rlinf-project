# Task-Specific RECAP — LIBERO-Goal Task 0

This experiment **trains a new value critic and RECAP policy for Goal Task 0**.
It is separate from the completed LIBERO-10 Task 0 RECAP checkpoint and the
cross-suite transfer evaluation of that checkpoint.

**Status:** Pipeline/configuration implemented; no Goal Task 0 data collection,
SIM-4 training, SIM-5 labeling, SIM-6 training, or specialist evaluation has
been executed in this repository change. Run on the AIST workstation.

## Experiment design

| Item | Setting |
| --- | --- |
| Benchmark | LIBERO-Goal, task index 0 |
| Initial policy | Existing `pi05_libero_sft_rlinf` SFT checkpoint |
| SFT data | 48 *successful autonomous SFT rollouts* (not expert demos) |
| Mixed training rollouts | 200 baseline rollouts, successes and failures |
| Held-out evaluation rollouts for the value model | 40 independently seeded baseline rollouts |
| SIM-4 value critic | 8,000 optimizer steps |
| SIM-5 labeling | Goal-task-specific return sidecars and advantages |
| SIM-6 RECAP fine-tuning | 3,000 optimizer steps, from the ORIGINAL SFT checkpoint |
| Comparison after training | 20 matched LIBERO-Goal Task 0 episodes per policy |

The collector is a **proof-of-concept** replacement for a task-specific
expert-demonstration dataset. Autonomous successful rollouts are *not* the same
as expert teleoperation demonstrations. Review task success balance and value
validation before making paper-level claims.

The existing 20-episode Goal transfer gallery contains MP4s, wrist MP4s and
sampled states, but **not all executed actions**. Do **not** use it as RECAP
training data and do not relabel LIBERO-10 data as Goal Task 0.

## Run the pipeline

From a clean `branch-openpi-recap-sim` checkout:

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull origin branch-openpi-recap-sim
source .venv/bin/activate
python -m unittest tests.unit_tests.test_recap_libero_goal_task0 -v
```

### Step 1 — Collect Goal-specific data

```bash
python -m toolkits.recap_libero_goal_collect
python -m toolkits.recap_libero_goal_preflight
```

Default output (separate from LIBERO-10):

```text
/data1/Muttaqien/rlinf_data/recap_libero_goal_task0/
    libero_goal_task0_sft/
    libero_goal_task0_train/
    libero_goal_task0_eval/
```

The collector runs the original SFT policy on LIBERO-Goal Task 0 and writes
**actual executed actions**, RGB images, wrist images, 8-D states, 7-D actions,
task prompts, episode-success flags, and episode-end markers into LeRobot
datasets. Separate seed ranges are used for SFT/train/eval. Output directories
are never overwritten; complete matching splits are reused.

You can collect one split at a time:

```bash
python -m toolkits.recap_libero_goal_collect --split sft
python -m toolkits.recap_libero_goal_collect --split train
python -m toolkits.recap_libero_goal_collect --split eval
```

If a split is incomplete, **inspect the partial dataset** before choosing a
new `--output-root`; do not blindly delete data or let two collectors write
to one directory.

### Step 2 — Compute returns (prerequisite to the value model)

```bash
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh \
  experiments/libero_goal_task0/recap_compute_returns

python -m toolkits.recap_libero_goal_preflight --stage returns
```

The new tag `libero_goal_task0_recap` is used for all three return sidecars.
Failed episodes carry the configured terminal failure penalty.

### Step 3 — SIM-4: Train the Goal Task 0 value critic

```bash
bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh \
  experiments/libero_goal_task0/recap_value_model_sft
```

The optimizer's training budget is **8,000 steps**. Locate the *new* Goal
checkpoint after successful training:

```bash
find logs/value_sft -path '*/global_step_8000/actor/model_state_dict/full_weights.pt'
```

The selected file must belong to the **Goal Task 0** value-model experiment.
Do not reuse the earlier LIBERO-10 Task 0 SIM-4 weights.

### Step 4 — SIM-5: Compute Goal-specific advantages

Set `GOAL_VALUE_CKPT` to the actual absolute `full_weights.pt` path:

```bash
GOAL_VALUE_CKPT=/absolute/path/to/GOAL_VALUE_MODEL/full_weights.pt

bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh \
  experiments/libero_goal_task0/recap_compute_advantages --nproc 4 \
  "advantage.value_checkpoint=$GOAL_VALUE_CKPT"

python -m toolkits.recap_libero_goal_preflight --stage advantages
```

The new label sidecar is
`libero_goal_task0_train/meta/advantages_libero_goal_task0_recap.parquet`.
Avoid using old `libero10_task0_recap` labels.

### Step 5 — SIM-6: Train the Goal-specific RECAP specialist

```bash
bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh \
  experiments/libero_goal_task0/cfg_rl_openpi
```

The Goal-specific policy initializes from the **existing original π₀.₅
LIBERO-SFT** model, not from the LIBERO-10 Task 0 RECAP specialist.

The planned budget is 3,000 optimizer steps. For a fair comparison against the
earlier shortened 3,000-step run, note that this Goal setup uses a 500-step
learning-rate warmup rather than the original 5,000-step policy warmup.
Document this distinction when interpreting performance.

### Step 6 — SIM-7: Compare the NEW Goal specialist against SFT

After training, choose the new Goal-specific `global_step_3000` checkpoint
**directory**:

```bash
GOAL_RECAP_CKPT=/absolute/path/to/GOAL_POLICY/checkpoints/global_step_3000
GOAL_EVAL=/data1/Muttaqien/rlinf_runs/cfg_rl/recap_goal_specialist_20episodes

python -m toolkits.recap_libero_compare_videos \
  --checkpoint "$GOAL_RECAP_CKPT" \
  --suite libero_goal --task-index 0 \
  --episodes 20 --seed 7 \
  --value-observations --resume \
  --output "$GOAL_EVAL"

python -m toolkits.recap_libero_value_curves \
  --output "$GOAL_EVAL" \
  --episodes 20 --seed 7 \
  --value-checkpoint "$GOAL_VALUE_CKPT"
```

**Do not overwrite**
`recap_sim7_value_comparison_20episodes_libero_goal`: that directory
contains cross-task transfer evaluation of the **LIBERO-10-trained** RECAP
policy and is useful as a separate scientific baseline.

The comparison to report is:

1. Original SFT baseline, Goal Task 0.
2. LIBERO-10-trained RECAP on Goal Task 0 (**transfer**, existing).
3. Goal-trained RECAP on Goal Task 0 (**specialist**, new).

All evaluations need the same LIBERO task, initial states, seeds, action
chunking and step budget to support a controlled comparison.

## Scientific limits and safety checks

- The SIM-4 critic is task-specific and is not the policy.
- The value-model preflight rejects corrupted/missing return sidecars.
- SIM-5 uses the newly trained Goal critic; no LIBERO-10 labels are copied.
- The learning pipeline does **not** run automatically as a background task.
- Training and evaluation are GPU jobs to be started on the workstation.
- Verify the newly collected sample quality, positive/negative outcome
  balance, and critic held-out metrics before trusting any improvement.
