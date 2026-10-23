# StackCube π₀.₅ Clean V1 — Experiment B (1,800 steps; 32D SFT loss)

## Purpose

Test whether Clean V1's 600-step budget underfits a successful 100-demo
ManiSkill StackCube dataset. Experiment B restarts training independently
from the **original pretrained pi05_base**, for 1,800 optimizer steps;
it does **not** resume the baseline's step-600 checkpoint.

The original baseline remains `maniskill_stackcube_pi05_clean_v1`.
The new experiment is `maniskill_stackcube_pi05_clean_v1_1800_32d`.
Use the branch `experiment/stackcube-pi05-restart-v1`.

## Exact changes relative to baseline A

Config file:
`examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_clean_v1_1800_32d.yaml`

The only non-comment YAML differences from A are:

| Field | Baseline A | Experiment B |
| --- | --- | --- |
| `runner.logger.experiment_name` | `maniskill_stackcube_pi05_clean_v1` | `maniskill_stackcube_pi05_clean_v1_1800_32d` |
| `runner.max_steps` | `600` | `1800` |
| `runner.save_interval` | `100` | `600` |

**Unchanged:** same 100-demo `local/stackcube_pi05_clean_v1` dataset;
same fresh Clean V1 norm stats, 8D state, two 224×224 RGB views,
7D `pd_ee_delta_pose` commands, 10-step action horizon/chunk, 32D padded
OpenPI flow-matching optimization objective, train-only image augmentation
disabled, global batch 16, microbatch 1, seed 42, base π₀.₅ architecture,
optimizer family and hyperparameters. The cosine learning-rate *schedule
duration* changes when `max_steps` changes, as
`actor.optim.total_training_steps` interpolates from `runner.max_steps`.
This is a comparison of **training budgets and the corresponding planned
schedules**, not a demonstration that only the number of optimizer
iterations differs.

## Run — one training job, no repeated smoke tests

From the repository root, in the same Python environment used for the
600-step baseline, set the **original pretrained** model location and
invoke the non-overwriting launcher:

~~~bash
git pull --ff-only origin experiment/stackcube-pi05-restart-v1
export PI05_MODEL_PATH=/absolute/path/to/original/pi05_base
bash scripts/run_stackcube_pi05_clean_exp_b.sh
~~~

If `PI05_MODEL_PATH` is already correctly exported, omit the second line;
do **not** use the step-600 SFT checkpoint here.

The launcher pins known workstation paths for Clean V1's existing
dataset, normalization, results, Ray temporary directory, and ROS-independent
GPU job. It verifies that the base contains `model.safetensors` without
competing `full_weights.pt`, that the 100-demo dataset and norm stats exist,
and that the new results directory does **not** exist. It then executes
the existing `examples/sft/train_vla_sft.py` Hydra entrypoint with the
new config. It makes no model/loader/objective changes and does **not**
pass `runner.resume_dir` or delete any existing outputs.

Experiment B outputs go to:

~~~text
/data1/Muttaqien/rlinf_data/results/maniskill_stackcube_pi05_clean_v1_1800_32d
~~~

Expected checkpoints under this directory:
- `checkpoints/global_step_600/actor`
- `checkpoints/global_step_1200/actor`
- `checkpoints/global_step_1800/actor`

The existing `maniskill_stackcube_pi05_clean_v1` baseline remains untouched.

## First evaluation: unchanged 10 held-out seeds

Only after `global_step_1800` has been saved:

~~~bash
PYTHONPATH="$PWD" python scripts/evaluate_stackcube_pi05_ee_delta.py \
  --config examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_clean_v1_1800_32d.yaml \
  --checkpoint "$NEW_ROOT/checkpoints/global_step_1800/actor" \
  --norm-stats "$STACKCUBE_NORM_STATS" \
  --episodes 10 --seed-start 10000 --action-chunk 10 \
  --policy-noise-seed 1234 \
  --output "$NEW_ROOT/evaluation/step_1800"
~~~

Compare against A's held-out seeds 10000–10009: **A = 0/10 task success;
8/10 ever-grasped**. Keep the evaluation settings, noise-seed argument,
camera input, and success criteria identical. As the current evaluator's
`ever_grasped` means a grasp happened at least once, it does **not**
measure held grasp, successful lift, or correct placement.

Primary metric: ManiSkill task success (number/10). Secondary: ever-grasp
and action-phase behavior. Training flow loss, even 7D flow loss, is not
a direct action-MAE or success metric.

For confidence after a positive 10-seed pilot, evaluate more unseen
seeds and (if needed) independently evaluate `global_step_1200`.
Avoid running all checkpoint/seed combinations without a decision reason.

## Decision rule

- **If B gains task successes or clearly progresses through lifting and
  placement:** investigate whether 1,800 steps suffice; avoid premature
  changes to controller/dataset.
- **If B remains 0/10 and fails similarly during lift:** compare a
  controlled 7D-only loss experiment using the same data, pretrained
  initialization and 1,800-step budget. That is a training-objective
  ablation, *not* an implementation bug fix.
- **If neither improves:** assess data diversity, representation/history,
  and then consider fresh 200-demo collection, preserving the first 100
  and using non-overlapping train/eval seeds.

**Execution status:** Configuration is committed on GitHub. Training and
closed-loop evaluation still must run on the user's workstation. Do not
claim success or checkpoints exist until the corresponding logs are
actually produced.
