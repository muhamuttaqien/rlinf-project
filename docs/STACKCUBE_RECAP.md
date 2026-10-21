# StackCube-v1: Panda demonstrations for RECAP

This experiment starts with ManiSkill's built-in StackCube-v1 motion-planning
expert, *not* a trained OpenPI policy. The first milestone is a valid LeRobot
SFT dataset that RLinf's existing RECAP return computation can read.

## Controller and observations

- Environment: `StackCube-v1`, native Franka Panda, `physx_cpu`.
- Expert controller: `pd_joint_pos`, 8 actions (7 arm targets + gripper).
- State: 9 Panda joint positions (7 arm + 2 finger).
- Image: 224×224 `base_camera` RGB (or `3rd_view_camera` if present).
- `wrist_image`: actual hand-camera RGB if available; otherwise duplicates the
  base-camera image **as a placeholder, not an independent wrist observation**.
- Instruction: `stack the red cube on top of the green cube and release it`.
- Success: native `info["success"]` returned by ManiSkill's planner.
- Only successful episodes are accepted into the SFT dataset.

This collector purposely does **not** change the controller to
`pd_ee_delta_pose`: the values and dimensions are not interchangeable.
Future policy inference must use exactly the same action semantics.
It also does not yet provide unsuccessful policy rollouts: those need their own
collector, with actual episode-level `is_success` labels.

## Run on the RLinf host

From the repository root, using an environment with ManiSkill and LeRobot:

```bash
git fetch origin
git switch branch-openpi-recap-chat
python tests/e2e_tests/embodied/check_stackcube_panda.py --sim-backend cpu
PYTHONPATH="$PWD" python toolkits/lerobot/collect_maniskill_stackcube_recap.py \
  --repo-id local/stackcube_sft --num-episodes 10 --max-attempts 100
```

If a previous LeRobot `create()` call failed, its output directory may already
exist. To retry without deleting a possibly useful dataset, select a fresh
repo ID such as `local/stackcube_sft_v2` and use that same dataset when
computing RECAP returns. A LeRobot scalar `is_success` label is declared with
`shape: (1,)` and written as a one-element NumPy Boolean array.

The solver runs in `pd_joint_pos` and the existing environment smoke test
runs in `pd_ee_delta_pose`. The smoke test verifies simulator availability,
not action compatibility with this dataset.

The LeRobot dataset is saved under the configured HF LeRobot home directory
(or the absolute `--repo-id` path). Do not check generated episodes or
model checkpoints into Git.

## Validate the working 10-episode dataset

The initial collector succeeded with 10 episodes using `local/stackcube_sft_v2`.
Run the pilot validator before proceeding:

```bash
PYTHONPATH="$PWD" python toolkits/lerobot/validate_maniskill_stackcube_recap.py \
  --dataset-path "$HOME/.cache/huggingface/lerobot/local/stackcube_sft_v2" \
  --expected-episodes 10 --decode-frames 3
```

The fast Parquet audit checks episode/frame indices, metadata counts, the
9-dimensional finite state, 8-dimensional finite actions, task IDs, and
per-episode success labels. The optional `--decode-frames` also requests
sampled RGB observations using LeRobot's loader. It does not physically
replay the trajectory or verify the stacking visually; that requires
reviewing recorded images or videos.

The installed LeRobot v3 build can put the task text in the Parquet
column `__index_level_0__` rather than `task`. RLinf now reads either
layout, including in the RECAP return-computation stage. No dataset
regeneration is necessary for this metadata-format difference.

## Collect and audit 100 pilot episodes

Keep pilot seeds separate from the initial smoke-test seeds (0–9); reserve
a different, held-out seed range for final evaluation.

```bash
PYTHONPATH="$PWD" python toolkits/lerobot/collect_maniskill_stackcube_recap.py \
  --repo-id local/stackcube_sft_pilot100 \
  --num-episodes 100 --max-attempts 250 \
  --seed 1000 --sim-backend physx_cpu

PYTHONPATH="$PWD" python toolkits/lerobot/validate_maniskill_stackcube_recap.py \
  --dataset-path "$HOME/.cache/huggingface/lerobot/local/stackcube_sft_pilot100" \
  --expected-episodes 100 --decode-frames 3
```

This creates a new dataset and preserves the 10-episode smoke-test data.
The pilot alone does not establish OpenPI learning performance. The collector
only saves successful expert demonstrations and does not currently report
all failed expert attempts as separate episodes.

## Run RECAP Stage 1 after inspecting saved episodes

Find the dataset's absolute directory first, then run:

```bash
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh \
  recap_compute_returns \
  'data.train_data_paths=[{dataset_path:/ABS/PATH/TO/stackcube_sft,type:sft}]' \
  data.gamma=1.0 data.tag=stackcube_sft data.num_workers=2
```

For the 100-episode pilot, replace the placeholder with the absolute path to
`stackcube_sft_pilot100` and set `data.tag=stackcube_pilot100`.

Stage 1 should produce `meta/returns_stackcube_sft.parquet`. Confirm the
number of episodes and successful end states before using it for training.
No performance or end-to-end training claim is implied by generating returns.

## Next milestones

1. Train/evaluate an OpenPI π₀.₅ baseline using a matching `pd_joint_pos`
   input/action adapter and action normalizations.
2. Collect **real** baseline rollouts, preserving failures and native success.
3. Validate RECAP's value-model dataset mapping and train the value model.
4. Compute advantages and use RLinf's native CFG policy training.
5. Evaluate baseline versus CFG policy on held-out StackCube seeds.

The official RECAP configs are under `examples/offline_rl/config/`. Their
default `libero` robot mappings and 7-action policy outputs are **not**
correct for this 8-action Panda joint-position dataset. They require
integration and tests before launching the large-model stages.


Stage 1 now handles LeRobot task descriptions from either `meta/tasks.jsonl`
or `meta/tasks.parquet`. For future failed-policy rollout datasets, it also
unpacks shape-(1,) Boolean success labels correctly, so `[False]` is a failed
episode rather than being interpreted as a truthy list.
