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

## Pi0.5 Panda joint baseline SFT configuration

The 100-episode pilot was collected and structurally validated on
2026-10-22: 100/100 recorded successes, 10,709 frames, mean 107.09 frames per
episode. RLinf's RECAP Stage 1 processed the copy
`local/stackcube_sft_pilot100_recap`, producing
`meta/returns_stackcube_pilot100.parquet` (10,709 rows, min -177, max 0).
This is *return preparation*, not model training or RECAP policy improvement.

Use the original **`local/stackcube_sft_pilot100`** for baseline SFT.
Do not point SFT at the copy whose manifest has been modified by Stage 1.

A dedicated OpenPI model config
`pi05_stackcube_panda_joint` is registered in
`rlinf/models/embodiment/openpi/dataconfig/__init__.py`. The recipe is
`examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05.yaml`.
The configuration explicitly uses:

- Panda `pd_joint_pos` 8D **absolute** target actions (no delta conversion).
- 9D joint `qpos` state, continuous input (`discrete_state_input=false`).
- A single real `base_camera` observation (`num_images_in_input=1`).
- The duplicated `wrist_image` storage field is **masked**, not a
  fictitious second camera.
- Network horizon 10; OpenPI action dimension 32 (8 unpadded environment
  actions); no RLT, no value head.
- Native OpenPI SFT loader and existing FSDP training infrastructure.

### Existing Python environment (single `.venv`)

Use the project's existing `.venv`. The verified workstation environment
has `torch==2.7.1+cu128`, `torchvision==0.22.1+cu128`,
`torchcodec==0.5+cu128`, `transformers==4.53.2`, and the RLinf OpenPI
fork `rlinf-openpi==0.1.1`, with OpenPI's Transformers patches applied.
`uv pip check` reported no dependency incompatibilities.

```bash
source .venv/bin/activate
python -c "import torch, openpi.training.data_loader, mani_skill, lerobot; print(torch.__version__, torch.cuda.is_available())"
uv pip check --python .venv/bin/python
```

Do **not** rerun the repository's broad installer or `uv sync` without first
reconciling the `pyproject.toml` PyTorch override (`torch==2.11.0`),
because that can restore incompatibilities with `rlinf-openpi==0.1.1`
and the installed LeRobot version. A second virtual environment is not
required.

Native NVIDIA hosts usually provide Vulkan's ICD at
`/usr/share/vulkan/icd.d/nvidia_icd.json`; the installer supports that
path and retains a fallback for Docker. Check with `vulkaninfo --summary`
only if Vulkan rendering is needed.

### Inspect configuration tests and compute normalization statistics

Run:

```bash
PYTHONPATH="$PWD" python -m pytest \
  tests/unit_tests/test_stackcube_openpi_sft_config.py -v

PYTHONPATH="$PWD" python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_stackcube_panda_joint \
  --repo-id local/stackcube_sft_pilot100
```

The normalization script now detects v3 `meta/tasks.parquet` and streams
the raw `state` and `actions` columns from the v3 Parquet shards, avoiding
the installed OpenPI loader's v2-only `meta/tasks.jsonl` expectation.
It validates dimensions, finite values and total frame count, then writes
OpenPI-compatible `norm_stats.json` in the configured assets directory.
For the StackCube pilot the expected vectors are state **9D**, action **8D**,
and 10,709 frames. No video decoding or Hugging Face Hub access is needed.

The OpenPI SFT path now has a **read-only, StackCube-specific LeRobot v3
adapter** in `rlinf/data/datasets/openpi/v3_stackcube_dataset.py`.
StackCube was collected with `dtype: image`: camera images are embedded
in Parquet, not MP4. In this environment the installed LeRobot reader is
**v2.1-only**, requiring `meta/tasks.jsonl`. Therefore the adapter reads
v3 numeric rows, episode boundaries and embedded base-camera bytes directly
from Parquet. Images are decoded lazily (one cached Parquet row group), and
10-step action windows clamp to the current episode's final frame.
The duplicate wrist view is replaced by zeros and remains masked. The
official OpenPI SFT transforms, quantile normalization and padding remain
unchanged. No dataset conversion, LeRobot upgrade, Hub lookup or mutation
of the original 100 episodes is needed. Other dataset loaders are unchanged.

Before GPU training, **validate the adapter on the workstation**:

```bash
git pull --ff-only origin branch-openpi-recap-chat

PYTHONPATH="$PWD" python -m pytest \
  tests/unit_tests/test_stackcube_openpi_sft_config.py \
  tests/unit_tests/test_stackcube_v3_openpi_loader.py -v

export STACKCUBE_NORM_STATS="$PWD/assets/pi05_stackcube_panda_joint/local/stackcube_sft_pilot100/norm_stats.json"
PYTHONPATH="$PWD" python scripts/check_stackcube_openpi_sft_dataloader.py
```

The smoke script now detects the actual local LeRobot task-metadata
format: `meta/tasks.jsonl` (v2.x) uses the installed OpenPI/LeRobot
loader; `meta/tasks.parquet` (v3.x) uses the read-only Parquet adapter.
It checks three camera frames and one normalized batch, without model
weights or training. If neither task file exists, it prints a clear
incomplete-dataset error instead of triggering an accidental Hub lookup.
Both formats retain the original 9D Panda state and 8D action semantics.
The synthetic v3 tests also check an episode spanning Parquet shards
and pickle-safe worker initialization.
Do **not** change the original `stackcube_sft_pilot100` data or spoof
`codebase_version`.

The verified pilot run wrote its statistics to
`assets/pi05_stackcube_panda_joint/local/stackcube_sft_pilot100/norm_stats.json`
relative to the repository root (not `checkpoints/torch/`).
Validate the output using OpenPI's loader:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from openpi.shared import normalize

root = Path("assets/pi05_stackcube_panda_joint/local/stackcube_sft_pilot100")
stats = normalize.load(root)
for key, dimension in (("state", 9), ("actions", 8)):
    item = stats[key]
    assert item.mean.shape == (dimension,)
    assert item.std.shape == (dimension,)
    assert np.isfinite(item.mean).all()
    assert np.isfinite(item.std).all()
    print(key, "mean", item.mean, "std", item.std)
print("StackCube normalization passed")
PY

export STACKCUBE_NORM_STATS="$PWD/assets/pi05_stackcube_panda_joint/local/stackcube_sft_pilot100/norm_stats.json"
test -f "$STACKCUBE_NORM_STATS"
```

Inspect the statistics for reasonable action/state ranges before starting
training. If normalization fails, collect the traceback before GPU training.

### Validate pretrained Pi0.5 checkpoint architecture

A candidate checkpoint was located at
`checkpoints/torch/pi05_base_rlinf/model.safetensors`. Its header contains
`llm.*` and `img.*` keys in the newer RLinf format; that is not enough to
prove it matches the model's tensor names/shapes.

The lightweight preflight constructs the StackCube 32D Pi0.5 architecture
on the PyTorch **meta** device and compares expected tensor names/shapes
against the safetensors header, without allocating checkpoint weights or
CUDA memory:

```bash
PYTHONPATH="$PWD" python scripts/check_stackcube_pi05_checkpoint.py \
  --checkpoint checkpoints/torch/pi05_base_rlinf/model.safetensors
```

A pass establishes architectural tensor-shape compatibility, **not** actual
weight values, SFT numerical stability, or GPU training readiness. The
on-workstation result is still pending.

### Run one SFT optimizer step on a GPU with sufficient VRAM

Full-parameter Pi0.5 using `no_shard` is not expected to be practical on
the desktop RTX 4070 Ti SUPER (16 GiB). Use the available 96 GiB GPU server,
after ensuring that this branch, the **same compatible Python environment**,
the read-only StackCube data, the normalization stats and the checkpoint are
accessible there. Do not reuse desktop absolute paths across machines.

From the server's repository root:

```bash
source .venv/bin/activate
export PI05_MODEL_PATH="$PWD/checkpoints/torch/pi05_base_rlinf"
export STACKCUBE_DATASET=local/stackcube_sft_pilot100
export STACKCUBE_NORM_STATS="$PWD/assets/pi05_stackcube_panda_joint/local/stackcube_sft_pilot100/norm_stats.json"
export EMBODIED_PATH="$PWD/examples/sft"
export CUDA_VISIBLE_DEVICES=0
# Lab workstation: isolate this run from pre-existing Ray 2.58 / Python 3.11.14.
# Current .venv has Ray 2.59 / Python 3.11.15. Do NOT stop shared Ray.
export RAY_ADDRESS=local
export RAY_TMPDIR="$HOME/.cache/ray-stackcube"
# Avoid Ray 2.59's MetricsHead dashboard startup failure in this local session.
# This flag is ignored when RAY_ADDRESS is not "local".
export RLINF_RAY_DISABLE_DASHBOARD=1
# Ray's zero-GPU node-probe actor otherwise inherits CUDA_VISIBLE_DEVICES=""
# and RLinf incorrectly reports zero accelerators despite nvidia-smi.
# Keep the driver's selected GPU visible to the probe and let RLinf's
# accelerator placement explicitly allocate the training GPU.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
# Ray 2.59 NVML enumeration may report zero devices even though PyTorch
# detects cuda:0. Reserve one GPU in this isolated local Ray instance and
# use the same explicit count in RLinf's node hardware probe.
export RLINF_RAY_LOCAL_NUM_GPUS=1
mkdir -p "$RAY_TMPDIR"
test -f "$PI05_MODEL_PATH/model.safetensors"
test -f "$STACKCUBE_NORM_STATS"
test -f "$HOME/.cache/huggingface/lerobot/$STACKCUBE_DATASET/meta/info.json"

PYTHONPATH="$PWD" python examples/sft/train_vla_sft.py \
  --config-path "$PWD/examples/sft/config" \
  --config-name maniskill_stackcube_panda_sft_openpi_pi05 \
  runner.max_steps=1 \
  runner.save_interval=-1 \
  actor.global_batch_size=1 \
  actor.micro_batch_size=1 \
  data.num_workers=0 \
  +actor.model.openpi.torch_compile=false
```

Here `runner.max_steps=1` requests **one optimizer step**, not the default
200. The override `runner.save_interval=-1` disables checkpoint writes
(including the final step), `data.num_workers=0` simplifies debugging,
and disabling `torch.compile` avoids compilation in the first smoke test.
Use the main Python entry point because the existing shell wrapper does not
forward Hydra overrides. Inspect a finite `train/loss` and successful
optimizer/learning-rate logging before increasing training steps.
A successful step is not evidence that the learned policy can stack cubes.


After validating training and action-representation consistency, increase
successful training demonstrations toward 1,000, keeping evaluation seeds
held out. Subsequent RECAP stages still require actual policy rollouts
(including failures), value-model checkpoints, advantage inference and CFG.


### Weights & Biases dashboard for StackCube SFT

The StackCube `maniskill_stackcube_panda_sft_openpi_pi05` configuration
now enables `wandb` alongside `tensorboard`. The W&B project is
`rlinf-openpi-recap`, with run name `maniskill_stackcube_panda_pi05_sft`.
RLinf's existing `MetricLogger` records `train/loss`,
`train/grad_norm`, `train/learning_rate` and timing scalars.

At run initialization, the **driver process** also reads the *local*
LeRobot StackCube training dataset, chooses up to two expert episodes,
and logs three base-camera snapshots plus one sampled video per episode.
The videos contain actual **expert demonstrations** from the dataset,
not model-generated trajectories or policy evaluation. Each video
decodes at most 32 frames, distributed across the entire episode.
MP4 encoding falls back to GIF if needed. Media uploads are attempted
once per run, not at every training step. They may be omitted with a
warning if local data or the encoder is unavailable; this does not
abort GPU training.

The lab workstation uses its own NVIDIA/NVML workaround as validated
separately. Keep the working GPU, Ray and `LD_LIBRARY_PATH` variables
in the same terminal. Verify W&B is installed and authenticated:

```bash
python -c "import wandb; print('wandb:', wandb.__version__)"
wandb login
```

Then run the usual 200-step command with W&B online (default). For
quick media verification, first use
`runner.max_steps=1 runner.save_interval=-1 actor.global_batch_size=1`.
Expected W&B panels include
`dataset/expert_episode_000/snapshots` and
`dataset/expert_episode_000/demonstration_video`, as well as training
metrics. The W&B run link appears in console output after initialization.

To turn off dataset media logging without disabling W&B metrics, pass
`runner.logger.sft_media.enabled=false`. Later, evaluate the saved
policy in ManiSkill to obtain actual **policy rollout** videos and
success rates; SFT demonstrations are not evidence of policy performance.

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


## Milestone 8 — Evaluate StackCube policy after SFT

The 200-step SFT pilot completed on the 96 GB lab GPU (final reported
`train/loss=0.0646`). This is a training metric, not a success rate.
RLinf's FSDP checkpoint writer saves inference-ready model weights at:

```text
../results/maniskill_stackcube_panda_pi05_sft/checkpoints/
  global_step_200/actor/model_state_dict/full_weights.pt
```

Use `scripts/evaluate_stackcube_pi05.py` to run real **policy-generated**
ManiSkill StackCube-v1 rollouts, without the expert planner. It constructs the
same Panda, RGB 224×224, `pd_joint_pos`, 9D qpos and 8D action environment
used to collect the SFT demonstrations. The official OpenPI transforms load
the exact SFT normalization stats and decode 10-action chunks to 8D absolute
joint targets. The local JSON file records every seed, success flag and
episode length. Video frames are captured from the simulator, not reused from
the demonstration dataset, and up to three rollout MP4s are uploaded to W&B.

From the same lab terminal where SFT succeeded, retain the working NVIDIA
NVML library override and CUDA selection. Run from the repository root:

```bash
test -s ../results/maniskill_stackcube_panda_pi05_sft/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt
PYTHONPATH="$PWD" python -m pytest tests/unit_tests/test_evaluate_stackcube_pi05.py -q
PYTHONPATH="$PWD" python scripts/evaluate_stackcube_pi05.py \
  --episodes 20 --seed-start 10000 --max-videos 3
```

This does not use Ray and will load only one checkpoint onto CUDA. The
default held-out candidate seeds are 10000–10019; confirm your demonstration
collection did not use these seeds. Results go to
`../results/maniskill_stackcube_panda_pi05_sft/evaluation/step_200/results.json`.
W&B project: `rlinf-openpi-recap`, run name:
`maniskill_stackcube_panda_pi05_step200_eval`.

The pretrained model baseline should use **the same seeds**, environment and
normalization, in a separate process/run (not simultaneously on the GPU):

```bash
PYTHONPATH="$PWD" python scripts/evaluate_stackcube_pi05.py \
  --checkpoint "$PI05_MODEL_PATH" \
  --label pretrained --episodes 20 --seed-start 10000 \
  --output ../results/maniskill_stackcube_panda_pi05_sft/evaluation/pretrained
```

Before making success-rate claims, inspect results and policy rollout videos.
If the evaluator reports an OpenPI transform error or ManiSkill mismatch,
fix the interface rather than changing action semantics or padding qpos.
The tests use fakes; they do not establish actual simulator performance.


## Milestone 8.4 — Diagnose action predictions and gripper control

The first 20 held-out rollouts for the 200-step SFT policy reported 0/20
StackCube successes (every episode reached 300 simulator steps). A low SFT
loss by itself does not demonstrate closed-loop task performance. Before any
more training, use the focused diagnostic to distinguish poor expert action
imitation from joint/gripper control failures.

From the lab repository root, with the working CUDA/NVML environment retained:

```bash
PYTHONPATH="$PWD" python -m pytest \
  tests/unit_tests/test_diagnose_stackcube_pi05.py -q

PYTHONPATH="$PWD" python scripts/diagnose_stackcube_pi05.py \
  --episodes 2 --seed-start 10000 \
  --expert-episodes 2 --expert-samples 3
```

The diagnostic loads the step-200 checkpoint (no retraining), applies
the same image/state transforms and training normalization, and runs
two complementary checks:

1. **Paired expert comparison:** six saved observation/action pairs from
   two training demonstrations. Predict 8D joint-position actions from
   each exact saved RGB image + 9D state, then compare the *first* predicted
   action against the corresponding expert target (7 arm joints, 1 gripper).
   These are stochastic flow-model predictions, so error is diagnostic,
   not a substitute for closed-loop success.
2. **Actual environment control:** two 300-step ManiSkill rollouts,
   recording every commanded 8D action, the observed 7 arm-joint
   positions, and both observed finger joint positions before/after
   execution. Controller gripper commands and measured finger qpos are
   **not directly the same units**, so they are charted separately.

The outputs are saved under
`../results/maniskill_stackcube_panda_pi05_sft/evaluation/step_200/diagnostics/`:
`diagnostics.json`, `rollout_actions.csv`, and a sampled policy video.
The W&B run `maniskill_stackcube_panda_pi05_step200_diagnostics` in
`rlinf-openpi-recap` contains paired-action tables, a gripper scatter
chart, gripper command/finger-joint traces, and target-tracking traces.

Interpretation: large paired expert discrepancies suggest problems with
learned actions, checkpoint/input alignment, or insufficient SFT. Small
paired discrepancies but poor closed-loop performance suggest accumulated
execution errors or weak feedback correction. Look at the gripper command
trajectory *alongside* actual finger qpos to distinguish targets from
physical actuation. This diagnostic does not alter saved datasets,
checkpoints, Ray processes, controllers, or NVIDIA driver configuration.


## Milestone 9 — State-conditioned Pi0.5 SFT (controlled comparison)

The initial vision-only 200-step SFT experiment reported `0/20`
successes. Dense expert-observation diagnostics (40 samples) showed
arm MAE `0.17909` and gripper command absolute error `0.71383`,
with an opening-direction prediction accuracy of 10/20 and
closing-direction accuracy of 17/20. These indicate poor expert action
imitation but do not identify a single causal mechanism.

A key hypothesis is that the initial Pi0.5 config used
`discrete_state_input=False`. In OpenPI Pi0.5, proprioceptive state is
normally discretized into the prompt by `TokenizePrompt` when this
flag is True; with it disabled, there is no separate Pi0 continuous-state
projection in Pi0.5. This experiment adds state-conditioned prompt
tokens without changing the dataset, 8D absolute `pd_joint_pos`
action targets, image cameras, controller, model weights used to
initialize training, optimizer, training length, or norm stats.

### Separate files and result namespaces

- Vision-only baseline **unchanged**:
  `examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05.yaml`
  with OpenPI `pi05_stackcube_panda_joint`,
  run `maniskill_stackcube_panda_pi05_sft`.
- State-conditioned experiment **new**:
  `examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_state.yaml`
  with OpenPI `pi05_stackcube_panda_joint_state`,
  run `maniskill_stackcube_panda_pi05_state_sft`.

Both are in W&B project `rlinf-openpi-recap`. The state-conditioned
experiment writes to a separate checkpoint/result directory. Do **not**
use the previous 200-step checkpoint as the initial `PI05_MODEL_PATH`;
use the **same pretrained Pi0.5 base** for a fair comparison.

### Run on laboratory GPU 0

From the RLinf repository root after `git pull`, using the same
working NVML user-level override and Ray-local settings as the
successful 200-step run:

```bash
export PI05_MODEL_PATH="$PWD/checkpoints/torch/pi05_base_rlinf"
export STACKCUBE_DATASET="local/stackcube_sft_pilot100_lab"
export STACKCUBE_NORM_STATS="$PWD/assets/pi05_stackcube_panda_joint/local/stackcube_sft_pilot100_lab/norm_stats.json"
export EMBODIED_PATH="$PWD/examples/sft"
export CUDA_VISIBLE_DEVICES=0
export RAY_ADDRESS=local
export RAY_TMPDIR="$HOME/.cache/ray-stackcube"
export RLINF_RAY_DISABLE_DASHBOARD=1
export RLINF_RAY_LOCAL_NUM_GPUS=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="$HOME/.cache/rlinf-nvidia-590:${LD_LIBRARY_PATH:-}"

PYTHONPATH="$PWD" python -m pytest \
  tests/unit_tests/test_stackcube_pi05_state_sft.py -q

PYTHONPATH="$PWD" python examples/sft/train_vla_sft.py \
  --config-path "$PWD/examples/sft/config" \
  --config-name maniskill_stackcube_panda_sft_openpi_pi05_state \
  runner.max_steps=200 runner.save_interval=100 \
  actor.global_batch_size=16 actor.micro_batch_size=1 \
  data.num_workers=0 \
  +actor.model.openpi.torch_compile=false
```

Do not run concurrently with other GPU-heavy jobs on GPU 0. Checkpoint
should be at:
`../results/maniskill_stackcube_panda_pi05_state_sft/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt`.

### Comparable evaluation and diagnostics

On completion, use the registered state-conditioned transforms at
inference time; otherwise the policy may silently receive the wrong
prompt format. The existing evaluator and diagnostics script both
accept the new `--config` option:

```bash
PYTHONPATH="$PWD" python scripts/evaluate_stackcube_pi05.py \
  --config examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_state.yaml \
  --checkpoint ../results/maniskill_stackcube_panda_pi05_state_sft/checkpoints/global_step_200/actor \
  --label state_step200 \
  --episodes 20 --seed-start 10000 --max-videos 3 \
  --output ../results/maniskill_stackcube_panda_pi05_state_sft/evaluation/step_200

PYTHONPATH="$PWD" python scripts/diagnose_stackcube_pi05.py \
  --config examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_state.yaml \
  --checkpoint ../results/maniskill_stackcube_panda_pi05_state_sft/checkpoints/global_step_200/actor \
  --label state_step200 --episodes 1 \
  --expert-episodes 2 --expert-samples 20 \
  --output ../results/maniskill_stackcube_panda_pi05_state_sft/evaluation/step_200/diagnostics
```

Compare the original vision-only 0/20 result with the state-conditioned
policy on seeds 10000–10019 and the same `pd_joint_pos` controller.
The diagnostic summarizes expert action/gripper errors on 40 matching
expert observations, and both experiments have separate W&B runs.
Performance improvement is a **hypothesis**, not an asserted result.


## Milestone 9.4 — Recover incomplete step-200 DCP after disk exhaustion

The state-conditioned SFT run completed 200 updates and logged final loss
0.01935, but the original /home filesystem filled completely while saving.
Only the step-200 DCP shard `actor/dcp_checkpoint/__0_0.distcp` was
reported (6,162,284,544 bytes); `full_weights.pt` was missing. Step 100
has a working full checkpoint and evaluated at 0/20 successes.

After moving data to `/data1/Muttaqien/rlinf_data`, **preserve both original
checkpoints**. From the repository root:

```bash
git pull --ff-only origin branch-openpi-recap-chat

PYTHONPATH="$PWD" python -m pytest \
  tests/unit_tests/test_recover_stackcube_step200.py -q

PYTHONPATH="$PWD" python scripts/recover_stackcube_step200.py \
  --results-root ../results/maniskill_stackcube_panda_pi05_state_sft
```

Inspection checks whether step-100 and step-200 DCP metadata exist, whether
all model tensor byte ranges fit inside the step-200 shard, and whether three
probed tensors have the expected shape/dtype. No source files are changed.

If inspection passes and step-200 `.metadata` is missing, the experimental,
explicitly opted-in recovery is:

```bash
PYTHONPATH="$PWD" python scripts/recover_stackcube_step200.py \
  --results-root ../results/maniskill_stackcube_panda_pi05_state_sft \
  --recover --use-step100-layout
```

If step-200 metadata is present, use `--recover` without the template flag.
Recovery checks **every tensor** against the step-100 model's keys, shapes,
and dtypes, and writes the candidate to
`/data1/Muttaqien/rlinf_data/recovery/state_step200/actor/model_state_dict/full_weights.pt`.
It never overwrites the original `global_step_200` or `global_step_100`.

**Caution:** With no original step-200 metadata, using step-100 byte offsets
is provisional. Structural validation cannot prove the byte-to-key mapping
when multiple tensors have identical shapes and dtypes. Keep the result
clearly labeled as recovered/provisional.

If a full model is successfully recovered, inspect its predictions and
evaluate on the original 20 seeds using the **state-conditioned** config:

```bash
PYTHONPATH="$PWD" python scripts/evaluate_stackcube_pi05.py \
  --config examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_state.yaml \
  --checkpoint /data1/Muttaqien/rlinf_data/recovery/state_step200/actor \
  --label state_step200_recovered \
  --episodes 20 --seed-start 10000 --max-videos 3 \
  --output ../results/maniskill_stackcube_panda_pi05_state_sft/evaluation/step_200_recovered
```

If the inspection reports missing metadata, truncated model bytes, or
incompatible tensors, **do not invent or copy missing weights**. The safe
fallback is to resume 100–200 updates from the intact step-100 checkpoint,
after checking that its DCP `.metadata` exists. Write the resumed checkpoint
to a distinct experiment directory rather than overwriting this run.


### Actual lab recovery outcome and step-100 resume

October 22, 2026: three DCP recovery tests passed. Inspection of the actual
step-200 checkpoint reported 667 model tensors, a required final tensor byte
offset of 18,736,609,059 (using step-100 layout), but just 6,162,284,544
bytes in the step-200 shard. Step-200 `.metadata` was also absent. **Model-only
recovery is blocked; preserve the incomplete files for forensic analysis.**

RLinf's `SFTRunner.init_workers()` supports `runner.resume_dir`: it parses
`global_step_100`, restores the FSDP training state and continues up to
`runner.max_steps=200`. Use the original state-conditioned model config,
optimizer schedule and batch settings; write new results to a separate name.

From the repository root on the lab workstation, after migration to /data1:

```bash
source .venv/bin/activate
export PI05_MODEL_PATH="$PWD/checkpoints/torch/pi05_base_rlinf"
export STACKCUBE_DATASET="local/stackcube_sft_pilot100_lab"
export STACKCUBE_NORM_STATS="$PWD/assets/pi05_stackcube_panda_joint/local/stackcube_sft_pilot100_lab/norm_stats.json"
export EMBODIED_PATH="$PWD/examples/sft"
export CUDA_VISIBLE_DEVICES=0
export RAY_ADDRESS=local
export RAY_TMPDIR=/data1/Muttaqien/rlinf_data/ray
export RLINF_RAY_DISABLE_DASHBOARD=1
export RLINF_RAY_LOCAL_NUM_GPUS=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="$HOME/.cache/rlinf-nvidia-590:${LD_LIBRARY_PATH:-}"
mkdir -p "$RAY_TMPDIR" /data1/Muttaqien/rlinf_data/results

# Ensure the source checkpoint exists and there is sufficient space.
test -s ../results/maniskill_stackcube_panda_pi05_state_sft/checkpoints/global_step_100/actor/dcp_checkpoint/.metadata
test -s ../results/maniskill_stackcube_panda_pi05_state_sft/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt
df -h /data1/Muttaqien /home

PYTHONPATH="$PWD" python examples/sft/train_vla_sft.py \
  --config-path "$PWD/examples/sft/config" \
  --config-name maniskill_stackcube_panda_sft_openpi_pi05_state \
  +runner.resume_dir="$PWD/../results/maniskill_stackcube_panda_pi05_state_sft/checkpoints/global_step_100" \
  runner.logger.log_path=/data1/Muttaqien/rlinf_data/results \
  runner.logger.experiment_name=maniskill_stackcube_panda_pi05_state_sft_resume100 \
  runner.max_steps=200 runner.save_interval=100 \
  actor.global_batch_size=16 actor.micro_batch_size=1 \
  data.num_workers=0 \
  +actor.model.openpi.torch_compile=false
```

Expected new export:

`/data1/Muttaqien/rlinf_data/results/maniskill_stackcube_panda_pi05_state_sft_resume100/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt`

Do **not** change the scheduler total horizon from 200 steps when loading a
checkpoint containing the original step-100 scheduler state. This resumes
optimizer/model state from step 100. If the OpenPI loader is not a
`StatefulDataLoader`, the exact minibatch order may not be restored, so the
new step 200 may not exactly match the failed run.

Evaluate with `--config ...pi05_state.yaml`, `--checkpoint` pointing to
the *new* `global_step_200/actor`, and a distinct output directory. Even if
the new model has low SFT loss, do not infer task success without rollouts.


## Milestone 9.7 — State step-200 evaluation and task-stage diagnostics

The resumed state-conditioned step-200 model evaluated at **0/20** successes
on held-out seeds 10000–10019. Its 40 paired expert-observation predictions
(from two *training* demonstrations) improved dramatically over the vision-only
baseline: arm target MAE **0.02829 vs 0.17909** and gripper action absolute
error **0.05714 vs 0.71383**. Gripper direction agreed with the expert in
40/40 paired samples. This shows training-data imitation, **not** held-out
success or generalized closed-loop control.

The seed-10000 autonomous rollout ran 300 steps without success. The gripper
finger qpos sum reached zero at step 81 and stayed closed for the final 220
steps. No positive gripper commands appeared after step 70. Arm position
target MAE after control was 0.00523 radians on average, so joint-level
tracking is working. ManiSkill's StackCube-v1 success requires cube A to be
stacked on cube B, stable and no longer grasped. The earlier diagnostic
could not tell whether any grasp or stack happened.

The existing `scripts/diagnose_stackcube_pi05.py` now records the simulator's
`is_cubeA_grasped`, `is_cubeA_on_cubeB`, and `is_cubeA_static` fields and
TCP-to-cube-A distance (meters) when exposed by the environment. It reports
first grasp step, grasp/stack step counts and minimum grasp approach distance.
This is an observability-only change, not a policy/controller modification.

From the repository root, using the previously working CUDA environment:

```bash
PYTHONPATH="$PWD" python -m pytest tests/unit_tests/test_diagnose_stackcube_pi05.py -q
PYTHONPATH="$PWD" python scripts/diagnose_stackcube_pi05.py \
  --config examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_state.yaml \
  --checkpoint /data1/Muttaqien/rlinf_data/results/maniskill_stackcube_panda_pi05_state_sft_resume100/checkpoints/global_step_200/actor \
  --label state_step200_task_stages --episodes 1 --expert-episodes 1 \
  --expert-samples 1 --seed-start 10000 \
  --output /data1/Muttaqien/rlinf_data/results/maniskill_stackcube_panda_pi05_state_sft_resume100/evaluation/step_200/task_stages
```

Read the terminal `TASK STAGES` summary and the corresponding
`diagnostics.json`. Do not draw grasp conclusions from finger closure alone;
inspect `is_cubeA_grasped` and `is_cubeA_on_cubeB` first. If grasping
never happens, inspect gripper-to-cube approach and the recorded video before
extending the training schedule.


## Milestone 9.10 — Premature grasp closure and visual sensitivity

The task-stage diagnostic on seed 0 showed the first closing gripper
command at simulator step 61, when TCP was 0.187 m from red cube A.
At the closest approach, step 110, TCP was still 0.074 m from cube A,
and the gripper was already completely closed. No grasp occurred.
This is a positioning/command-timing failure, not an inability of the
joint controller to execute commands. It is not yet proof of absent
visual conditioning.

To isolate visual vs proprioceptive input sensitivity, use the new
non-destructive diagnostic. It resets ManiSkill twice, on seeds 0 and
10000, takes base camera images and qpos from the two scenes, and
predicts four action chunks with **identical flow noise**: baseline
(image0, qpos0), repeat (image0, qpos0), image swap (image10000,
qpos0), and state swap (image0, qpos10000). No actions are applied.
The same-input repeat quantifies unavoidable stochastic differences.
The report compares physical 8D action chunks, not normalized outputs.

From the repository root:

```bash
git pull --ff-only origin branch-openpi-recap-chat
PYTHONPATH="$PWD" python -m pytest \
  tests/unit_tests/test_stackcube_pi05_visual_sensitivity.py -q

PYTHONPATH="$PWD" python scripts/diagnose_stackcube_pi05_visual_sensitivity.py \
  --checkpoint /data1/Muttaqien/rlinf_data/results/maniskill_stackcube_panda_pi05_state_sft_resume100/checkpoints/global_step_200/actor \
  --config examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_state.yaml \
  --seed-a 0 --seed-b 10000
```

Output:
`/data1/Muttaqien/rlinf_data/results/maniskill_stackcube_panda_pi05_state_sft_resume100/evaluation/step_200/visual_sensitivity/visual_sensitivity.json`.

If swapping a visibly different image barely changes actions beyond
repeat noise, visual grounding is suspect. If action differences are
substantial, image information affects inference, but its causal
appropriateness and generalization still require evaluation. Replacing
one sensor modality while holding the other fixed produces an artificial
observation pairing, so do not interpret the size of the difference
as task performance.


## Milestone 9.12 — Object-position generalization at fixed policy state

The step-200 state-conditioned model improved action imitation, but all 20
held-out StackCube rollouts failed. On seed 0, closing began at 0.187 m from
red cube A (step 61), and the closest approach was 0.074 m at step 110.
On seed 10000 the closest approach was 0.288 m. A two-scene
visual sensitivity experiment found deterministic repeats and arm MAE
0.00777 rad for an image swap versus 0.03238 rad for a state swap.
These do not yet show that the policy uses the target cube's location.

The new `scripts/diagnose_stackcube_pi05_object_generalization.py` measures
initial-action sensitivity over eight deterministic reset seeds:
`0 1 2 3 10000 10001 10002 10003`. It reads both cube world poses and
the robot TCP pose, captures each scene's base camera, and asks the model
for 10-step actions using the **first scene's identical 9D qpos**,
unchanged task prompt, and seeded sampling noise. No actions are executed,
no object/robot state is changed, and no checkpoint is modified.

From the repository root after restoring the previously working CUDA
environment:

```bash
git pull --ff-only origin branch-openpi-recap-chat
PYTHONPATH="$PWD" python -m pytest \
  tests/unit_tests/test_stackcube_pi05_object_generalization.py -q

PYTHONPATH="$PWD" python scripts/diagnose_stackcube_pi05_object_generalization.py \
  --checkpoint /data1/Muttaqien/rlinf_data/results/maniskill_stackcube_panda_pi05_state_sft_resume100/checkpoints/global_step_200/actor \
  --seeds 0 1 2 3 10000 10001 10002 10003 \
  --save-images
```

Output:
`/data1/Muttaqien/rlinf_data/results/maniskill_stackcube_panda_pi05_state_sft_resume100/evaluation/step_200/object_generalization/object_generalization.json`
and optional per-seed PNGs. The CLI prints only three concise summary lines
(`OBJECT GENERALIZATION`, `POLICY RESPONSE`, `LIMITS`).

The JSON contains red/green cube world positions, initial TCP-to-cube
distance, the robot's *actual* qpos deviation (for image confounds),
every seed's first/last 8D action, and all pairwise cube separation,
image difference and action MAE values. The pairwise Pearson coefficient
is **descriptive only**, since pairs are not independent and eight scenes
cannot establish statistical significance.

Interpretation: if the red cube moves significantly while the 10-step
arm actions change little beyond repeat noise, suspect poor sensitivity to
object location. A nonzero correlation or change demonstrates scene
sensitivity, **not** that a robot trajectory moves toward the cube or that
the network responds specifically to red cube position; the green cube,
robot rendering and background may also differ between resets. Match
training and rollout camera settings before changing dataset/training code.


## Milestone 9.13 — Controlled red-cube relocation and closed-loop approach

Milestone 9.12 (eight scene resets) produced eight distinct red cube positions,
0.270 m X span and 0.332 m Y span, averaging 0.198 m pairwise red-cube
separation. With *fixed model proprioception and seeded sampling*, first
10-step arm action chunks differed by only 0.00463 rad MAE, and gripper
commands by 0.00769. Repeated identical inputs differed by 0.0.
The descriptive correlation between pairwise cube distance and action
difference was -0.0042. However the actual initial robot state varied by
up to 0.107 L2 across resets, while the green cube position also varied.
This is evidence of weak scene sensitivity but cannot isolate red-cube
conditioning or guarantee the first 10 actions should immediately differ.

`scripts/diagnose_stackcube_pi05_controlled_cube.py` improves this with
an **isolated red-cube intervention** and short policy rollouts. It gathers
valid red cube targets from seeded resets, but for *every* trial resets
the **same** base seed 0, then changes only cube A's pose. The robot,
camera, green cube, cube orientation and model task remain fixed; the
script checks robot 9D state, green cube and TCP coordinates are identical.
It re-renders the camera without stepping physics. It predicts the
initial 10 actions using fixed random flow noise, then rolls out 120
policy-controlled steps with real feedback from ManiSkill. Valid targets
must avoid overlap with the fixed green cube and initial TCP.
No training, dataset, controller or checkpoint files are changed.

From the lab repository root:

```bash
git pull --ff-only origin branch-openpi-recap-chat
PYTHONPATH="$PWD" python -m pytest \
  tests/unit_tests/test_stackcube_pi05_controlled_cube.py -q

PYTHONPATH="$PWD" python scripts/diagnose_stackcube_pi05_controlled_cube.py \
  --checkpoint /data1/Muttaqien/rlinf_data/results/maniskill_stackcube_panda_pi05_state_sft_resume100/checkpoints/global_step_200/actor \
  --base-seed 0 --max-scenes 6 --max-steps 120 --save-images
```

The default candidate seeds are `0 1 2 3 4 5 10000 10001 10002 10003
10004 10005`. The script uses the first six distinct collision-free red
cube positions relative to the common base scene. It prints four short
lines: `CONTROLLED CUBE`, `POLICY`, `OUTCOME`, and `Saved`.

Report:
`/data1/Muttaqien/rlinf_data/results/maniskill_stackcube_panda_pi05_state_sft_resume100/evaluation/step_200/controlled_cube/controlled_cube.json`.

The report includes each cube pose, actual initial/best/final TCP distance,
first closure distance and step, grasp status, image change relative to the
baseline, full first/last predicted action vectors in the initial chunk and
aggregate initial action differences.

This is a targeted *causal* scene perturbation because only the target red
cube pose varies before each rollout, unlike full seeded resets. It tests
approach and grasp outcomes, **not** complete StackCube success or
generalization to the whole training/held-out distribution. Similar actions
and failures across these controlled changes would support inadequate
object-conditioned closed-loop control. If the controlled results show
strong adaptation, investigate more complex scene differences and long
horizon rollout dynamics before collecting more data.


## Milestone 9.14 — Minimal SFT baseline: one-action closed-loop replanning

Do not retrain or run more scene-sensitivity diagnostics until checking the
inference/replanning cadence. The existing
`scripts/evaluate_stackcube_pi05.py` now accepts `--action-chunk 1`:
it still predicts 10 actions using exactly the same step-200 model,
but executes only the *first* action, reobserves RGB and 9D qpos,
then repredicts. The default `--action-chunk 10` preserves existing
evaluation behavior and comparability. The executed chunk is included
in W&B config and `results.json`.

Run once on seed 0 (same seed as the previous 300-step unsuccessful
task-stage experiment) before a longer multi-seed comparison:

```bash
git pull --ff-only origin branch-openpi-recap-chat
PYTHONPATH="$PWD" python -m pytest \
  tests/unit_tests/test_evaluate_stackcube_pi05.py -q

PYTHONPATH="$PWD" python scripts/evaluate_stackcube_pi05.py \
  --config examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_state.yaml \
  --checkpoint /data1/Muttaqien/rlinf_data/results/maniskill_stackcube_panda_pi05_state_sft_resume100/checkpoints/global_step_200/actor \
  --label state_step200_replan1_seed0 \
  --action-chunk 1 --episodes 1 --seed-start 0 \
  --max-steps 300 --max-videos 1 \
  --output /data1/Muttaqien/rlinf_data/results/maniskill_stackcube_panda_pi05_state_sft_resume100/evaluation/replan1_seed0
```

This uses up to **300 inference calls** for one 300-step episode versus
about **30 inference calls** with the original 10-action execution chunk.
It does not change the policy action-horizon architecture or weights.
If seed 0 succeeds (or visibly progresses), repeat on held-out seeds
10000–10019; if it still fails, focus the next decision on whether expert
action replay from saved seeds works and whether more SFT is justified.
All runtime results are placed under `/data1/Muttaqien/`.
