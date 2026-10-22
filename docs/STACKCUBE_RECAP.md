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
