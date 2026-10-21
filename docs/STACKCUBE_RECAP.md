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

### Install the RLinf OpenPI runtime in a separate environment

The ManiSkill data-collection venv does not necessarily contain OpenPI.
Use a dedicated environment, leaving the working collector `.venv` intact:

```bash
# From the repository root
bash requirements/install.sh embodied --model openpi --env maniskill_libero \
  --venv .venv-openpi --no-flash-attn --no-root
source .venv-openpi/bin/activate

python -c "import openpi, openpi.training.data_loader; print(openpi.__file__)"
uv pip show rlinf-openpi
```

The installer pins `rlinf-openpi==0.1.1`, its compatible transformer
package, JAX/Orbax versions, and tokenizer assets. Do not simply install
a generic `openpi` package into the simulator venv.

Native NVIDIA hosts usually provide the ICD at
`/usr/share/vulkan/icd.d/nvidia_icd.json`; the installer now selects
that path when present, otherwise retaining the Docker fallback
`/etc/vulkan/icd.d/nvidia_icd.json`. Verify with `vulkaninfo --summary`
if the OpenPI environment is also used for RGB rendering.

The `--no-flash-attn` flag skips an optional CUDA-kernel installation for
initial import/statistics tests. Training may require additional GPU
libraries; verify that separately on the intended training device.

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

This fixes **normalization only**: the currently installed OpenPI SFT loader
still uses LeRobot's v2.1 reader and cannot directly ingest multi-episode
v3 shards. Prepare a verified v3-to-v2.1 training view or a v3-capable
loader before attempting model training. Do **not** change the original
`stackcube_sft_pilot100` data or spoof `codebase_version` merely to avoid
an exception.

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

### Run a short training smoke test (after norm stats and checkpoint setup)

The model requires a compatible **PyTorch π₀.₅** checkpoint. Set paths via
environment variables (no hard-coded workstation usernames):

```bash
export STACKCUBE_DATASET=local/stackcube_sft_pilot100
export STACKCUBE_NORM_STATS=/absolute/path/from/stats-script/norm_stats.json
export PI05_MODEL_PATH=/absolute/path/to/pi05_base_pytorch

# Check before launch
test -f "$STACKCUBE_NORM_STATS" || { echo "Missing normalization stats"; exit 1; }
test -d "$PI05_MODEL_PATH" || { echo "Missing Pi0.5 checkpoint directory"; exit 1; }
ls -lh "$PI05_MODEL_PATH" | head

bash examples/sft/run_vla_sft.sh maniskill_stackcube_panda_sft_openpi_pi05
```

Use a compatible OpenPI PyTorch checkpoint in the format expected by RLinf,
and inspect its contents before running; the directory-existence check does
not validate checkpoint weights.
The YAML defaults to a 200-step **configuration smoke**, not an evaluated
baseline model. Full-parameter π₀.₅ with `no_shard` may exceed a 16 GB GPU;
the 96 GB GPU server is the safer place to try initial training.
Do not launch on a production device without confirming memory.

After validating training and action-representation consistency, increase
successful training demonstrations toward 1,000, keeping evaluation seeds
held out. Subsequent RECAP stages still require actual policy rollouts
(including failures), value-model checkpoints, advantage inference and CFG.

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
