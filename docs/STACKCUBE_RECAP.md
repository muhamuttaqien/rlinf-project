# OpenPI + RECAP: Panda StackCube-v1

This branch isolates the **Franka Panda / ManiSkill StackCube-v1** experiment from the existing state-based SAC-Flow work.

## Status

**Phase 0 — simulator and data-interface smoke test.** This is not an end-to-end RECAP training implementation yet. The random-action probes below are **not expert demonstrations** and must **never** be passed to RECAP as the successful SFT dataset. RGB tensors are inspected but are not yet serialized into LeRobot episodes.

## Environment check

From the repository root in the existing embodied virtual environment:

```bash
git switch branch-openpi-recap
source .venv/bin/activate
PYTHONPATH="$PWD" python scripts/smoke_test_stackcube.py --robot panda_wristcam --obs-mode rgb --steps 10
PYTHONPATH="$PWD" python scripts/probe_stackcube_recap.py --episodes 3 --max-steps 100
cat outputs/stackcube_recap_probe/schema.json
cat outputs/stackcube_recap_probe/summary.json
```

The probe checks Panda camera observations, the 7D-ish end-effector action interface (inspect actual dimension), reset/step, and task-success reporting. The output directory is local experiment data; exclude it from commits.

## Planned progression

1. **Expert data:** Obtain successful StackCube trajectories via a validated ManiSkill scripted motion planner or teleoperation. Verify all successes by environment info and video; do not use random probes as demonstrations.
2. **LeRobot conversion:** Export the **full synchronized RGB observations, proprioceptive state, actions, timestamps, task text, and per-episode outcomes**, with correct feature schema, metadata, and episode indexing. Validate replay and inspect sample videos.
3. **π₀.₅ SFT:** Map the Panda image/state/action keys to the selected OpenPI RLinf model adapter; decide the actual action horizon and normalization using real dataset statistics.
4. **Rollouts:** Evaluate SFT policy using different seeds; collect success and failure episodes, preserving terminal outcomes. Measure baseline success rate before RECAP.
5. **RECAP stage 1:** `bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh recap_compute_returns` with dataset and tag set in `examples/offline_rl/config/recap_compute_returns.yaml`.
6. **RECAP stage 2:** `bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh recap_value_model_sft` with a valid SigLIP2/Gemma3 setup and matching tag.
7. **RECAP stages 3–4:** Compute advantages with the value-model checkpoint, then train the π₀.₅ CFG policy using `recap_compute_advantages.yaml` and `cfg_rl_openpi.yaml`. Consult the commands in the upstream RLinf RECAP guide for the exact checkout.
8. **Evaluation:** Same seeds and episode budget as baseline; log success, trajectory length, checkpoints, and simulation videos to W&B.

Reference: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html

## Caveats

- `StackCube-v1` is a ManiSkill **simulation** with Panda; this is not real-world FR3 support.
- RECAP is **offline**. Simulator interactions happen during demonstration and rollout collection, not inside the four optimization stages.
- The existing examples are LIBERO-oriented; Panda-specific LeRobot feature mappings and checkpoint compatibility require verification before training.
- A successful smoke test is necessary but not sufficient proof that full RECAP works.

## Phase 1 complete: LeRobot dataset validation

Validated on `pca03395`: 10 successful StackCube demonstrations, 1,084 total
frames, 18-dim Panda proprioception (qpos+qvel), 7-dim `pd_ee_delta_pose`
actions, two 128×128 RGB videos. TorchCodec decoding, per-frame action
alignment, and terminal success checks passed.

## Phase 2 preflight: OpenPI π₀.₅ SFT

The StackCube-specific adapter is registered as `pi05_stackcube_panda`.
It maps LeRobot keys `observation.images.base_camera`,
`observation.images.hand_camera`, `observation.state`, `action`, and
`task` onto the existing RLinf ManiSkill policy transform. The official
OpenPI loader constructs 10-step action windows from the singular
`action` key and pads state/actions to 32 model dimensions. Actions are
*already* delta-pose commands, so there is no second delta conversion.

Run from the repository root with the embodied virtual environment:

```bash
git pull --ff-only origin branch-openpi-recap
python -c "import openpi, lerobot; print('OpenPI and LeRobot import OK')"
PYTHONPATH="$PWD" python scripts/check_stackcube_openpi_inputs.py
PYTHONPATH="$PWD" python scripts/compute_stackcube_openpi_norm_stats.py
PYTHONPATH="$PWD" python scripts/check_stackcube_openpi_loader.py
```

The stats are computed from the **same replayed dataset** using 10-step
future action windows and stored at
`outputs/stackcube_recap/openpi_norm/stackcube_panda/norm_stats.json`.
Regenerate statistics if the training set changes. Do not train with stale
normalization or silently reuse norms from LIBERO or another robot.

A first short SFT experiment configuration is at:
`examples/sft/config/stackcube_panda_sft_openpi_pi05.yaml`.
It is **not yet a verified training recipe**. It requires a compatible
pretrained PyTorch π₀.₅ checkpoint, sufficient GPU memory, and a successful
full dataloader preflight. When those prerequisites are met, an operator
can run:

```bash
export REPO_PATH="$PWD"
export EMBODIED_PATH="$PWD/examples/sft"
PYTHONPATH="$PWD:${PYTHONPATH:-}" python examples/sft/train_vla_sft.py \
  --config-path "$PWD/examples/sft/config" \
  --config-name stackcube_panda_sft_openpi_pi05 \
  actor.model.model_path=/absolute/path/to/pi05_base_pytorch
```

The direct Python invocation accepts Hydra overrides, unlike the
`run_vla_sft.sh` helper. Do not launch training until the checkpoint path
and hardware resources are verified.

**Resource caution:** A 16 GB GPU is not guaranteed to accommodate π₀.₅ SFT,
even with a micro-batch of 1 and gradient checkpointing. CPU/loader checks
are deliberately separated from GPU training.

Future milestones: scale to 100–200 validated demonstrations; establish a
π₀.₅ SFT baseline; collect autonomous successful *and failed* rollouts;
apply RLinf's four RECAP offline stages; compare held-out success rates.

## SFT checkpoint and hardware preflight (next milestone)

Before loading π₀.₅, inspect GPU capacity (the workstation GPU is not yet
established) and confirm the checkpoint layout:

```bash
git pull --ff-only origin branch-openpi-recap
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
PYTHONPATH="$PWD" python scripts/check_stackcube_sft_readiness.py
# When you have a converted base checkpoint:
PYTHONPATH="$PWD" python scripts/check_stackcube_sft_readiness.py \
  --checkpoint /absolute/path/to/pi05_base_openpi
```

A model checkpoint is **not** part of the 10 demonstration episodes.
RLinf's supported base checkpoint layout stores bare `Pi0` tensors
(`llm.*`, `img.*`, …) in `model.safetensors`. Arbitrary third-party
π₀.₅ safetensors files may contain the official upstream
`paligemma_with_expert.*` layout and need conversion.

Official π₀.₅ base weights are available as a JAX Orbax checkpoint at
`gs://openpi-assets/checkpoints/pi05_base`. See
https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft_openpi.html
and the in-repo
`rlinf/utils/ckpt_convertor/openpi/README.md`.
To download (requires Google Cloud CLI and enough disk space):

```bash
mkdir -p checkpoints/jax
gcloud storage cp --recursive \
  gs://openpi-assets/checkpoints/pi05_base checkpoints/jax/
```

To convert to the RLinf layout using the **same StackCube norm stats** and
the action horizon chosen in `pi05_stackcube_panda`:

```bash
python -m rlinf.utils.ckpt_convertor.openpi.convert \
  --mode jax_to_openpi \
  --input-model "$PWD/checkpoints/jax/pi05_base" \
  --input-norm-stats "$PWD/outputs/stackcube_recap/openpi_norm/stackcube_panda/norm_stats.json" \
  --output-model "$PWD/checkpoints/torch/pi05_stackcube_panda_base" \
  --output-norm-stats "$PWD/checkpoints/torch/pi05_stackcube_panda_base/stackcube_panda/norm_stats.json" \
  --action-dim 32 --action-horizon 10 --max-token-len 200
```

Do not start this large model download or conversion unless GPU memory,
storage, and checkpoint provenance are understood. The preflight reads
only safetensors metadata, **not** all model tensors; it cannot prove
the full model can train.

The SFT YAML deliberately sets `openpi.torch_compile: False` for easier
first-step debugging. The preflight YAML uses micro-batch 1, but π₀.₅
full-model gradients and Adam optimizer states may still exceed a single
GPU's memory. No multi-GPU/FSDP allocation or QLoRA solution is validated
for this StackCube config.

Once the model and resources are verified, make the first test **one
optimization step** with explicit Hydra overrides:

```bash
export REPO_PATH="$PWD"
export EMBODIED_PATH="$PWD/examples/sft"
PYTHONPATH="$PWD:${PYTHONPATH:-}" python examples/sft/train_vla_sft.py \
  --config-path "$PWD/examples/sft/config" \
  --config-name stackcube_panda_sft_openpi_pi05 \
  actor.model.model_path="$PWD/checkpoints/torch/pi05_stackcube_panda_base" \
  runner.max_steps=1 runner.save_interval=1
```

Record whether it reaches an optimizer step, checkpoint write, and the
peak GPU memory usage. The complete RLinf training run is not yet
validated; RECAP can begin only after SFT baseline and rollout collection.
