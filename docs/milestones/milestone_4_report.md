# Research Milestone 4 Report — Full StackCube π₀.₅ SFT

## Status

**PASSED for the authorized SFT run and inference readiness.** The standard
RLinf Ray/FSDP launcher completed all 1,084 approved optimizer steps on physical
GPU 1, wrote and verified checkpoints at steps 542 and 1,084, and the final
checkpoint loaded through RLinf's OpenPI evaluation factory. A real inference
call returned finite `(1, 10, 7)` actions. This does not establish policy
performance or task success; ManiSkill evaluation remains Milestone 5.

## Starting State and Objective

- Repository: `/home/muttaqien/Desktop/AIST-Tokyo/gamma/rlinf-project`
- Branch: `branch-openpi-recap`
- Starting commit: `1e096ce6` (`docs(recap): record bounded SFT commit`); the
  branch was six commits ahead of its remote.
- The starting worktree was clean. Existing datasets, checkpoints, and earlier
  milestone artifacts were preserved.
- The objective was a fresh four-epoch SFT run on the validated StackCube
  demonstrations, initialized from official pretrained OpenPI π₀.₅ weights.
  No rollout collection, RECAP, or Milestone 5 evaluation was run.

## Pre-Training Validation and Provenance

The LeRobot dataset at `outputs/stackcube_recap/lerobot_stackcube_v2` contains
10 complete demonstrations and 1,084 frames. It retains the 18D Panda state,
7D `pd_ee_delta_pose` action, and two RGB cameras. The official OpenPI loader
passed with actions `(1, 10, 32)`, state `(1, 32)`, two 224×224 image streams,
and 200 prompt tokens. It exposed 1,084 batches per epoch; with effective batch
size 4, the approved schedule is 271 optimizer updates per epoch.

Normalization remained at
`outputs/stackcube_recap/openpi_norm/stackcube_panda/norm_stats.json`, SHA-256
`9740881e6e1ff87773a0247397826ace1d2935a79e668105ec3eee45170456c7`.
No normalization or data representation changes were made.

The official OpenPI Orbax base source was
`gs://openpi-assets/checkpoints/pi05_base/params`, locally cached at
`/home/muttaqien/.cache/openpi/openpi-assets/checkpoints/pi05_base` and
converted with the existing RLinf utility to
`checkpoints/torch/pi05_stackcube_panda_base/model.safetensors` (13,413,809,816
bytes). The trained model factory reported Pi0.5, Gemma 2B plus 300M action
expert, 32D padded model action, horizon 10, and 3.35B parameters. The final
checkpoint loading log reported no missing or unexpected model keys.

The earlier M3.2 checkpoint at global step 10 was compatible with model resume,
but had no `data.pt` cursor. Exact data-order continuation was unavailable, so
this run deliberately started from the official pretrained base as a new
experiment. The existing SFT evaluation worker does not implement validation
inference, and the official loader does not expose the needed episode subset
selection here. A held-out trajectory validation set was therefore not
created; all ten episodes remained training data. No validation loss is
reported.

## Configuration and Approved Budget

The run used the dedicated config
`outputs/stackcube_recap/full_sft/config/stackcube_panda_sft_openpi_pi05.yaml`
and immutable resolved snapshot
`outputs/stackcube_recap/full_sft/config/resolved_training_config.yaml`.
The configured values were:

| Setting | Value |
|---|---:|
| Epochs / optimizer steps | 4 / 1,084 |
| Effective / micro-batch size | 4 / 1 |
| Gradient accumulation | 4 micro-batches |
| Optimizer | AdamW, β=(0.9, 0.95), ε=1e-8 |
| Peak learning rate | `2.5e-5` |
| Warmup / scheduler | 2 steps / cosine |
| Gradient clipping | 1.0 |
| Seed | 42 |
| Checkpoints | steps 542 and 1,084 |
| State / environment action | 18D / 7D |
| Model action padding / horizon | 32D / 10 |
| Cameras | Two RGB streams, 224×224 loader inputs |

The user approved this budget and the estimated 55–115 minute runtime and
55.6 GB of checkpoint writes before launch. No budget changes were made.

## Execution and Resource Results

The standard command used was:

```bash
set -o pipefail && env \
  LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib \
  REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" RAY_ADDRESS=local \
  CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=WARN PYTHONUNBUFFERED=1 \
  HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets \
  PYTHONPATH="$PWD:${PYTHONPATH:-}" timeout 7200s \
  .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path "$PWD/outputs/stackcube_recap/full_sft/config" \
  --config-name stackcube_panda_sft_openpi_pi05 \
  2>&1 | tee outputs/stackcube_recap/full_sft/logs/training.log
```

The final preflight confirmed physical GPU 1 was idle (97,235 MiB free of
97,887 MiB, 0% utilization, no compute process), system memory had 473 GiB
available, and disk had 166 GiB free. PyTorch was 2.11.0+cu130 / CUDA 13.0.
The process-scoped NVML workaround was
`/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib`; no system driver, CUDA,
kernel, or system configuration was changed. Ray enumerated the four devices
for placement, assigned the only actor to hardware rank 1 (`visible_accelerators=['1']`),
and used physical GPU 1 for training.

The launcher exited successfully after global step 1,084/1,084. The tqdm
training loop duration was 51:09; timestamps from launcher startup at 13:03:40
and the first idle GPU sample at 13:55:50 (Asia/Tokyo) bound total launch time
to about 52 minutes. Training had four complete data passes. No resume or
restart was needed. All 1,084 logged losses, gradient norms, and learning
rates were finite. The per-step values are preserved in `training.log`.

| Metric | First | Final | Minimum | Maximum | Mean |
|---|---:|---:|---:|---:|---:|
| Training loss | 0.141 | 0.0119 | 0.00222 | 0.141 | 0.022706 |
| Reported grad norm | 1.66 | 0.493 | 0.0942 | 3.51 | 0.563395 |
| Learning rate | 1.25e-5 | 6.25e-8 | 6.25e-8 | 2.5e-5 | 1.25318e-5 |

GPU 1 sampling at one-second intervals recorded peak used memory of 36,578
MiB, maximum utilization of 69%, and 60,673 MiB minimum free. The inference
readiness check used 6,808.2 MiB peak allocated and 6,968.0 MiB peak reserved.
After all work, GPU 1 returned to 97,235 MiB free, 0% utilization, and no
compute processes. RAM then had 472 GiB available; disk had 114 GiB free.

The runner warned that FSDP AMP was disabled by the existing configuration,
and PyTorch emitted a process-group barrier device-context warning. Neither
stopped training, caused non-finite metrics, or required a configuration
change. No worker failure or unsafe resource contention was observed.

## Checkpoints and Inference Readiness

Both periodic checkpoints contain DCP metadata, a nonempty distributed shard,
and consolidated full weights:

- `outputs/stackcube_recap/full_sft/results/stackcube_panda_pi05_full_sft/checkpoints/global_step_542/`
- `outputs/stackcube_recap/full_sft/results/stackcube_panda_pi05_full_sft/checkpoints/global_step_1084/`

Each DCP shard is 20,843,201,461 bytes and each `full_weights.pt` is
6,946,527,306 bytes. The final weights loaded successfully with RLinf's
existing `get_model` factory using `task=eval`, loaded the existing norm stats,
and matched the expected 3.35B parameter Pi0.5 architecture. Comparing the
trained `action_out_proj.weight` with the same tensor from the official base
found a nonzero L2 delta of `0.09267354`.

One call through `Pi0Eval.predict_action_batch` used the first actual dataset
sample and the existing input/output transform pipeline. Inputs were
`main_images=(1,128,128,3)`, `wrist_images=(1,128,128,3)`, and
`states=(1,18)`; the task prompt was “Stack the red cube on the green cube”.
The finite output had shape `(1,10,7)` and range `[-0.22195, 1.01727]`.
This verifies checkpoint loading and inference interface compatibility only;
the predicted action was not executed in ManiSkill.

## Tests, Logs, Changes, and Git

- Dataset input adapter and official OpenPI loader checks passed during
  preflight; expected state/action/image/prompt shapes and normalization were
  observed.
- The standard RLinf Ray/FSDP training launcher completed all 1,084 steps and
  wrote both scheduled checkpoints.
- The RLinf/OpenPI evaluation factory loaded the final checkpoint and completed
  one finite inference call with the expected action shape.
- Final resource check confirmed GPU 1 idle and no remaining compute process.
- `git diff --check` is run after this report and progress log are finalized.
  No source or tracked experiment configuration changed; no additional unit
  test was needed.

Persistent output is under `outputs/stackcube_recap/full_sft/`, including
`logs/training.log`, `logs/gpu_memory.csv`, `logs/resource_watch.log`,
`logs/postrun_resources.log`, `logs/inference_readiness.log`,
`logs/inference_readiness.json`, `logs/inference_readiness_first_attempt.log`,
and `logs/final_resources.log`. The inference check script is also stored
under `logs/` for reproducibility. The preflight and inference first-attempt
failures are retained in their logs; the inference script was corrected to
initialize the CUDA context before resetting peak memory counters, then
succeeded.

Tracked changes are this report and `docs/CODEX_EXPERIMENT_PROGRESS.md`. No
Milestone 4 commit was created. A prior sandbox attempt to write `.git/index`
was blocked because `.git` is read-only; repository permissions were not
changed, and no bypass was attempted. No commit was pushed. The starting
commit remains `1e096ce6`.

## Limitations and Next Milestone

This run establishes successful execution and numerical stability for the
approved short SFT budget, plus checkpoint/inference compatibility. The
decreasing training loss does not establish convergence, generalization, or
StackCube task success. There is no held-out validation loss, and no policy
was evaluated in the simulator.

The next research milestone is **Milestone 5: baseline ManiSkill policy
evaluation**. It requires separate user approval; no further training,
rollouts, or RECAP work is authorized by this report.
