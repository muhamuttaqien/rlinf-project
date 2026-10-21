# Research Milestone 3.2 Report — Bounded StackCube SFT

## Status

**PASSED for bounded training execution, numerical checks, checkpoint saving,
and resume.** The standard RLinf Ray/FSDP launcher completed nine new
optimizer updates in this milestone, resumed from two saved checkpoints, and
reached global step 10 in the same standard-launcher checkpoint lineage.
Every recorded loss and global gradient norm was finite. This does not establish
learning convergence, policy improvement, or StackCube task success.

No model architecture, action representation, normalization, dataset, system
configuration, driver, CUDA, or PyTorch changes were made. No full training was
started.

## Starting state and safeguards

- Repository: `/home/muttaqien/Desktop/AIST-Tokyo/gamma/rlinf-project`
- Branch: `branch-openpi-recap`
- Starting HEAD: `1d9ffebc` (`docs(recap): report Ray NCCL readiness`), branch
  ahead of origin by four commits.
- Root `AGENTS.md`, `docs/STACKCUBE_RECAP.md`, the existing StackCube SFT YAML,
  and Milestone 3/3.1 reports were read.
- Existing artifacts were preserved. The selected input checkpoint was the
  prior standard-launcher FSDP checkpoint at `global_step_1`; existing
  standalone step-1 artifacts and experiment data were left in place.
- Preflight on physical GPU 1: RTX PRO 6000 Blackwell Max-Q, 97,235 MiB free,
  0% utilization, and no compute process. PyTorch 2.11.0+cu130 initialized
  logical CUDA device 0 with `CUDA_VISIBLE_DEVICES=1`.
- Disk space before training: 218 GiB free. After the two new checkpoints:
  166 GiB free.

The runs reused the process-scoped Milestone 3.1 NVML workaround:
`LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib`. It uses the
already-installed NVML 590.48.01 binary matching the host driver. No system
configuration was changed.

## Run plan and step budget

The existing step-1 checkpoint contains model, optimizer, scheduler, FSDP
version, and RNG state. The SFT runner resumes at the step parsed from the
`global_step_N` directory. To keep exactly ten standard-launcher updates in
this checkpoint lineage, counting the one completed in Milestone 3.1:

1. Resume `global_step_1`, set `runner.max_steps=9`, and run eight updates.
   Save only the final state as `global_step_9`.
2. Resume `global_step_9`, set `runner.max_steps=10`, and run one update.
   Save the final state as `global_step_10`.

Milestone 3.2 therefore performed **nine optimizer updates**, not ten. The
prior step-1 update is included in the ten-step standard-launcher lineage. An
earlier standalone Milestone 3 smoke test was a separate output/checkpoint
lineage and was not used for this resume sequence.

The temporary config places only the actor on hardware rank 1. The launcher
enumerated the workstation's four GPU resources for RLinf placement, while the
actor's worker placement assigned only physical GPU 1 (`visible_accelerators=['1']`).
The environment was otherwise the existing one-GPU FSDP run used in Milestone
3.1.

## Commands

First segment, resume step 1 through step 9:

```bash
set -o pipefail; mem_log=outputs/stackcube_recap/milestone3/logs/m32_gpu_memory_run_a.csv; nvidia-smi --id=1 --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader --loop-ms=1000 > "$mem_log" 2>&1 & monitor_pid=$!; trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT; env LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=WARN PYTHONUNBUFFERED=1 HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD:${PYTHONPATH:-}" timeout 1800s .venv/bin/python examples/sft/train_vla_sft.py --config-path "$PWD/outputs/stackcube_recap/milestone3/config" --config-name stackcube_panda_sft_openpi_pi05 actor.model.model_path="$PWD/checkpoints/torch/pi05_stackcube_panda_base" actor.global_batch_size=1 runner.max_steps=9 runner.save_interval=9 runner.logger.log_path="$PWD/outputs/stackcube_recap/milestone3/results" +runner.resume_dir="$PWD/outputs/stackcube_recap/milestone3/results/stackcube_panda_pi05_sft/checkpoints/global_step_1" 2>&1 | tee outputs/stackcube_recap/milestone3/logs/m32_sft_resume_step1_to9.log
```

Second segment, resume step 9 through step 10:

```bash
set -o pipefail; mem_log=outputs/stackcube_recap/milestone3/logs/m32_gpu_memory_run_b.csv; nvidia-smi --id=1 --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader --loop-ms=1000 > "$mem_log" 2>&1 & monitor_pid=$!; trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT; env LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=WARN PYTHONUNBUFFERED=1 HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD:${PYTHONPATH:-}" timeout 900s .venv/bin/python examples/sft/train_vla_sft.py --config-path "$PWD/outputs/stackcube_recap/milestone3/config" --config-name stackcube_panda_sft_openpi_pi05 actor.model.model_path="$PWD/checkpoints/torch/pi05_stackcube_panda_base" actor.global_batch_size=1 runner.max_steps=10 runner.save_interval=10 runner.logger.log_path="$PWD/outputs/stackcube_recap/milestone3/results" +runner.resume_dir="$PWD/outputs/stackcube_recap/milestone3/results/stackcube_panda_pi05_sft/checkpoints/global_step_9" 2>&1 | tee outputs/stackcube_recap/milestone3/logs/m32_sft_resume_step9_to10.log
```

Both commands exited with status 0. They used the existing
`examples/sft/train_vla_sft.py` runner, FSDP backend, pretrained Pi0.5
checkpoint, StackCube LeRobot dataset, normalization statistics, batch size 1,
and AdamW configuration. No implementation code changed, so no unit tests were
run. The two actual training runs exercised the requested launcher and
checkpoint resume paths.

## Step-wise training history

Values came from TensorBoard event files. The event step is the runner's
zero-based update index; “optimizer update” below is one-based. Update 1 was
completed in Milestone 3.1. Updates 2–9 came from the first M3.2 resumed run;
update 10 came from the resumed step-9 run. Gradient norm is RLinf's reported
global norm before/while applying the configured `clip_grad=1.0` operation.

| Optimizer update | Loss | Grad norm | Learning rate | Run |
|---:|---:|---:|---:|---|
| 1 | 0.0460704379 | 0.805865467 | 1.24999997e-5 | Milestone 3.1 checkpoint |
| 2 | 0.0884265825 | 1.607570529 | 2.49999994e-5 | M3.2 resume step 1→9 |
| 3 | 0.1728941053 | 5.640073776 | 2.40508725e-5 | M3.2 resume step 1→9 |
| 4 | 0.0674103126 | 1.150674224 | 2.13479871e-5 | M3.2 resume step 1→9 |
| 5 | 0.1475402862 | 3.173048973 | 1.73028347e-5 | M3.2 resume step 1→9 |
| 6 | 0.0366445780 | 1.074344754 | 1.25312499e-5 | M3.2 resume step 1→9 |
| 7 | 0.0976210758 | 1.376269102 | 7.75966600e-6 | M3.2 resume step 1→9 |
| 8 | 0.0517143980 | 0.717163324 | 3.71451233e-6 | M3.2 resume step 1→9 |
| 9 | 0.0676105991 | 1.485428214 | 1.01162709e-6 | M3.2 resume step 1→9 |
| 10 | 0.0073775412 | 0.366477668 | 6.24999998e-8 | M3.2 resume step 9→10 |

The nine new M3.2 losses and gradient norms were all finite. No “Non-finite
grad norm” skip message, NaN/Inf metric, exception, or traceback appeared in
either launcher log. The largest reported gradient norm was 5.6401; the
configured norm clipping threshold is 1.0. Losses varied across batches and
steps. That history is not evidence of convergence or policy improvement.

## GPU memory and post-run state

GPU 1 was sampled every second using `nvidia-smi`; raw samples are in
`m32_gpu_memory_run_a.csv` and `m32_gpu_memory_run_b.csv`.

- Samples: 258 across the two runs.
- Peak sampled used memory: **41,642 MiB** of 97,887 MiB.
- Minimum sampled free memory: **55,609 MiB**.
- Peak sampled utilization: **76%**.
- Post-run: 97,235 MiB free, 0% utilization, no compute processes.
- Disk after both new checkpoint saves: 166 GiB free.

These are host-level one-second samples, not PyTorch allocator peak counters.

## Checkpoints and resume verification

New checkpoints:

- `outputs/stackcube_recap/milestone3/results/stackcube_panda_pi05_sft/checkpoints/global_step_9/`
- `outputs/stackcube_recap/milestone3/results/stackcube_panda_pi05_sft/checkpoints/global_step_10/`

Each contains a DCP shard (`__0_0.distcp`, 20,843,201,397 bytes), DCP metadata
(about 973 KB), and full model weights (`full_weights.pt`, 6,946,527,306
bytes). Metadata inspection found model, optimizer, LR scheduler, RNG, and
FSDP-version state. Both shards and full weights are nonempty. Loading
`global_step_1` and then `global_step_9` through `SFTRunner.init_workers()`
completed without error; the run progress started from steps 1 and 9
respectively, and the second resume completed the update to step 10. This
validates restoration and continuation, including optimizer/scheduler state.
The actor checkpoint directories do not contain `data.pt`; the SFT worker only
saves the loader cursor sidecar when the loader is a `StatefulDataLoader`.
Therefore these resumes did not restore or verify the OpenPI data iterator
position. The training loader may restart its iteration after resume; this
run validates training-state restoration, not exact sample-order continuation.

## Stability versus policy performance

- **Execution:** the Ray/FSDP launcher completed the bounded training and both
  checkpoint resume paths.
- **Numerical stability:** all nine new losses and global grad norms were
  finite. One norm exceeded the configured clipping threshold and was handled
  by the configured clipping operation. No non-finite skip occurred.
- **Policy performance:** not measured. No ManiSkill evaluation, rollout,
  convergence analysis, or RECAP work was run. The loss sequence is not a
  task-success metric.

## Changes, logs, and commits

- Tracked code/config changes: none.
- Requested docs: this report and the Milestone 3.2 section in
  `docs/CODEX_EXPERIMENT_PROGRESS.md`.
- Launcher logs: `outputs/stackcube_recap/milestone3/logs/m32_sft_resume_step1_to9.log`
  and `m32_sft_resume_step9_to10.log`.
- GPU memory logs: `m32_gpu_preflight.log`, `m32_gpu_memory_run_a.csv`,
  `m32_gpu_memory_run_b.csv`, and `m32_gpu_postrun.log` under the same logs
  directory.
- TensorBoard scalars: `outputs/stackcube_recap/milestone3/results/tensorboard/`.
- Starting HEAD: `1d9ffebc`. Documentation commit will be recorded in the
  progress log after it is created. No commit was pushed.

## Recommendation

Stop after this bounded validation and review the report and checkpoints.
Further SFT duration, evaluation, rollout collection, and RECAP require
separate authorization. No learning or task-success claim follows from this
ten-update checkpoint lineage.
