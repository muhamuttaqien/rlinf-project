# Codex Experiment Progress — StackCube π₀.₅ + RECAP

Updated: 2026-10-21 (Asia/Tokyo)  
Branch: `branch-openpi-recap`  
Current milestone: Research Milestone 4 — **PASSED (1,084/1,084 steps; inference readiness passed)**

## State at start

- The required branch was checked out at `8c6e3e10`; it matched
  `origin/branch-openpi-recap`.
- `docs/milestones/` already contained untracked Milestone 0 and 1 reports. They
  were preserved and were not staged.
- The LeRobot v2 dataset, StackCube normalization statistics, SFT YAML, loader
  checks, and the official OpenPI conversion utilities were present.
- The dataset contains 10 episodes and 1,084 frames. Dataset loading produced
  the previously validated 18D observation state and 7D action before OpenPI
  padding.

## Milestone 3 execution

### Phase A — Repository, data, software, and GPU

- Read `AGENTS.md`, `docs/STACKCUBE_RECAP.md`, the StackCube SFT YAML, the OpenPI
  model factory/checkpoint loader, the SFT worker/runner, and the checkpoint
  converter documentation.
- The project `.venv` imports PyTorch 2.11.0+cu130, LeRobot 0.3.3, JAX 0.5.3,
  and Orbax 0.11.13. `openpi` imports from `.venv/site-packages`; that install
  does not expose package version metadata.
- Disk had 263 GiB free before conversion; 244 GiB remained after conversion
  and the one-step checkpoint. Large weights and run outputs are ignored by
  Git.
- The sandbox could not access NVIDIA devices. Host-context preflight found
  four RTX PRO 6000 Blackwell Max-Q GPUs. GPU 1 had 97,235 MiB free, 0%
  utilization, and no compute process; it was selected. With
  `CUDA_VISIBLE_DEVICES=1`, PyTorch saw one logical CUDA device.
- PyTorch CUDA worked, but NVML initialization failed with
  `NVMLError_LibRmVersionMismatch`. This affected Ray accelerator detection and
  later NCCL initialization; it did not mean the CUDA device or NVIDIA driver
  was absent.

### Phase B — Official checkpoint

- Reused the complete local Orbax source at
  `/home/muttaqien/.cache/openpi/openpi-assets/checkpoints/pi05_base` (12 GiB,
  including `params/commit_success.txt`). No weights were downloaded.
- The local OpenPI source checkout points its `pi05_base` training config at
  `gs://openpi-assets/checkpoints/pi05_base/params`; its HEAD was
  `15a9616a00943ada6c20a0f158e3adb39df2ccac`. The checkout has unrelated local
  modifications, which were not changed. No separate asset revision/hash
  manifest was found.
- Converted with the existing `jax_to_openpi` utility, using the StackCube
  normalization file and action dimension 32, horizon 10, token length 200.
  The result is `checkpoints/torch/pi05_stackcube_panda_base/` with
  `model.safetensors`, `config.json`, and copied StackCube norm stats.
- Safetensors header verification found 667 tensors and bare Pi0 prefixes
  (`img.*`, `llm.*`, etc.); checkpoint size is 12.49 GiB. The generated config
  says Pi0.5, Gemma 2B + 300M action expert, action dim 32, horizon 10, and
  token length 200.

### Phase C — Model/data loading

- The StackCube input check passed with two active camera streams, 18D state,
  7D actions, and the existing task prompt. A first attempt failed because
  Hugging Face tried to create a lock in its read-only user cache; rerunning
  with `HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets` passed without changing data.
- The official OpenPI loader passed: actions `(1,10,32)`, state `(1,32)`,
  camera tensors 224×224, prompt tokens `(1,200)`, and StackCube norm stats
  loaded from the validated path.
- RLinf's model factory loaded the checkpoint as Pi0.5 with 3,353,433,872
  parameters. The loader rejects missing backbone tensors and unexpected
  tensors; it reported the intended action horizon and config.

### Phase D — SFT smoke test

- `train_vla_sft.py` first attached to a pre-existing shared Ray cluster and
  stopped before worker startup because it had Ray 2.59.0/Python 3.11.15 while
  the cluster had Ray 2.58.0/Python 3.11.14. The shared cluster was not stopped.
- `RAY_ADDRESS=local` isolated Ray, but NVML failure made RLinf report zero
  accelerators. A focused fallback was added to the NVIDIA accelerator adapter:
  when NVML returns no devices, use PyTorch's visible CUDA count and device
  name. Two targeted regression tests passed. Code/test commit:
  `b140d271a6d58f9f31894f857f4ac5e568f7bc25`.
- With all devices visible to the scheduler and a temporary copy of the same
  SFT YAML placing the actor on rank 1, RLinf selected physical GPU 1 and loaded
  the pretrained model. FSDP initialization then failed inside NCCL with
  `nvmlInit_v2() failed: Driver/library version mismatch`. Retrying once with
  `NCCL_P2P_DISABLE=1` produced the same NCCL error. No optimizer step occurred
  in either launcher attempt.
- The bounded fallback used RLinf's Pi0.5 model factory, the official StackCube
  SFT dataloader, `ForwardType.SFT`/the model's real flow-matching supervised
  loss, and RLinf's `build_adamw` with the YAML optimizer values. It loaded one
  batch and called `optimizer.step()` exactly once on physical GPU 1.
- Observed loss: `0.19008885324001312` (finite). Finite gradients were verified
  for `action_in_proj.weight`, `time_mlp_in.weight`, and
  `action_out_proj.weight`, with respective norms `0.0828830`, `0.7388377`, and
  `0.7832464`. `action_out_proj.weight` changed by max absolute `2.50004e-05`.
- Peak CUDA allocated memory was 27,225,376,256 bytes (25.36 GiB); peak
  reserved memory was 27,715,960,832 bytes (25.81 GiB). GPU 1 returned to
  97,235 MiB free, 0% utilization, and no compute process after exit.
- Saved step-1 model checkpoint:
  `outputs/stackcube_recap/milestone3/results/stackcube_panda_pi05_sft_single_step.pt`
  (6,946,594,594 bytes). ZIP integrity verification passed. Metrics are in
  `outputs/stackcube_recap/milestone3/results/single_step_metrics.json`.

## Commands and validation

- Official checkpoint conversion (completed in host context with CPU JAX):

  ```bash
  JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 PYTHONPATH="$PWD" timeout 600s \
    .venv/bin/python -m rlinf.utils.ckpt_convertor.openpi.convert \
    --mode jax_to_openpi \
    --input-model /home/muttaqien/.cache/openpi/openpi-assets/checkpoints/pi05_base \
    --input-norm-stats outputs/stackcube_recap/openpi_norm/stackcube_panda/norm_stats.json \
    --output-model "$PWD/checkpoints/torch/pi05_stackcube_panda_base" \
    --output-norm-stats "$PWD/checkpoints/torch/pi05_stackcube_panda_base/physical-intelligence/stackcube_panda/norm_stats.json" \
    --action-dim 32 --action-horizon 10 --max-token-len 200
  ```

- Passed dataset checks:
  `HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD" .venv/bin/python scripts/check_stackcube_openpi_inputs.py`
  and
  `HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD" .venv/bin/python scripts/check_stackcube_openpi_loader.py`.
- Passed checkpoint header check:
  `PYTHONPATH="$PWD" .venv/bin/python scripts/check_stackcube_sft_readiness.py --checkpoint checkpoints/torch/pi05_stackcube_panda_base`.
- Passed focused regression tests:
  `HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD" .venv/bin/pytest -q tests/unit_tests/test_real_env.py -k 'nvidia_device_count_falls_back or nvidia_model_falls_back'`
  — 2 passed, 234 deselected.
- Passed `ruff check` and `ruff format --check` on the two changed source/test
  files, plus `git diff --check`.
- Passed `python -m py_compile` on the local single-step driver and ZIP integrity
  check on the saved `.pt` artifact.
- Failed standard-launcher attempts were capped at one step and saved in
  `sft_one_step.log`, `sft_one_step_isolated_ray.log`,
  `sft_one_step_nvml_fallback.log`, `sft_one_step_gpu1_rank1.log`, and
  `sft_one_step_nccl_p2p_disabled.log`. The first showed the shared Ray version
  mismatch; the second showed zero detected accelerators; the third showed a
  rank-0/physical-GPU mismatch and was stopped before model loading. The last
  two correctly placed on rank 1, loaded the model, then showed the NCCL/NVML
  version error before a forward pass.
- The first three launcher invocations used the original config path, model
  override, `actor.global_batch_size=1`, `runner.max_steps=1`,
  `runner.save_interval=1`, and the project-local log path:

  ```bash
  REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" CUDA_VISIBLE_DEVICES=1 HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD:${PYTHONPATH:-}" timeout 900s .venv/bin/python examples/sft/train_vla_sft.py --config-path "$PWD/examples/sft/config" --config-name stackcube_panda_sft_openpi_pi05 actor.model.model_path="$PWD/checkpoints/torch/pi05_stackcube_panda_base" actor.global_batch_size=1 runner.max_steps=1 runner.save_interval=1 runner.logger.log_path="$PWD/outputs/stackcube_recap/milestone3/results" 2>&1 | tee outputs/stackcube_recap/milestone3/logs/sft_one_step.log
  ```

  ```bash
  REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=1 HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD:${PYTHONPATH:-}" timeout 900s .venv/bin/python examples/sft/train_vla_sft.py --config-path "$PWD/examples/sft/config" --config-name stackcube_panda_sft_openpi_pi05 actor.model.model_path="$PWD/checkpoints/torch/pi05_stackcube_panda_base" actor.global_batch_size=1 runner.max_steps=1 runner.save_interval=1 runner.logger.log_path="$PWD/outputs/stackcube_recap/milestone3/results" 2>&1 | tee outputs/stackcube_recap/milestone3/logs/sft_one_step_isolated_ray.log
  ```

  ```bash
  REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=1 HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD:${PYTHONPATH:-}" timeout 900s .venv/bin/python examples/sft/train_vla_sft.py --config-path "$PWD/examples/sft/config" --config-name stackcube_panda_sft_openpi_pi05 actor.model.model_path="$PWD/checkpoints/torch/pi05_stackcube_panda_base" actor.global_batch_size=1 runner.max_steps=1 runner.save_interval=1 runner.logger.log_path="$PWD/outputs/stackcube_recap/milestone3/results" 2>&1 | tee outputs/stackcube_recap/milestone3/logs/sft_one_step_nvml_fallback.log
  ```

- The final two launcher invocations used a temporary copy of the same YAML,
  narrowed to `cluster.component_placement.actor: "1-1"`, with all GPUs visible
  to the scheduler so it could set worker `CUDA_VISIBLE_DEVICES=1`:

  ```bash
  set -o pipefail && env REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD:${PYTHONPATH:-}" timeout 900s .venv/bin/python examples/sft/train_vla_sft.py --config-path "$PWD/outputs/stackcube_recap/milestone3/config" --config-name stackcube_panda_sft_openpi_pi05 actor.model.model_path="$PWD/checkpoints/torch/pi05_stackcube_panda_base" actor.global_batch_size=1 runner.max_steps=1 runner.save_interval=1 runner.logger.log_path="$PWD/outputs/stackcube_recap/milestone3/results" 2>&1 | tee outputs/stackcube_recap/milestone3/logs/sft_one_step_gpu1_rank1.log
  ```

  ```bash
  set -o pipefail && env REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD:${PYTHONPATH:-}" timeout 900s .venv/bin/python examples/sft/train_vla_sft.py --config-path "$PWD/outputs/stackcube_recap/milestone3/config" --config-name stackcube_panda_sft_openpi_pi05 actor.model.model_path="$PWD/checkpoints/torch/pi05_stackcube_panda_base" actor.global_batch_size=1 runner.max_steps=1 runner.save_interval=1 runner.logger.log_path="$PWD/outputs/stackcube_recap/milestone3/results" 2>&1 | tee outputs/stackcube_recap/milestone3/logs/sft_one_step_nccl_p2p_disabled.log
  ```

- Successful supervised step command:

  ```bash
  env REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" \
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
    HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets \
    PYTHONPATH="$PWD:${PYTHONPATH:-}" timeout 900s \
    .venv/bin/python outputs/stackcube_recap/milestone3/run_openpi_single_step.py
  ```

- SFT attempt logs and the post-run GPU inventory are under
  `outputs/stackcube_recap/milestone3/logs/`. The converter emits no stdout; its
  three log files are empty, while the converter exit code, output files, and
  readiness checks establish completion.
- An open-PR overlap search was attempted with `gh pr list`, but GitHub returned
  HTTP 401 because the CLI is unauthenticated. No overlap conclusion can be
  made from that search.

## Remaining limitations and next milestone

- The regular Ray/FSDP launcher still cannot initialize NCCL on this host while
  NVML reports a driver/library version mismatch. The completed one-step
  objective used the same RLinf model, loader, supervised loss, and AdamW
  builder without the FSDP/Ray wrapper. Full SFT remains unrun.
- The result demonstrates one optimization update only. It does not demonstrate
  convergence or StackCube task success.
- The next research milestone should first establish a repeatable FSDP launcher
  run after the host's NVML/NCCL compatibility is resolved, then evaluate an
  SFT baseline. No full training or Milestone 4 work was started.

## Milestone 3.1 — Ray/FSDP and NCCL readiness (2026-10-21)

**Status: PASSED.** The root cause was a host NVML library/driver mismatch.
Kernel driver and `nvidia-smi` reported 590.48.01, while the active
`libnvidia-ml.so.1` resolved to 610.43.02 (`libnvidia-compute-610` package
610.43.02-0ubuntu0.24.04.1). A matching `libnvidia-ml.so.590.48.01` file was
already present. CUDA worked without adjustment; NVML failed in the host
process and Ray workers, so this was not a sandbox device restriction. The
inherited `LD_LIBRARY_PATH=/opt/slurm/24.11.0/lib` did not contain NVML; the
loader selected the 610 library from the system linker cache. No system
configuration, package, driver, or symlink was changed.

A process-scoped workaround made a private `/tmp/rlinf_m31_nvml590/libnvidia-ml.so.1`
symlink to the existing 590.48.01 library and prepended that directory to
`LD_LIBRARY_PATH`. Under that environment, Ray worker NVML reported driver
590.48.01 and NVML 13.590.48.01. The Ray worker saw
`CUDA_VISIBLE_DEVICES=1`, one CUDA device named RTX PRO 6000 Blackwell Max-Q,
initialized NCCL, and all-reduced `[3.0]` to `3.0`. NCCL diagnostics reported
runtime 2.32.3+cuda12.9; `torch.cuda.nccl.version()` returned `(2, 28, 9)`.
The log also reported unavailable optional IB/GDRCopy plugins, but selected
Socket and completed the collective. GPU 1 was idle after the test.

With the same scoped library path, the standard `examples/sft/train_vla_sft.py`
launcher initialized Ray and FSDP on hardware rank 1 (physical GPU 1), loaded
the 3.35B pretrained Pi0.5 model and existing StackCube normalization, and
completed exactly one step from the existing dataset. TensorBoard recorded
loss `0.046070437878370285`, grad norm `0.805865466594696`, learning rate
`1.249999968422344e-05`, and step time 34.70 s. The runner wrote
`outputs/stackcube_recap/milestone3/results/stackcube_panda_pi05_sft/checkpoints/global_step_1/`;
the saved full weights are 6,946,527,306 bytes and the distributed checkpoint
data file is 20,843,201,397 bytes. No further training ran. Post-run GPU 1 was
0% utilized, had 97,235 MiB free, and had no compute process.

### Milestone 3.1 commands and validation

- Host diagnostic output is in
  `outputs/stackcube_recap/milestone3/logs/m31_host_nvml_cuda_preflight.log`.
  It recorded driver 590.48.01, CUDA 13.0 available, and NVML mismatch against
  active library 610.43.02.
- Matching-library probe used:
  `LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c "import torch,
  ray._private.thirdparty.pynvml as p; print(torch.cuda.is_available(),
  torch.cuda.get_device_name(0)); p.nvmlInit(); print(p.nvmlSystemGetDriverVersion(),
  p.nvmlSystemGetNVMLVersion()); p.nvmlShutdown()"`. It passed and reported
  NVML 13.590.48.01.
- Bounded Ray/NCCL test:
  `LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib
  CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=INFO timeout 300s .venv/bin/python
  outputs/stackcube_recap/milestone3/run_m31_ray_nccl_probe.py`.
  It passed in a fresh local Ray instance, which was shut down afterward.
  Logs: `m31_ray_worker_nccl_probe.log` and
  `m31_ray_worker_nccl_probe_retry.log`.
- Standard launcher command (one actor on rank 1, one batch, one step):

  ```bash
  set -o pipefail && env LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=WARN PYTHONUNBUFFERED=1 HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD:${PYTHONPATH:-}" timeout 900s .venv/bin/python examples/sft/train_vla_sft.py --config-path "$PWD/outputs/stackcube_recap/milestone3/config" --config-name stackcube_panda_sft_openpi_pi05 actor.model.model_path="$PWD/checkpoints/torch/pi05_stackcube_panda_base" actor.global_batch_size=1 runner.max_steps=1 runner.save_interval=1 runner.logger.log_path="$PWD/outputs/stackcube_recap/milestone3/results" 2>&1 | tee outputs/stackcube_recap/milestone3/logs/m31_sft_standard_launcher_one_step.log
  ```

- Launcher result: exit code 0, global step 1/1, finite loss and gradient norm,
  checkpoint saved. Focused tests:
  `HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD"
  .venv/bin/pytest -q tests/unit_tests/test_real_env.py -k
  'nvidia_device_count_falls_back or nvidia_model_falls_back'` — 2 passed,
  234 deselected in 2.58 s.
- Post-run GPU inventory is in `m31_gpu_postrun.log`; GPU 1 was idle and its
  compute-process list was empty.
- No source files changed in Milestone 3.1. The project-local fallback and its
  tests remain in commit `b140d271a6d58f9f31894f857f4ac5e568f7bc25`. The
  Milestone 3 reports are in `a4faddf1`; environment-report commit is
  `6a526f47`. No commit was pushed.

The workaround is limited to the process tree launched with the private
library path. The system-wide NVML package/link mismatch remains and requires
an administrator-managed correction for a persistent host fix. The regular
one-step FSDP path is verified with the scoped workaround. No full SFT,
rollout, RECAP, or Milestone 4 work was started.

## Milestone 3.2 — Bounded SFT training validation (2026-10-21)

**Status: PASSED for bounded execution and checkpoint resume.** GPU 1 was idle
at preflight (97,235 MiB free, 0% utilization, no compute process). The
standard RLinf SFT launcher resumed the step-1 FSDP checkpoint, ran eight
updates through global step 9, saved step 9, resumed that checkpoint, ran one
more update, and saved global step 10. This is nine optimizer updates in
Milestone 3.2 and ten cumulative standard-launcher updates including the
Milestone 3.1 step. No full SFT was started.

Both invocations used the M3.1 process-scoped NVML path
`/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib`, set `CUDA_VISIBLE_DEVICES=0,1,2,3`
for RLinf hardware enumeration, and assigned the sole actor to physical
hardware rank 1 (`visible_accelerators=['1']`). Existing dataset, pretrained
checkpoint, action layout, and normalization were unchanged. Logs:
`m32_sft_resume_step1_to9.log` and `m32_sft_resume_step9_to10.log`.

TensorBoard contained ten finite loss/grad-norm samples, including the prior
step-1 update and nine M3.2 updates. For the complete history and exact values,
see `docs/milestones/milestone_3_2_report.md`. M3.2 loss ranged from
`0.00737754` to `0.17289411`; gradient norms ranged from `0.366478` to
`5.640074`. The YAML's existing norm clipping is 1.0. No non-finite warning
was found. These results support numerical execution stability only; they do
not establish convergence or task success.

Saved FSDP checkpoints:
- `outputs/stackcube_recap/milestone3/results/stackcube_panda_pi05_sft/checkpoints/global_step_9/`
- `outputs/stackcube_recap/milestone3/results/stackcube_panda_pi05_sft/checkpoints/global_step_10/`

Each has a nonempty DCP shard, metadata, and full weights. DCP metadata
contains model, optimizer, LR scheduler, RNG, and FSDP version state. The
step-9 checkpoint was successfully resumed for the step-10 update. GPU memory
sampling at 1 s intervals reported peak used memory 41,642 MiB (of 97,887
MiB), minimum free memory 55,609 MiB, and maximum utilization 76%; post-run
GPU 1 had 97,235 MiB free and no compute process. Disk free space after both
checkpoints was 166 GiB.

The saved actor directories have no `data.pt` iterator sidecar. RLinf's VLA
worker saves that state only for a `StatefulDataLoader`; the OpenPI loader
used here did not write one. Thus resume restored model, optimizer, scheduler,
RNG, and runner step, but exact data cursor/sample-order continuation was not
verified and the loader may restart after resumption.

No tracked source or configuration changes were needed. No tests were run
because no implementation code changed. The run logs, per-second memory CSVs,
checkpoint data, and TensorBoard events are ignored local experiment artifacts.
The report and initial progress update were committed locally as
`f46af797add1c3436458407fe0f44f902f42d7b2`; no commit was pushed.

## Milestone 4 — Full StackCube π₀.₅ SFT (2026-10-21)

**Status: PASSED for approved training execution, numerical stability, and
inference readiness.** This does not claim policy performance, convergence, or
StackCube success. Milestone 5 remains pending approval.

- The standard `examples/sft/train_vla_sft.py` Ray/FSDP launcher completed the
  approved 4 epochs and exactly 1,084 optimizer steps on physical GPU 1. The
  training loop took 51:09; launcher startup-to-idle samples bound the total
  duration to about 52 minutes. No interruption or resume was needed.
- Dataset and normalization remained unchanged: 10 demos/1,084 frames, 18D
  Panda state, 7D `pd_ee_delta_pose` action, two RGB cameras, and norm file
  SHA-256 `9740881e6e1ff87773a0247397826ace1d2935a79e668105ec3eee45170456c7`.
  The official OpenPI loader passed with action `(1,10,32)`, state `(1,32)`,
  two 224×224 image streams, and 200 prompt tokens.
- The run was initialized fresh from the official pretrained Orbax source
  `gs://openpi-assets/checkpoints/pi05_base/params` converted to
  `checkpoints/torch/pi05_stackcube_panda_base/model.safetensors`. M3.2's
  step-10 checkpoint lacked a data cursor, so this was not an exact resume.
- Exact approved settings: effective batch 4; micro-batch 1; accumulation 4;
  seed 42; AdamW LR `2.5e-5`; 2 warmup steps; cosine schedule; gradient clip
  1.0; periodic saves at steps 542 and 1,084. Dedicated config and resolved
  snapshot are under `outputs/stackcube_recap/full_sft/config/`.
- All 1,084 logged loss/gradient/LR readings were finite. Loss first/final was
  0.141/0.0119, min/max/mean 0.00222/0.141/0.022706. Gradient norm
  first/final was 1.66/0.493, min/max/mean 0.0942/3.51/0.563395. Logged LR
  ranged from `6.25e-8` to `2.5e-5` (mean `1.25318e-5`). Full per-step
  history is in `outputs/stackcube_recap/full_sft/logs/training.log`.
- Checkpoints at
  `outputs/stackcube_recap/full_sft/results/stackcube_panda_pi05_full_sft/checkpoints/global_step_542/`
  and `global_step_1084/` each contain DCP metadata/shard and 6,946,527,306-byte
  consolidated weights. The final weights loaded with the RLinf OpenPI eval
  factory as the expected 3.35B Pi0.5 model. `action_out_proj.weight` differs
  from the official base (L2 delta 0.09267354).
- One call through `Pi0Eval.predict_action_batch` with a real dataset sample
  loaded the established normalization path and returned finite actions
  shaped `(1,10,7)` from two `(1,128,128,3)` images and state `(1,18)`. This
  was an inference-readiness check only; no action was executed in ManiSkill.
- GPU 1 peak sampled usage was 36,578 MiB, peak utilization 69%, and minimum
  free memory 60,673 MiB. Inference peak allocated/reserved memory was
  6,808.2/6,968.0 MiB. Post-run GPU was idle with 97,235 MiB free and no
  compute process. System RAM available was 472 GiB and disk free was 114 GiB.
- The existing FSDP config had AMP disabled; a PyTorch process-group barrier
  device-context warning was emitted. Neither affected the run. No validation
  loss was available because the current embodied SFT evaluation path does not
  implement validation, and no trajectory-held-out split was supported.
- Persistent run logs, memory sampling, and inference evidence are under
  `outputs/stackcube_recap/full_sft/logs/`. The completed report is
  `docs/milestones/milestone_4_report.md`.
- Tracked changes are the progress log and Milestone 4 report. The start commit
  is `1e096ce6`; no Milestone 4 commit was made because a sandbox attempt to
  write `.git/index` was blocked by read-only repository metadata. No
  permissions were changed and no commit was pushed.

The next research milestone is baseline ManiSkill evaluation (Milestone 5),
which is not started and requires separate approval.
