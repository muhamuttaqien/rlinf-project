# Codex Experiment Progress — StackCube π₀.₅ + RECAP

Updated: 2026-10-21 (Asia/Tokyo)  
Branch: `branch-openpi-recap-codex`
Current milestone: Research Milestone 6 — **PASSED for the bounded rollout pilot and dataset validation; not fully RECAP training-ready**

## Milestone 6 — Rollout collection and dataset validation (2026-10-21)

**Status: PASSED for the 10-episode pilot.** Collected 10 closed-loop SFT
episodes at K=5 and a 150-step horizon on physical GPU 1. The dataset contains
1,500 transitions, 0 native StackCube successes, 2 grasp-heuristic episodes,
1 lift-heuristic episode, 1 transport-heuristic episode, and no native
cube-on-cube placements. All 10 episodes timed out. Heuristic stages are not
task success.

- The bounded collector is `scripts/collect_stackcube_rollouts.py`; the
  dataset/trace validator is `scripts/validate_stackcube_rollouts.py`.
- Dataset and per-step raw trace artifacts are under
  `outputs/stackcube_recap/milestone6_rollouts/pilot/`. The official RECAP
  returns processor succeeded on a separate dataset copy at
  `outputs/stackcube_recap/milestone6_rollouts/returns_processed/`.
- Validation matched all 1,500 saved state/action/native-reward rows to raw
  traces and decoded 60 sampled camera frames with PyAV. LeRobot's configured
  torchvision decoder raised an unavailable `VideoReader` API error; direct
  PyAV decoding verified the stored videos.
- Pilot dataset size: 6.05 MB; full pilot artifacts: 9.11 MB. Mean episode
  runtime 8.78 s, mean policy inference 180.8 ms/call, peak GPU allocation
  6,808.2 MiB and reservation 6,968.0 MiB.
- RECAP return labeling requires terminal `is_success` and creates its own
  dynamic reward labels. It does not use the stored native ManiSkill rewards.
  Current advantage/value data mappings have no StackCube robot type, and the
  existing Franka EE transform is incompatible with this 18D qpos+qvel state
  and 7D delta-pose action. The pilot is therefore not yet suitable for value
  or advantage training.
- A proposed, not authorized or started, first scaled batch is 100 episodes
  with seeds 4000–4099 and the same protocol. Estimated episode runtime is
  about 15 minutes and dataset storage about 61 MB (about 91 MB including
  pilot-like trace/metadata). Low observed success makes success-balanced data
  unlikely without a separately authorized collection strategy.
- No model weights, prior outputs, system settings, or earlier experiment
  artifacts were modified. No RECAP optimization or scaled collection started.
- Validated source and report commit: `d0700826`
  (`feat(stackcube): collect validated recap rollouts`); no push was made.

See `docs/milestones/milestone_6_report.md` for the algorithm audit, detailed
schema, actual validation and return-label statistics, GPU/runtime results,
limitations, and next-step proposal.

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

Milestone 5 was subsequently authorized and is recorded below.

## Milestone 5 — ManiSkill StackCube baseline evaluation (2026-10-21)

**Status: PASSED for execution of the paired evaluation.** Neither policy
succeeded on the 10 tested seeds; that is the measured result, not a pipeline
failure. No policy improvement in success rate was observed.

- Confirmed branch `branch-openpi-recap`, HEAD `eeb5f39d`, and a clean starting
  worktree. Read `AGENTS.md`, `docs/STACKCUBE_RECAP.md`, the M4 report, progress
  log, SFT YAML, and relevant ManiSkill/OpenPI model code.
- Verified the official base checkpoint at
  `checkpoints/torch/pi05_stackcube_panda_base/model.safetensors` and final SFT
  weights at `outputs/stackcube_recap/full_sft/results/stackcube_panda_pi05_full_sft/checkpoints/global_step_1084/actor/model_state_dict/full_weights.pt`.
  M4 already verified final model loading through the RLinf OpenPI evaluation
  factory.
- Host preflight selected physical GPU 1: 97,235 MiB free of 97,887 MiB, 0%
  utilization, and no listed compute processes. System memory had 473 GiB
  available and disk had 114 GiB free. CUDA 13.0/PyTorch 2.11.0+cu130
  initialized with `CUDA_VISIBLE_DEVICES=1` and the process-scoped M3.1 NVML
  path `/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib`. The sandbox cannot see
  NVIDIA devices; both failed sandbox output and successful host preflight are
  retained in `outputs/stackcube_recap/full_sft/logs/m5_preflight_sandbox.log`.
- The generic RLinf ManiSkill eval wrapper does not directly produce the
  training state (18D qpos+qvel) and two-camera input mapping. A task-specific
  evaluator under `outputs/stackcube_recap/milestone5_eval/` uses RLinf's
  existing OpenPI `get_model`/`Pi0Eval` and transforms with ManiSkill's native
  `StackCube-v1`, `panda_wristcam`, `pd_ee_delta_pose`, and CPU PhysX. CPU
  simulation matches the expert replay backend and keeps only model inference
  on GPU 1. It reuses the exact M4 normalization file for both checkpoints.
- This task has a native 50-step limit. The evaluator used matched seeds 42–51,
  10-step policy chunks, and recorded success, termination, return, grasp/lift/
  transport/placement stage flags, failure stage, and both RGB camera streams.
  It completed 1,000 environment transitions and 100 policy calls.
- The standalone evaluator's `--help` and `py_compile` checks passed after
  setting the repository `PYTHONPATH`; the first help invocation omitted that
  path and failed with `ModuleNotFoundError: rlinf` before running evaluation.
- Outcome: pretrained and SFT were each 0/10 successes; all 20 episodes reached
  the 50-step limit and were categorized `no_grasp`. Neither policy recorded a
  lift, transport, or placement event. Mean episode return was 2.003 for
  pretrained and 4.184 for SFT; this reward difference did not translate into
  task success. The paired success-rate delta was 0/10.
- Saved camera recordings are in
  `outputs/stackcube_recap/milestone5_eval/videos/` (20 compressed NPZ files,
  main and wrist streams). Reviewed sample frames show the SFT robot moved
  toward the cubes on seed 50 but did not grasp; the pretrained seed-42 sample
  also did not reach a grasp.
- Policy peak CUDA allocated/reserved memory was 6,808.2/6,968.0 MiB for
  pretrained and 6,809.4/6,958.0 MiB for SFT. Post-run GPU 1 had 97,235 MiB
  free, 0% utilization, and no compute process. System memory had 473 GiB
  available and disk had 114 GiB free.
- Evaluation outputs are under `outputs/stackcube_recap/milestone5_eval/`;
  the report is `docs/milestones/milestone_5_report.md`. No autonomous rollout
  collection or RECAP work was started. Further work is paused for user review.

## Milestone 5.1 — StackCube policy failure diagnosis (2026-10-21)

**Status: PASSED for the requested audit.** No reproducible policy/evaluator
implementation error justified a code change. The verified diagnosis is a
large expert/evaluation horizon mismatch; the original M5 rule and all M5
results remain unchanged.

- Audited all 20 M5 episode records and contact sheets from every stored video
  (base and wrist camera, steps 0/10/20/30/40/50). Both checkpoints remained
  0/10 success, 0/10 grasp, and `no_grasp` on every episode. Each ended at
  50 steps with `truncated=true`, `terminated=false`, matching ManiSkill's
  registered `StackCube-v1` max episode length of 50.
- The ten source expert trajectories contain 94–120 actions (mean 108.4),
  all labeled terminal-success. ManiSkill replay from recorded initial
  environment states replayed the first three successfully (3/3); the replay
  continues after Gym's 50-step truncation. Therefore the dataset expert
  trajectories exceed the task/evaluation horizon. This is a verified
  comparison limitation, not evidence that increasing the horizon alone
  would make either policy succeed.
- Confirmed evaluator environment and robot: `StackCube-v1`,
  `panda_wristcam`, RGB observations, `pd_ee_delta_pose`, one environment,
  CPU PhysX. Termination, truncation, grasp, and task-success reporting agree
  with the native task. The policy executes its 10-action chunk sequentially
  and requeries after each ten actions, matching the trained horizon and
  RLinf's action-chunk contract.
- Raw HDF5 and LeRobot data checks passed. On five sampled rows, qpos+qvel
  state and 7D actions matched exactly (maximum absolute difference 0); both
  128×128 RGB features decoded under the expected base/hand keys. The adapter
  maps base and wrist cameras to their intended model slots, preserves the
  18D qpos-then-qvel state, and uses the same StackCube normalization for
  training and inference.
- Expert gripper targets include both normalized commands (`-1`: close,
  `+1`: open); conversion copies the command unchanged and Panda controller
  polarity matches. No gripper sign issue was found.
- Inference-only comparison used the official pretrained and M4 SFT
  checkpoints at expert states (episodes 0 and 1, frames 0/20/40). All 12
  output chunks were finite `(1,10,7)` actions. Mean chunk MAE was 0.034 for
  SFT and 0.274 for pretrained; this supports correct teacher-forced mapping
  and unnormalization, but does not establish closed-loop performance.
- Checks: `scripts/validate_stackcube_replay.py` passed 10/10;
  `scripts/validate_stackcube_lerobot.py` passed 10 episodes / 1,084 frames
  using its default TorchCodec decoder; expert replay passed 3/3; sampled
  state/action equality passed. An explicit `--video-backend pyav` check
  failed because this environment's `torchvision.io` lacks `VideoReader`; the
  default TorchCodec decoder passed without dependency changes. A sandboxed
  replay attempt failed at renderer initialization because sandbox device
  access is unavailable; the bounded host-context replay succeeded.
- GPU 1 preflight: 97,235 MiB free of 97,887 MiB, 0% utilization, no compute
  process; 473 GiB available system RAM and 114 GiB free disk. Inference used
  `CUDA_VISIBLE_DEVICES=1` and the process-scoped M3.1 NVML path. Peak memory
  for this diagnostic was not recorded. Post-run GPU 1 was idle again with
  97,235 MiB free. Resource and inference artifacts are under
  `outputs/stackcube_recap/milestone5_eval/`.
- No tracked code, checkpoint, normalization, or evaluation rule changed.
  A longer policy horizon was not run because ManiSkill's native limit is
  50; changing it only to improve success counts would alter the benchmark
  rule. SFT action reproduction is strong at sampled expert states, while
  closed-loop distribution shift, 10-step open-loop error accumulation, and
  the ten-episode training set remain plausible, unisolated contributors.
  The saved M5 artifacts lack predicted action traces and use different reset
  seeds from the experts, so the exact first closed-loop divergence is unknown.
- Report: `docs/milestones/milestone_5_1_report.md`. Starting commit was
  `eeb5f39d`; no commit or push was made. No further rollout collection or
  RECAP work was started.

Recommended next decision: define whether future comparisons retain the
official 50-step benchmark (requiring experts within 50 steps) or use a
separately approved longer-horizon diagnostic. Any additional data collection,
training, or evaluation protocol change needs approval.

## Milestone 5.2 — Extended-horizon closed-loop evaluation (2026-10-21)

**Status: PASSED for the evaluation and behavioral diagnosis.** The separate
150-step diagnostic completed for the SFT pilot and all 20 paired seeds. Both
policies had 0/20 native StackCube successes. This result does not indicate
policy convergence or task competence.

- Read the repository instructions and M4, M5, and M5.1 documentation. Kept
  M5's 50-step configuration, results, and videos unchanged. Used the M4 final
  checkpoint and the official pretrained checkpoint with the validated
  normalization statistics.
- Used ManiSkill's supported `max_episode_steps=150` wrapper override while
  retaining `StackCubeEnv.evaluate()`, the task's termination logic, the
  Panda wrist-camera robot, `pd_ee_delta_pose`, 18D qpos+qvel state, both RGB
  streams, CPU PhysX, and 7D actions. Confirmed the live `TimeLimitWrapper`
  exposes `_max_episode_steps=150`.
- Policy observations and action predictions refresh at each 10-action chunk
  boundary; simulator observations refresh at every step. All 150 action
  values in each episode were passed unchanged to `env.step`; resulting Panda
  arm/gripper controller targets were also logged. Native success, termination,
  and truncation were checked after every transition.
- Pilot: SFT on seeds 200–202, each 150 transitions. Outcomes were
  `no_grasp`, `grasped_not_lifted`, `no_grasp` (0/3 task successes). Paired:
  seeds 42–61, 150 transitions for each pretrained and SFT policy. All 20
  pretrained initial-state signatures matched their SFT counterparts exactly.
  Pretrained was 0/20, with `no_grasp` on every seed. SFT was 0/20, with
  13 `no_grasp`, 3 `grasped_not_lifted`, 2 `lifted_not_transported`, and 2
  `transported_not_stacked` episodes. Paired success-rate difference: 0/20.
- SFT had verified simulator grasps on 7/20 episodes, lift events on 4/20,
  and transport-stage events on 2/20. Two episodes reduced red-cube/target
  XY distance to 3.3 cm and 3.9 cm but never reached native task success.
  Pretrained never grasped; its closest TCP-to-red-cube distance across all
  transitions was 11.8 cm, versus 0.5 cm for SFT. These stage and distance
  measures describe behavior, not task completion.
- In the Panda root-translation command frame, mean/median normalized
  translation-command magnitude was 0.142/0.121 for pretrained and
  0.084/0.062 for SFT. Command-to-cube direction cosine averaged -0.242 and
  +0.178; cosine was positive on 33% and 61% of transitions, respectively.
  This descriptive action comparison is consistent with SFT's closer
  approaches but does not establish success.
- Action/trace validation passed for 6,000 paired transitions: 150 sequential
  steps per episode, correct chunk-refresh boundaries, one native truncation
  at step 150 per episode, finite predicted actions, and equality of predicted
  and submitted actions. SFT produced both open and close gripper commands;
  physical gripper qpos ranged 0–0.04 m. Traces record pose, cube/target pose,
  gripper command and qpos, actions, controller joint targets, refresh flags,
  success, and termination/truncation at every step.
- Saved per-step JSONL traces, episode metrics, initial-state signatures,
  environment configs, pilot/paired logs, pre/postflight resource logs, and
  dual-camera recordings under `outputs/stackcube_recap/milestone5_2_eval/`.
  This includes 43 compressed NPZ episode recordings and four side-by-side
  representative MP4s (pretrained seed 42; SFT seeds 53, 58, and 60). The
  paired evaluation used physical GPU 1, process-scoped M3.1 NVML libraries,
  `CUDA_VISIBLE_DEVICES=1`, and PyTorch 2.11.0+cu130. Peak allocated/reserved
  memory was 6,808.2/6,968.0 MiB (pretrained) and 6,809.4/6,958.0 MiB (SFT).
  Paired run config-to-summary time was 4m00s; mean simulator episode runtime
  was 4.42s per policy. Postflight GPU 1 returned to 97,235 MiB free and 0%
  utilization with no compute processes; disk had 114 GiB free.
- `py_compile`, the evaluator's `--help` with repository `PYTHONPATH`, and
  trace/video integrity checks passed. An initial `--help` check without
  `PYTHONPATH` failed before evaluation with `ModuleNotFoundError: rlinf`; the
  corrected command passed. No evaluator semantics or model code were changed.
  Two setup guards also stopped before rollout because ManiSkill exposes the
  effective horizon through its custom `TimeLimitWrapper`, not `env.spec` or
  Gymnasium's generic wrapper type. The guard was updated to verify the live
  wrapper's `_max_episode_steps == 150`; the successful pilot log records it.
  The helper and generated data live under ignored `outputs/`; original
  checkpoints, dataset, and M5 artifacts were preserved.
- Report: `docs/milestones/milestone_5_2_report.md`. Git HEAD remains
  `eeb5f39d`; existing uncommitted M5 documentation was preserved. No commit
  or push was made. No retraining, rollout collection, or RECAP work started.

The longer horizon resolves the duration confound for this diagnostic, but
does not by itself explain all failures: pretrained remained far from the cube,
while SFT sometimes grasped and lifted but failed to stack. Closed-loop
distribution shift, grasp precision, and placement behavior remain plausible
training limitations; no single cause was isolated. Recommended next step is
to review these traces/videos and decide whether to authorize a narrowly scoped
policy/data improvement study. Do not start rollout collection or RECAP until
approved.

## Milestone 5.3 — OpenPI π₀.₅ behavioral failure analysis (2026-10-21)

**Status: PARTIAL.** Offline event reconstruction identifies concrete SFT
behavioral failures, but does not isolate a single causal mechanism. The
Milestone 5.2 result remains unchanged: paired pretrained and SFT policies
both achieved 0/20 native task success; SFT had 7 native grasp episodes, 4
episodes above the 2.5 cm lift marker, and 2 above the 3 cm transport marker.

- Audited all 20 paired SFT traces (3,000 transitions) and 10 successful
  expert HDF5 trajectories. Reconstructed per-episode grasp intervals, lift
  and transport events, TCP/cube/target distances, placement-position errors,
  gripper behavior, action chunk timing, and task flags. Generated CSV/JSON
  reconstructions, plots, and dual-camera contact sheets under
  `outputs/stackcube_recap/milestone5_3_diagnostics/` using
  `analyze_m53.py`.
- Three grasp episodes had only 3–7 total native contact steps and cube height
  increases of 2.1–3.7 mm, so did not reach the 25 mm lift diagnostic. Seeds
  53 and 59 maintained grasp after lifting but made only 7.2 mm and 18.4 mm
  targetward XY progress, below the 30 mm marker. Seed 58 stayed grasped to
  timeout and ended at 32.92 mm XY error but 10.61 mm vertical placement error.
  Seed 60 opened at step 109 while still 41.8 mm from target in XY and 14.7 mm
  off the target vertical offset; the cube returned to table height by step
  113. No SFT episode met both positional tolerances or native stack success.
- All 10 expert trajectories were successful, with 94–120 actions. Expert
  event-aligned medians were 16 steps from first close command to 25 mm lift,
  22 from lift to transport marker, and 16 from transport to success. These
  are descriptive comparisons: expert close is an action proxy, and expert
  reset states differ from SFT states. Three expert replays were already
  verified successful in the M5.1 audit.
- Trace checks found the expected 10-action chunk schedule, with a current
  simulator state every step and a new policy observation/prediction at chunk
  boundaries. The controller’s configured clipping is active; 1,968/3,000 SFT
  gripper command values exceeded normalized magnitude 1 and were subject to
  that expected clipping. This is a policy-command pattern, not a verified
  controller defect. The seed 60 early release occurs on action 8 of a chunk,
  making delayed correction plausible but untested. No evaluation, mapping,
  denormalization, or simulator defect was identified.
- Exact contact geometry, arm tracking error, cube static/velocity state, and
  causal attribution remain unavailable from the saved traces. No extra
  simulator/GPU experiment, source change, or checkpoint modification was
  made. The analysis script completed successfully; no project tests were
  applicable. Report: `docs/milestones/milestone_5_3_report.md`.
- Git remains on `branch-openpi-recap` at `eeb5f39d`, ahead 7; pre-existing
  dirty milestone documentation was preserved. No commit or push was made.
  Recommended next step, pending approval: instrument contact, actual joint
  state, and cube velocity/static data in a small fixed-checkpoint replay, then
  compare 1-step and 10-step policy refresh on matched initial states. Do not
  begin rollout collection, retraining, or RECAP without approval.

## Milestone 5.4 — Controlled action chunking experiment (2026-10-21)

**Status: PASSED for the controlled evaluation.** The 80 matched episodes
completed; K=5 recorded one native success, and K=10, K=2, and K=1 recorded
none. The single success is encouraging but statistically inconclusive.

- Implemented the isolated M5.4 evaluator at
  `outputs/stackcube_recap/milestone5_4_eval/evaluate_chunking.py`. It keeps
  model output `(1,10,7)` and executes only the first K actions before fresh
  inference. The M5.2 evaluator and outputs were not changed. Four regression
  tests, Python compilation, trace validation, analysis, and `git diff --check`
  passed.
- The three-seed pilot (200–202) completed for K=10/5/2/1 with 0/3 task success
  for every K. All pilot reset signatures matched across K and the prior pilot;
  K=10 reproduced prior pilot stage outcomes. Inference calls per 150-step
  episode were 15/30/75/150. Pilot mean runtimes were 4.59/7.27/14.78/28.31 s.
- The full paired evaluation used seeds 42–61. K=10 reused the protocol-
  compatible M5.2 SFT baseline after exact reset-signature validation. Results:
  K=10: 0/20 success, 7 grasp, 4 lift, 2 transport; K=5: 1/20, 7 grasp, 4
  lift, 2 transport; K=2: 0/20, 6 grasp, 3 lift, 3 transport; K=1: 0/20, 7
  grasp, 4 lift, 3 transport. K=2 recorded one transient cube-on-cube flag but
  no task success. K=5 seed 60 succeeded natively at step 129.
- The original K=5 seed-60 run executed one queued action after termination.
  The guard was moved inside the per-action loop; a same-seed replay matched
  the first 129 actions and robot/cube states exactly and stopped on native
  success/termination at step 129. The corrected trace is used in the final
  analysis; the original trace is preserved. No other sweep episode had an
  early success or termination.
- Per-episode mean runtime was 4.41/7.13/14.58/29.09 s for K=10/5/2/1.
  Measured mean inference latency was 180.4/170.9/181.4 ms for K=5/2/1; the
  K=10 pilot measured 189.2 ms. Peak allocated/reserved CUDA memory was about
  6,808–6,809/6,958–6,968 MiB. The host monitor sampled up to 8,014 MiB GPU
  use and 41% utilization. Postflight GPU 1 had 97,235 MiB free, 0%
  utilization, and no compute processes; disk had about 114 GiB free.
- The paired success difference versus K=10 was one K=5-only success and no
  K=10-only successes; exact two-sided paired p=1.0. Confidence intervals are
  wide. Target distances and stage outcomes were otherwise mixed, and
  inference-call counts grew from 15/episode at K=10 to 29.8, 75, and 150 at
  K=5/2/1. These results do not establish K=5 as generally superior.
- All configs, traces, videos, resource logs, analysis, paired metrics, and
  validation results are under `outputs/stackcube_recap/milestone5_4_eval/`.
  Report: `docs/milestones/milestone_5_4_report.md`. No training, autonomous
  rollout collection, or RECAP was started.
- Git remains on `branch-openpi-recap` at `eeb5f39d`, ahead 7. The M5.4
  evaluator/tests are in ignored experiment outputs; no tracked source code
  changed. Pre-existing progress and milestone documentation changes were
  preserved. No commit or push was made.

Recommended next step, pending approval: repeat K=5 on a larger, predeclared
held-out seed set, with common inference-noise control if available. Do not
start additional evaluation, training, rollout collection, or RECAP without
approval.

## Milestone 5.5 — reproducibility and generalization (complete)

- Start state: branch `branch-openpi-recap`, HEAD `eeb5f39d`; pre-existing
  progress edits and untracked M5 reports preserved. No source checkpoint or
  dataset files changed.
- Verified final SFT `full_weights.pt` SHA-256
  `1db71780bcad741053517000652f01bb8f8383e43f1b2dacb4da5ab9f02c09c3` and
  normalization SHA-256
  `9740881e6e1ff87773a0247397826ace1d2935a79e668105ec3eee45170456c7`.
- The corrected M5.4 seed-60 reference has native success/termination at step
  129, 26 inference calls, K=5, and 150-step Gym TimeLimit. Physical GPU 1 was
  idle with 97,235 MiB free and 0% utilization before evaluation; host CUDA
  preflight passed. Sandbox CUDA access remains unavailable.
- M5.5 copied the existing instrumented evaluator into its isolated output
  directory, supporting exact-seed replay and 30-episode generalization runs.
  The native success evaluator, policy preprocessing, controller, checkpoint,
  and K=5/K=10 chunk semantics are unchanged. Seed manifest
  `outputs/stackcube_recap/milestone5_5_eval/generalization_seed_manifest.json`
  predeclared seeds 1000–1029 after finding no collisions in existing JSON
  episode artifacts.
- Five K=5 seed-60 replays have completed: all 5 succeeded natively at step
  129, each with 26 inference calls. Replay outputs are under
  `outputs/stackcube_recap/milestone5_5_eval/replays/`; mean episode runtime is
  about 6.51 s and reported peak allocated memory is 6,808.2 MiB. All five
  control traces exactly match the corrected M5.4 reference, excluding latency
  fields.
- K=5 generalization on seeds 1000–1029 completed 0/30 native successes;
  heuristic stage categories were 26 no-grasp, 2 grasped-not-lifted, and 2
  transported-not-stacked. Mean episode runtime was 6.99 s, peak allocated
  memory 6,808.2 MiB. Outputs include per-step traces and videos.
- K=10 completed on the exact same seeds. All 30 K=5/K=10 reset signatures
  match. Both schedules achieved 0/30 native success; exact two-sided McNemar
  had 0 discordant pairs and p=1.0 (uninformative for equivalence). Exact 95%
  binomial intervals were 0.0–11.57% for both.
- Diagnostic counters were K=5 contact/grasp 4, lift-height 2,
  target-distance transport 3, cube-on-cube 0; K=10 6, 3, 5, and 1. These
  markers can occur out of order and are not validated sequential subtasks.
  K=10 seed 1000 had a transient cube-on-cube flag at step 131, opened at 130,
  and did not reach native success. K=5 averaged 30 calls and 6.99 s/episode;
  K=10 averaged 15 calls and 4.32 s/episode. Both reported 6,808.2 MiB peak
  CUDA allocation.
- Physical GPU 1 ended at 0% utilization with 16 MiB reported used and 97,235
  MiB free; 113 GiB remained on disk. Paired traces contain 4,500 simulator
  transitions per K, with videos and initial state signatures saved for all
  episodes.
- Offline analysis validated five reference-identical seed-60 replay traces
  and all paired reset states. The evaluator and analysis scripts compiled,
  the paired-state checks passed, and `git diff --check` passed. Full report:
  `docs/milestones/milestone_5_5_report.md`.
- All outputs are under `outputs/stackcube_recap/milestone5_5_eval/`. No
  tracked source code changed, no commit or push was made, and no training or
  RECAP was started. Git remains at `eeb5f39d`; pre-existing edits and reports
  were preserved.
- Recommended next step: review the transient K=10 placement and K-dependent
  contact/transport traces before proposing a bounded policy or data
  intervention. M5.5 is exploratory because K=5 was selected from the earlier
  chunking sweep; the new data does not establish schedule superiority.
- An initial replay command exited before model initialization because
  `PYTHONPATH` was missing; the corrected retry sets `PYTHONPATH=$PWD`. No GPU
  work or episode artifacts were produced by that failed attempt.

## Milestone 6.1 — StackCube RECAP value/advantage integration (partial)

- Audited the existing RECAP path: it constructs task-level return labels,
  trains a separate categorical value critic, then uses N-step rewards and
  `V(o_t), V(o_{t+N})` for advantage labels. The current critic consumes
  camera images and task text; it does not consume Panda state/actions. A
  truncated episode is treated as an end boundary with zero next value.
- Added a StackCube-specific mapping for the value-training dataset,
  advantage inference, and existing value input transforms. The two observed
  cameras map to base and left-wrist inputs; 18D state and 7D action are
  preserved then padded to the existing 32D model interface. Existing
  normalization semantics are unchanged.
- CPU integration smoke against the actual Milestone 6 dataset passed:
  1,500 transitions across 10 aligned 150-step episodes; native outcomes are
  0 successes and 10 time-limit truncations. RECAP failure labels are -1 for
  steps 0–148 and -300 at step 149; at gamma=1, returns are -449 initially
  and -300 at the final step. Actual frame-0 images decoded from both pilot
  videos and preprocess to finite 224×224 tensors; the absent right-wrist
  input is masked. State/action values come from the actual episode parquet.
- Focused test file: 6 passed (17 upstream deprecation warnings). Ruff and
  `git diff --check` passed. A numeric fixture validates the official
  advantage arithmetic and terminal boundary, but its supplied values are
  test inputs, not critic predictions. No dataset advantages were generated.
- No compatible local SigLIP2/Gemma3 assets or trained RECAP value checkpoint
  were found. The π₀.₅ policy checkpoint is not a value checkpoint. No critic
  was initialized; no GPU was used. Since the value model is unavailable and
  does not consume proprioception, full value/advantage integration and RECAP
  readiness remain unverified. Status: **PARTIAL**.
- Report: `docs/milestones/milestone_6_1_report.md`. Smoke summary:
  `outputs/stackcube_recap/milestone6_1_recap/data_integration_smoke.json`.
  Source changes are the StackCube mapping, three registrations, and focused
  tests. Existing pilot data, SFT checkpoints, and prior evaluation artifacts
  were preserved. No commit or push was made.
- Recommended next milestone: approve a value-model preparation design,
  including backbone asset provenance, whether an image/text critic is
  scientifically suitable or proprioception is required, and how successful
  expert and failed rollout episodes will be labeled and split. Then validate
  bounded value SFT before computing policy advantages. Do not begin RECAP
  policy optimization or scaled rollout collection without approval.

## Milestone 6.2 — RECAP critic readiness and training strategy (partial)

- Audited the local RECAP algorithm, critic, return/advantage processing,
  value-training configuration, and CFG policy configuration against the
  π*0.6 paper and RLinf guide. The critic predicts a 201-bin return
  distribution from images and task text; local advantage labels use N-step
  rewards plus adjacent value estimates. Policy extraction uses CFG-conditioned
  supervised training.
- Confirmed that `ValueCriticModel` does not consume StackCube's preserved 18D
  Panda state or 7D actions. No local SigLIP2/Gemma3 assets or trained critic
  checkpoint were found. No random predictions, critic training, rollout
  collection, or policy update were performed.
- A bounded CPU `ValueDataset` load against the actual 1,500-transition
  processed pilot passed. Existing test file: 6 passed, 17 upstream warnings.
- Recomputed returns from the ten successful expert trajectories (1,084
  transitions): range -119 to 0. The ten failed 150-step policy trajectories
  have current configured targets -449 to -300. Under global min -449,
  expert normalized targets span about -0.265 to 0 and failures -1 to -0.668.
  These are consequences of the configured synthetic reward, not native dense
  ManiSkill rewards.
- Latest resource snapshot available: 112 GB disk free and 472 GiB RAM
  available. No GPU workload was run and no current GPU snapshot was taken.
- Report: `docs/milestones/milestone_6_2_report.md`. Worktree's pre-existing
  Milestone 6.1 edits were preserved; no new code change, commit, or push.
- Remaining decisions: image/text-only versus state-conditioned critic, reward
  and timeout semantics, data mixing/splitting, model asset provenance, and
  separate π₀.₅ CFG compatibility. Recommended next step is an approved,
  bounded critic preparation/training pilot after those decisions; no
  advantages or RECAP policy training should proceed before checkpoint
  validation.
