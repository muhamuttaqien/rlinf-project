# Research Milestone 3.1 Report — Ray/FSDP and NCCL Readiness

## Status

**PASSED.** The NVML/NCCL failure was traced to the active host NVML userspace
library being newer than the loaded NVIDIA kernel driver. A process-scoped
library path selected an already-installed NVML library that matches the
driver. A Ray worker then initialized CUDA and NVML, completed an NCCL
all-reduce, and the standard RLinf SFT launcher completed exactly one FSDP
optimizer step.

The host-wide package/link mismatch remains. No system packages, driver,
symlink, kernel, or service were changed.

## Starting state

- Repository: `/home/muttaqien/Desktop/AIST-Tokyo/gamma/rlinf-project`
- Branch: `branch-openpi-recap`
- Starting HEAD: `6a526f47` (`docs: add Codex environment inspection reports`);
  branch was ahead of origin by three commits.
- Existing Milestone 1–3 datasets, normalization, pretrained model conversion,
  single-step checkpoint, and logs were preserved.
- One GPU was selected for the tests and actual launcher step: physical GPU 1.

## Driver and library investigation

The host was inspected outside the device-restricted sandbox after the normal
sandbox lacked NVIDIA device access. On the host, CUDA worked while NVML
failed, so sandbox restrictions were not the cause of the earlier mismatch.

- `/proc/driver/nvidia/version` and `nvidia-smi`: NVIDIA driver 590.48.01.
- PyTorch: `2.11.0+cu130`, CUDA runtime 13.0. With
  `CUDA_VISIBLE_DEVICES=1`, CUDA initialized and reported one RTX PRO 6000
  Blackwell Max-Q Workstation Edition.
- System `libnvidia-ml.so.1` resolved to
  `/lib/x86_64-linux-gnu/libnvidia-ml.so.610.43.02`.
- Installed package metadata included `libnvidia-compute-610`
  `610.43.02-0ubuntu0.24.04.1`.
- A matching file, `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.590.48.01`, was
  already present. No package install or system link change was made.
- Inherited `LD_LIBRARY_PATH` was `/opt/slurm/24.11.0/lib`. It did not contain
  `libnvidia-ml`; the default dynamic linker cache selected the 610 library.
  Thus the inherited variable did not cause the mismatch, though the scoped
  override below prepends a private directory before it.
- Ray worker processes inherited the job `LD_LIBRARY_PATH`, and RLinf worker
  placement set `CUDA_VISIBLE_DEVICES=1`. The earlier failure occurred in the
  host process as well as workers, ruling out Ray-only environment corruption.

The read-only host transcript is
`outputs/stackcube_recap/milestone3/logs/m31_host_nvml_cuda_preflight.log`.
The initial CUDA/NVML comparison failed only at NVML initialization with
`NVMLError_LibRmVersionMismatch`; CUDA initialized successfully.

## Scoped remediation

A private path under `/tmp/rlinf_m31_nvml590` was created with
`libnvidia-ml.so.1` pointing to the existing 590.48.01 library. Runs prepended
that directory to `LD_LIBRARY_PATH`:

```bash
mkdir -p /tmp/rlinf_m31_nvml590
ln -sfn /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.590.48.01 \
  /tmp/rlinf_m31_nvml590/libnvidia-ml.so.1
LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib
```

With that path, NVML reported driver `590.48.01`, NVML `13.590.48.01`, and the
expected GPU model. The change applied only to launched processes and their
children. This does not repair the system-wide package/link mismatch; an
administrator would need to resolve that separately for other users or
processes.

## Ray worker and NCCL diagnostic

The bounded diagnostic ran in a fresh local Ray instance with one GPU resource
and one actor. It reported from inside the worker:

- `CUDA_VISIBLE_DEVICES=1`
- PyTorch `2.11.0+cu130`, CUDA 13.0
- CUDA available, device count 1, physical GPU 1 model as expected
- NVML driver 590.48.01, NVML 13.590.48.01
- NCCL communicator initialized and a one-rank all-reduce returned `3.0` from
  the input value `3.0`
- Ray local instance shut down after the test

NCCL's log printed runtime `2.32.3+cuda12.9`; PyTorch's
`torch.cuda.nccl.version()` returned `(2, 28, 9)`. Both values are recorded as
observed; the diagnostic operation passed. NCCL selected the Socket network
backend. Logs mentioned unavailable optional IB and GDRCopy support, without
preventing communicator initialization or the collective.

Diagnostic helper: `outputs/stackcube_recap/milestone3/run_m31_ray_nccl_probe.py`.
Logs: `m31_ray_worker_nccl_probe.log` and
`m31_ray_worker_nccl_probe_retry.log`.

The direct NVML probe command was:

```bash
LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib \
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c "import torch, ray._private.thirdparty.pynvml as p; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); p.nvmlInit(); print(p.nvmlSystemGetDriverVersion(), p.nvmlSystemGetNVMLVersion()); print(p.nvmlDeviceGetName(p.nvmlDeviceGetHandleByIndex(0))); p.nvmlShutdown()"
```

## Standard RLinf SFT launcher

The existing SFT entry point was run with the existing converted Pi0.5
checkpoint, StackCube dataset and normalization, and temporary config that
places the actor only on hardware rank 1. The launcher used four visible
resource IDs so RLinf could select rank 1 (physical GPU 1); only that GPU was
assigned to the actor. It was bounded to batch size 1 and `runner.max_steps=1`.

Command:

```bash
set -o pipefail && env LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=WARN PYTHONUNBUFFERED=1 HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD:${PYTHONPATH:-}" timeout 900s .venv/bin/python examples/sft/train_vla_sft.py --config-path "$PWD/outputs/stackcube_recap/milestone3/config" --config-name stackcube_panda_sft_openpi_pi05 actor.model.model_path="$PWD/checkpoints/torch/pi05_stackcube_panda_base" actor.global_batch_size=1 runner.max_steps=1 runner.save_interval=1 runner.logger.log_path="$PWD/outputs/stackcube_recap/milestone3/results" 2>&1 | tee outputs/stackcube_recap/milestone3/logs/m31_sft_standard_launcher_one_step.log
```

Result: exit code 0. Ray detected four accelerators and RLinf placed the
single actor on rank 1. FSDP loaded the existing Pi0.5 checkpoint (3.35B
parameters), loaded StackCube normalization, initialized NCCL, and trained one
batch. TensorBoard values at global step 0 (the update performed before step
counter increment) were:

- Training loss: `0.046070437878370285` (finite)
- Gradient norm: `0.805865466594696` (finite)
- Learning rate: `1.249999968422344e-05`
- Step time: 34.70 s

The run reached global step `1/1` and saved:

- `outputs/stackcube_recap/milestone3/results/stackcube_panda_pi05_sft/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt` (6,946,527,306 bytes)
- `outputs/stackcube_recap/milestone3/results/stackcube_panda_pi05_sft/checkpoints/global_step_1/actor/dcp_checkpoint/__0_0.distcp` (20,843,201,397 bytes)
- DCP metadata at the same `dcp_checkpoint/` path
- TensorBoard event file under `outputs/stackcube_recap/milestone3/results/tensorboard/`

The full launcher log is
`outputs/stackcube_recap/milestone3/logs/m31_sft_standard_launcher_one_step.log`.
The observed NCCL warning about missing GDRCopy support did not prevent the
step. No second step or extended training was run.

## Tests and post-run state

Focused accelerator regression tests:

```bash
HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets PYTHONPATH="$PWD" \
  .venv/bin/pytest -q tests/unit_tests/test_real_env.py \
  -k 'nvidia_device_count_falls_back or nvidia_model_falls_back'
```

Result: **2 passed, 234 deselected in 2.58 s**.

Post-run GPU inventory is in `outputs/stackcube_recap/milestone3/logs/m31_gpu_postrun.log`.
GPU 1 had 97,235 MiB free, 0% utilization, and an empty compute process list.
GPU 0 retained its unrelated desktop workload; no other process was stopped.

## Changes and commits

No tracked source code changed during Milestone 3.1. The test helper is an
ignored local diagnostic artifact under `outputs/`. The prior project-local
NVML fallback and focused tests remain in commit
`b140d271a6d58f9f31894f857f4ac5e568f7bc25`. Reports from Milestone 3 are in
`a4faddf1`; the environment report is in `6a526f47`. No commit was pushed.

This Milestone 3.1 report and the update to
`docs/CODEX_EXPERIMENT_PROGRESS.md` are the only intended tracked changes in
this milestone.

## Remaining blockers and next step

The one-step standard launcher path is verified when the process tree uses the
matching private NVML path. The host-wide active NVML library still mismatches
the kernel driver, so future jobs that omit the scoped path can fail again.
That persistent system correction is outside this task's boundaries. The
one-step result verifies readiness and one optimizer update; it does not
demonstrate convergence or StackCube success. Stop here for review. Do not
start full SFT, rollout collection, RECAP, or Milestone 4.
