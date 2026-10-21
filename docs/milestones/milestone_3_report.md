# Research Milestone 3 Report — OpenPI π₀.₅ StackCube SFT

## Status

**PASSED** — the pretrained Pi0.5 model consumed one validated StackCube batch,
completed the RLinf/OpenPI supervised forward and backward passes, produced
finite loss and gradients, and completed exactly one optimizer step on GPU 1.

The regular Ray/FSDP launcher remains blocked by this host's NVML/NCCL
`Driver/library version mismatch`. The successful bounded smoke test therefore
used the same RLinf model factory, official OpenPI dataloader, `ForwardType.SFT`
objective, and RLinf AdamW builder without the FSDP/Ray wrapper. No full SFT was
run.

## Starting state

- Repository: `/home/muttaqien/Desktop/AIST-Tokyo/gamma/rlinf-project`
- Branch: `branch-openpi-recap`
- Starting HEAD: `8c6e3e10`; branch was synchronized with its remote tracking
  branch.
- Pre-existing untracked `docs/milestones/milestone_0_report.md` and
  `milestone_1_report.md` were preserved and not staged.
- Existing Milestone 1/2 dataset and loader artifacts were not regenerated or
  altered.

## Objective and phases

The objective was to load an official pretrained OpenPI Pi0.5 checkpoint and
execute exactly one supervised optimization step from the validated
`StackCube-v1` LeRobot dataset. Phases covered repository/environment
verification, checkpoint conversion, model/data loading, then one bounded
optimization step.

## Changes made

- Updated `rlinf/scheduler/hardware/accelerators/nvidia_gpu.py` to fall back to
  PyTorch CUDA device count and model name when NVML cannot enumerate devices.
  This was required because Ray's bundled NVML raised
  `NVMLError_LibRmVersionMismatch` while CUDA itself remained usable.
- Added two focused tests in `tests/unit_tests/test_real_env.py` for the count
  and model-name fallback.
- Implementation commit:
  `b140d271a6d58f9f31894f857f4ac5e568f7bc25`
  (`fix(scheduler): fall back to CUDA when NVML fails`).
- Created ignored local artifacts under `outputs/stackcube_recap/milestone3/`.
  No datasets, weights, checkpoints, logs, or secrets were added to Git.

## Pretrained checkpoint provenance and compatibility

- Source path:
  `/home/muttaqien/.cache/openpi/openpi-assets/checkpoints/pi05_base`
- Source format: complete local Orbax/JAX checkpoint; `params/commit_success.txt`
  was present. The source occupied about 12 GiB.
- Upstream origin: official OpenPI asset URI
  `gs://openpi-assets/checkpoints/pi05_base/params`, referenced by the local
  OpenPI Pi0.5 training config. The local OpenPI checkout HEAD was
  `15a9616a00943ada6c20a0f158e3adb39df2ccac` and contained unrelated uncommitted
  changes; none were modified. No asset revision or checksum manifest was
  available locally.
- Conversion used RLinf's existing `jax_to_openpi` converter with the validated
  StackCube normalization file, `--action-dim 32 --action-horizon 10
  --max-token-len 200`.
- Converted path:
  `checkpoints/torch/pi05_stackcube_panda_base/`
- Converted `model.safetensors`: 12.49 GiB, 667 tensor keys, recognized bare
  Pi0 layout. `config.json` reports `pi05: true`, Gemma 2B PaliGemma, Gemma
  300M action expert, action dimension 32, horizon 10, token length 200, and a
  bfloat16 dtype hint. StackCube normalization was copied to
  `physical-intelligence/stackcube_panda/norm_stats.json`.
- RLinf model factory loaded the checkpoint as a 3,353,433,872-parameter Pi0.5
  model. Backbone key omissions and unexpected checkpoint keys are rejected by
  the existing loader; no fallback to random initialization occurred.

## Dataset, normalization, and representations

- Dataset: `outputs/stackcube_recap/lerobot_stackcube_v2`
- Dataset validation: 10 episodes, 1,084 frames; input adapter retained the
  18D proprioceptive state, 7D `pd_ee_delta_pose` action, base and hand RGB
  streams, and prompt “Stack the red cube on the green cube”.
- Existing norm file:
  `outputs/stackcube_recap/openpi_norm/stackcube_panda/norm_stats.json`
- The OpenPI loader reported action `(1,10,32)`, state `(1,32)`, 224×224 RGB
  tensors, and prompt tokens `(1,200)`. The third output camera key was masked
  off; the two dataset camera streams remained active.
- One initial dataset check failed while Hugging Face attempted to create a
  lock in the sandbox's read-only user cache. Setting
  `HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets` resolved it; dataset contents were
  not modified.

## SFT smoke test

### Standard launcher attempts

The first `train_vla_sft.py` attempt attached to an existing shared Ray cluster
and failed before worker/model initialization because Ray/Python versions did
not match (shared cluster 2.58.0 / Python 3.11.14; project environment 2.59.0 /
Python 3.11.15). The shared cluster was left running.

An isolated `RAY_ADDRESS=local` attempt avoided the version conflict but
initially saw zero GPUs because NVML discovery failed. After the targeted
fallback, RLinf correctly placed the actor on hardware rank 1 (physical GPU 1)
and loaded the model. FSDP initialization then failed during an NCCL broadcast
with `nvmlInit_v2() failed: Driver/library version mismatch`. A retry with
`NCCL_P2P_DISABLE=1` produced the same NCCL error. These failed-attempt logs
are preserved under `outputs/stackcube_recap/milestone3/logs/`.

### Successful one-step command

The successful bounded fallback command was:

```bash
env REPO_PATH="$PWD" EMBODIED_PATH="$PWD/examples/sft" \
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets \
  PYTHONPATH="$PWD:${PYTHONPATH:-}" timeout 900s \
  .venv/bin/python outputs/stackcube_recap/milestone3/run_openpi_single_step.py
```

The driver used the existing Pi0.5 model factory, official StackCube SFT
dataloader, `ForwardType.SFT` flow-matching objective, and RLinf's
`build_adamw` with the StackCube YAML optimizer values. It loaded one batch and
called `optimizer.step()` once.

- Loss: **0.19008885324001312**, finite.
- Finite relevant gradient norms:
  - `action_in_proj.weight`: `0.0828830004`
  - `time_mlp_in.weight`: `0.738837719`
  - `action_out_proj.weight`: `0.783246398`
- `action_out_proj.weight` norm changed from `3.50348616` to `3.50297141`; max
  absolute element change was `2.50004232e-05`.
- Optimizer steps: **1**.
- Physical device: GPU 1, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation
  Edition; within the process it appeared as logical CUDA device 0.
- Initial selected-GPU state: 97,235 MiB free, 0% utilization, no compute
  process. Post-run: 97,235 MiB free, 0% utilization, no compute process.
- Peak CUDA allocated: 27,225,376,256 bytes (**25.36 GiB**).
- Peak CUDA reserved: 27,715,960,832 bytes (**25.81 GiB**).
- The checkpoint was saved and its ZIP integrity check passed:
  `outputs/stackcube_recap/milestone3/results/stackcube_panda_pi05_sft_single_step.pt`
  (6,946,594,594 bytes). Metrics are also saved in
  `outputs/stackcube_recap/milestone3/results/single_step_metrics.json`.

## Tests and validation

- `scripts/check_stackcube_openpi_inputs.py`: passed with the temporary
  Hugging Face cache.
- `scripts/check_stackcube_openpi_loader.py`: passed; expected batch shapes and
  normalization loaded.
- `scripts/check_stackcube_sft_readiness.py --checkpoint
  checkpoints/torch/pi05_stackcube_panda_base`: checkpoint header passed with
  667 tensors and recognized prefixes. Its sandbox CUDA warning was superseded
  by successful host-context CUDA preflight.
- Focused tests:
  `pytest -q tests/unit_tests/test_real_env.py -k
  'nvidia_device_count_falls_back or nvidia_model_falls_back'` — **2 passed,
  234 deselected**.
- `ruff check` and `ruff format --check` on the modified adapter and test:
  passed. `git diff --check`: passed.
- The local one-step driver passed `py_compile`; the saved checkpoint passed
  ZIP integrity verification.
- Full SFT, multi-epoch training, task-success evaluation, rollout collection,
  and RECAP were not run.

## Logs, artifacts, and commits

- Execution logs:
  `outputs/stackcube_recap/milestone3/logs/`
- Post-run GPU preflight transcript:
  `outputs/stackcube_recap/milestone3/logs/gpu_preflight_postrun.log`
- One-step metrics:
  `outputs/stackcube_recap/milestone3/results/single_step_metrics.json`
- Step-1 model checkpoint:
  `outputs/stackcube_recap/milestone3/results/stackcube_panda_pi05_sft_single_step.pt`
- The model conversion and run outputs are ignored by Git.
- Code/test commit: `b140d271a6d58f9f31894f857f4ac5e568f7bc25`.

## Remaining risks and next milestone

The default RLinf Ray/FSDP SFT launcher remains unverified through an optimizer
step because NCCL cannot initialize NVML on this host. The successful result
proves one supervised update, not learning convergence or StackCube success.
The next research milestone should first establish a repeatable FSDP launcher
run after host NVML/NCCL compatibility is resolved, then evaluate the SFT
baseline. No later milestone was started.
