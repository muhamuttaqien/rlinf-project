# Codex Experiment Progress — StackCube π₀.₅ + RECAP

Updated: 2026-10-21 (Asia/Tokyo)  
Branch: `branch-openpi-recap`  
Current milestone: Research Milestone 3 — **PASSED**

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
