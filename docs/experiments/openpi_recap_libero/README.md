# OpenPI pi0.5 + RECAP on LIBERO-10 Task 0

**Status:** SIM-0 scaffold. No dataset, checkpoint, training result, or simulator evaluation has been verified on the GPU Workstation for this branch.

## Goal

Reproduce the official RLINF RECAP example on LIBERO-10 Task 0 before adapting the pipeline to real Franka hardware. RECAP trains offline on recorded LeRobot datasets; the LIBERO simulator is needed for policy rollout generation and evaluation, not for the four offline optimization stages themselves.

Reference: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html

## Existing upstream entry points (do not duplicate their implementations)

| Stage | Configuration | Launcher |
| --- | --- | --- |
| 1. Compute returns | `examples/offline_rl/config/recap_compute_returns.yaml` | `examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh` |
| 2. Train value model | `examples/offline_rl/config/recap_value_model_sft.yaml` | `examples/offline_rl/advantage_labeling/recap/run_value_sft.sh` |
| 3. Compute advantages | `examples/offline_rl/config/recap_compute_advantages.yaml` | `examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh` |
| 4. Train CFG policy | `examples/offline_rl/config/cfg_rl_openpi.yaml` | `examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh` |

Use `robot_type=libero`, `model_type=pi05` for value processing, and `actor.model.openpi.config_name=pi05_libero` for CFG. Preserve the original upstream files; use a versioned experiment configuration or Hydra overrides after the preflight.

## SIM-0: Environment and data

1. On the GPU Workstation, use a clean checkout of `branch-openpi-recap-sim`. Do not interfere with running Ray processes, jobs, or the Franka branch.
2. For a fresh RLINF environment, follow the upstream setup:

       bash requirements/install.sh embodied --model openpi --env maniskill_libero
       source .venv/bin/activate

3. Download the official reproduction dataset outside the repository (about **86.9 GB** on Hugging Face). Check free disk space and required credentials/access before downloading.

       hf download RLinf/RECAP-Libero10-Task0-48succ-Data --repo-type dataset --local-dir /absolute/path/to/RECAP-Libero10-Task0

   Dataset: https://huggingface.co/datasets/RLinf/RECAP-Libero10-Task0-48succ-Data/tree/main

   Expected child directories: `libero10_task0_sft`, `libero10_task0_train`, `libero10_task0_eval`. The SFT directory holds successful demonstrations; the train directory has mixed-success policy rollouts; eval is held out. Do not treat these as data collected by this branch.

4. Run the **read-only layout preflight**:

       python toolkits/recap_libero_preflight.py --data-root /absolute/path/to/RECAP-Libero10-Task0

   This only checks that the three LeRobot directories and basic metadata exist and are readable. It does **not** decode parquet/video, verify `is_success` labels, instantiate OpenPI transforms, check GPU compatibility, or launch training.

5. Confirm that the following model assets are available locally before training: `google/siglip2-so400m-patch14-224`, `google/gemma-3-270m`, and a compatible PyTorch OpenPI pi0.5 policy checkpoint. Their actual filesystem paths and the exact baseline policy used for rollouts must be recorded for reproducibility.

## Stage boundaries and gates

- **SIM-0 (this change):** Branch, guide, static data-layout checker and CPU unit tests. No claims about real dataset or model readiness.
- **SIM-1 (pending):** Inspect real LeRobot samples and labels; validate camera slots, prompts, actions, episode indices, success flags, and policy normalization. Establish a matched pi0.5 SFT baseline and a fixed LIBERO evaluation protocol.
- **SIM-2 (pending):** Configure Stage 1 for SFT, rollout **and held-out evaluation** returns as required by the value loader; verify `meta/returns_{tag}.parquet` for each dataset consumed downstream.
- **SIM-3 (pending):** Train the value model and check held-out return prediction quality. Explicitly configure SigLIP2/Gemma3 asset paths and checkpoint provenance.
- **SIM-4 (pending):** Compute advantages for exactly the datasets consumed by CFG; verify labels and tag linkage.
- **SIM-5 (pending):** Train the CFG policy from an explicitly selected pi0.5 checkpoint and evaluate it against the matched baseline under identical LIBERO Task 0 conditions.

Tag contract: Stage 1 `data.tag` = Stage 2 `data.tag` = Stage 3 `advantage.returns_tag`; Stage 3 `advantage.tag` = Stage 4 `data.advantage_tag`. Use new tags per experiment to avoid overwriting sidecars.

Upstream's example reports 48.8% to 66.5% success for its *specific* few-shot LIBERO Task 0 reproduction. These numbers are reference values, not results from this branch.

**Do not launch the stock example YAML unchanged:** it contains placeholder dataset/checkpoint paths, and its train/eval dataset lists must be aligned with the actual files and return sidecars before training.