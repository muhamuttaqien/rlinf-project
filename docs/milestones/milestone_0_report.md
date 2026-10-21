# Milestone 0 — Repository and Environment Inspection

**Inspection date:** 2026-10-21  
**Repository:** `/home/muttaqien/Desktop/AIST-Tokyo/gamma/rlinf-project`

## 1. Repository overview

RLinf is a distributed reinforcement learning framework. The checked-out tree contains:

- `rlinf/`: framework code, including scheduler, workers, runners, algorithms, models, environments, and hybrid engines.
- `examples/`: training entry points and Hydra configurations for embodied, reasoning, agent, SFT, and offline RL workflows.
- `evaluations/`: standalone embodied evaluation entry point and benchmark-specific configurations.
- `tests/`: unit tests, end-to-end test configs/runners, parity tests, and robot mocks.
- `requirements/`, `docker/`, and `docs/`: installation support, container definitions, and documentation.

Git was on branch `branch-openpi-recap`, at commit `8c6e3e10` (`docs(recap): document official pi0.5 checkpoint preparation and cautious SFT preflight`). The initial `git status --short --branch` showed no modified or untracked files.

## 2. Python environment and dependencies

The active interpreter was the repository environment at `.venv/bin/python`:

| Component | Observed value |
|---|---|
| Python | 3.11.15 |
| pip | 26.2.2 |
| PyTorch | 2.11.0+cu130 |
| Ray | 2.59.0 |
| Hydra Core | 1.4.0.dev6 |
| Transformers | 4.57.6 |
| PEFT | 0.21.2 |
| pytest | 9.1.1 |

Torch, Ray, and Hydra were import-discoverable in the active environment. The `rlinf` distribution was not installed according to `pip show`; importing the checkout requires adding the repository to `PYTHONPATH` (as the repository instructions prescribe for unit tests). The project metadata declares Python `>=3.10`, core dependencies including Ray, PyTorch, Hydra, NumPy, and data/utilities packages, and an `embodied` extra including Transformers, PEFT, timm, Gym/Gymnasium, and image/video dependencies.

No dependencies were installed or changed during this inspection.

## 3. GPU and CUDA information

- PyTorch reports version `2.11.0+cu130` and CUDA build `13.0`.
- `torch.cuda.is_available()` returned `False`; `torch.cuda.device_count()` returned `0`.
- `nvidia-smi` was present, but failed with: “NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.”
- PyTorch emitted a warning that NVML could not be initialized.

This environment therefore did not expose a usable NVIDIA GPU. No GPU training or evaluation was started.

## 4. OpenVLA-OFT implementation status

The repository contains in-tree OpenVLA-OFT support:

- Model implementations: `rlinf/models/embodiment/openvla_oft/`, with `official/` and `rlinf/` variants.
- Model registration: `rlinf/models/__init__.py` registers the `openvla_oft` builder.
- Shared configuration: `examples/embodiment/config/model/openvla_oft.yaml`.
- Training configurations include LIBERO, ManiSkill, MetaWorld, RoboTwin, CALVIN, Behavior, OpenSora, and Wan variants. Both training and e2e configuration files were found in `examples/embodiment/config/` and `tests/e2e_tests/embodied/`.
- Evaluation configurations include LIBERO, ManiSkill, and RoboTwin OpenVLA-OFT examples under `evaluations/`; an additional RoboTwin evaluation config exists under `tests/e2e_tests/evaluations/`.
- English and Chinese OpenVLA-OFT example documentation exists at `docs/source-en/rst_source/examples/embodied/openvla_oft.rst` and `docs/source-zh/rst_source/examples/embodied/openvla_oft.rst`.

The presence of code and configurations was verified by repository inspection; model loading and execution were not tested.

## 5. Training and evaluation pipelines

Embodied training entry points include:

- `examples/embodiment/train_embodied_agent.py` for the standard embodied runner.
- `examples/embodiment/train_async.py` for asynchronous training.
- `examples/embodiment/run_embodiment.sh` and `run_async.sh` as launch scripts.
- `examples/sft/train_vla_sft.py` and `examples/sft/run_vla_sft.sh` for VLA supervised fine-tuning.

Standalone embodied evaluation is launched through `evaluations/run_eval.sh`, which selects benchmark/config settings and invokes `evaluations/eval_embodied_agent.py`. OpenVLA-OFT evaluation YAML files are present under `evaluations/libero/`, `evaluations/maniskill/`, and `evaluations/robotwin/`.

The repository also has e2e launchers/configurations under `tests/e2e_tests/embodied/` and a distinct e2e evaluation runner under `tests/e2e_tests/evaluations/`. These pipelines were identified, not executed.

## 6. Test results

Two lightweight repository checks were run:

1. `python tests/unit_tests/check_missing_init.py` — **passed**: `OK: all python dirs have __init__.py`.
2. `PYTHONPATH="$(pwd):$(pwd)/tests/unit_tests" python tests/unit_tests/check_import_rlinf_package.py` — **failed**: 406 modules discovered; 54 import failures.

The import-check failures included missing optional components such as `latex2sympy2`, `vllm`, and `megatron`. Gym emitted a warning that it is unmaintained and does not support NumPy 2.0. A first invocation without `PYTHONPATH` failed all imports because `rlinf` is not installed; the corrected invocation is the 54-failure result above.

No pytest unit-test suite or end-to-end tests were run. No training or evaluation workloads were run.

## 7. Identified issues

- There is no usable NVIDIA GPU/driver in the inspected environment, preventing GPU-backed OpenVLA-OFT execution here.
- The checkout is not installed as a Python distribution, so repository imports need `PYTHONPATH` or a separately prepared package environment.
- The package-wide import check has 54 failures, including absent optional dependencies for math rewards, vLLM, and Megatron. The failure count does not by itself indicate core OpenVLA-OFT import failure; this was not separately exercised.
- Gym reports its upstream maintenance and NumPy 2 compatibility warning.
- The OpenVLA-OFT model and simulator assets/checkpoints were not loaded or validated during this inspection.

## 8. Recommended Milestone 1

First define the intended OpenVLA-OFT target workflow (benchmark, training or evaluation, config, checkpoint, and available hardware). Then perform a focused preflight: verify the selected config and required model/simulator assets, and run a narrowly scoped CPU-safe test or import/config check. GPU execution should follow only in an environment with a working driver and the required accelerator and assets.

No project files other than this report were intended to be changed for Milestone 0.
