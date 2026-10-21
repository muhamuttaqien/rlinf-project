# Milestone 1 — GPU and OpenVLA-OFT Readiness

**Inspection date:** 2026-10-21  
**Repository:** `/home/muttaqien/Desktop/AIST-Tokyo/gamma/rlinf-project`

This report records read-only checks from Milestone 1. No dependencies were installed or upgraded, no driver or kernel settings were changed, and no training was executed.

## 1. GPU diagnostics

### Verified: Codex sandbox command context

The following checks ran in the normal sandbox command context:

```text
CUDA_VISIBLE_DEVICES=unset
NVIDIA_VISIBLE_DEVICES=unset
ls: cannot access '/dev/nvidia*': No such file or directory
NVRM version: NVIDIA UNIX Open Kernel Module for x86_64 590.48.01
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
/bin/bash: nvcc: command not found
```

PyTorch, checked in this context, reported:

```text
torch: 2.11.0+cu130
torch CUDA build: 13.0
cuda available: False
device count: 0
```

PyTorch also warned that NVML could not be initialized.

### Verified: host command context

After approval, read-only host-context checks showed NVIDIA device files `/dev/nvidia0` through `/dev/nvidia3`, `/dev/nvidiactl`, `/dev/nvidia-modeset`, and NVIDIA UVM devices. `/proc/driver/nvidia/version` reported NVIDIA UNIX Open Kernel Module version `590.48.01`.

Host `nvidia-smi` output reported:

```text
NVIDIA-SMI 590.48.01     Driver Version: 590.48.01     CUDA Version: 13.3
GPU 0–3: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
```

Host-context PyTorch, using the same repository `.venv`, reported:

```text
torch: 2.11.0+cu130
torch CUDA build: 13.0
cuda available: True
device count: 4
device 0–3: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
```

The host `nvidia-smi` snapshot showed GPU 0 at 1061 MiB / 97887 MiB and GPUs 1–3 at 16 MiB / 97887 MiB each at that time. This was only a utilization snapshot, not a reservation or availability guarantee. `nvcc` was not found in the host context either.

### Assessment

**Verified:** the ordinary sandbox command context lacks NVIDIA device nodes and cannot communicate with the NVIDIA driver; host-context checks can enumerate four GPUs and host-context PyTorch CUDA works.

**Inference:** the Milestone 0 GPU failure was caused by the sandbox device-access boundary rather than an absent or inactive host GPU/driver. The driver advertises CUDA 13.3 while PyTorch is built for CUDA 13.0; successful host CUDA initialization verifies that this combination works for basic device discovery. No model kernels or training behavior were tested.

## 2. OpenVLA-OFT dependency validation

### Existing upstream integration

The repository contains two OpenVLA-OFT model implementation variants:

- `rlinf/models/embodiment/openvla_oft/rlinf/`
- `rlinf/models/embodiment/openvla_oft/official/`

The installer function `install_openvla_oft_model()` selects the upstream source by environment. Most branches install `https://github.com/moojink/openvla-oft`; the RoboTwin and Franka branches install `https://github.com/RLinf/openvla-oft.git@RLinf/v0.1`. The installer also selects common embodied dependencies, environment-specific installation, and FlashAttention. No installer was run.

### Verified package availability

Import-spec checks in the existing `.venv` found:

| Dependency | Result |
|---|---|
| PyTorch | Present, 2.11.0+cu130 |
| TorchVision | Present, 0.26.0+cu130 |
| Transformers | Present, 4.57.6 |
| PEFT | Present, 0.21.2 |
| timm | Present |
| safetensors | Present |
| accelerate | Present |
| FlashAttention | Present, 2.8.3+cu13torch2.11cxx11abiTRUE |
| Hydra / OmegaConf | Present |
| Pillow, NumPy, einops, tqdm | Present |
| `prismatic` | **Missing** |

The inspected OpenVLA-OFT source imports PyTorch, NumPy, Pillow, Transformers, OmegaConf and `prismatic` modules. The `prismatic` imports include OpenVLA config/model and processing code, projectors, VLA constants, and dataset normalization utilities. `torchvision.transforms` is imported by the local Prismatic processor code.

### Verified import results

With the repository and unit-test directory on `PYTHONPATH`:

```text
rlinf.models.embodiment.openvla_oft IMPORT OK
rlinf.models.embodiment.openvla_oft.rlinf.openvla_oft_action_model
  IMPORT FAILED: ModuleNotFoundError No module named 'prismatic'
rlinf.models.embodiment.openvla_oft.official.openvla_oft_action_model
  IMPORT FAILED: ModuleNotFoundError No module named 'prismatic'
```

The package-level module imports lazily, so its success does not establish that either concrete implementation can load. The direct implementation import failures establish that the current environment is not ready for OpenVLA-OFT model loading.

## 3. Configuration validation

Hydra composition was checked against `examples/embodiment/config/maniskill_ppo_openvlaoft_quickstart.yaml`, setting `EMBODIED_PATH` to the repository’s `examples/embodiment` directory. The successful check reported:

```text
compose: OK
model_type: openvla_oft
implement_version: rlinf
model_path: /path/to/model/RLinf-OpenVLAOFT-ManiSkill-Base-Main/
required fields: True
config resolve: True
```

Hydra emitted a warning that the config defaults list is missing `_self_`. A first inspection attempt queried `rollout.model`, but this config places the policy under `actor.model`; after querying the configured location, composition and resolution succeeded.

The configured checkpoint is a placeholder. The checked path `/path/to/model/RLinf-OpenVLAOFT-ManiSkill-Base-Main` did not exist. The alternate placeholder `/path/to/model/Openvla-oft-SFT-libero10-traj1` did not exist either. No checkpoint or model weights were loaded.

## 4. Identified blockers

### Verified blockers

- The default sandbox command context cannot access the host GPUs; host-context PyTorch can.
- `prismatic` is absent, and both concrete OpenVLA-OFT action-model imports fail with `ModuleNotFoundError: No module named 'prismatic'`.
- The quickstart model path is a placeholder and absent locally.
- The checked placeholder locations `/path/to/ManiSkill` and `/path/to/dataset` were absent. These checks do not establish whether assets are available at other locations.
- `nvcc` was not found in either context. PyTorch CUDA does work in host context, so the missing compiler did not prevent basic device discovery.
- Hydra composition succeeds but emits a missing-`_self_` defaults warning.

### Smoke-test implications

**Inference:** a minimal model-backed training smoke test cannot currently proceed in the sandbox because it lacks GPU access, and the inspected environment cannot import the concrete model implementation or find the configured checkpoint. Even from host context, model loading would remain blocked until the upstream package and valid model/assets are available. No end-to-end simulator readiness check was performed.

## 5. Recommendations for Milestone 2

1. Select the intended benchmark and implementation variant (`rlinf` or `official`) and use the corresponding existing RLinf installer environment definition; do not create a separate training pipeline.
2. Arrange an approved execution context with host GPU device access for later GPU checks and smoke testing.
3. Make the selected upstream OpenVLA-OFT dependency available in the target environment, then rerun the two direct implementation imports.
4. Point the existing config at a real checkpoint and verify the chosen simulator and dataset assets at their actual locations.
5. Address the Hydra `_self_` warning if the project maintainers want a warning-free config, then rerun composition.
6. After those preflight checks pass and with explicit approval, run a narrowly scoped smoke test using the existing RLinf entry point and config. Keep model training out of scope until that approval.

## 6. Change and execution record

- Files changed by this report-writing task: this report only.
- Dependencies installed or upgraded: none.
- Driver or kernel settings changed: none.
- Training or model-weight loading executed: none.
- GPU diagnostics: read-only `ls`, `/proc/driver/nvidia/version`, `nvidia-smi`, `nvcc --version`, and PyTorch CUDA availability/device enumeration.
- Configuration and imports: read-only Hydra composition/resolution and Python import checks.
