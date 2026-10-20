# StackCube OpenVLA-OFT: experimental 16 GiB QLoRA

This is an **experimental** one-step SFT training path for the RTX 4070 Ti
SUPER. It changes only `external/openvla-oft/vla-scripts/finetune.py`.
The functioning Wan and RLDS pipelines remain intact.

From the RLinf project root:

```bash
git pull origin branch-world-model
# Prefer a dedicated .venv-policy if upstream OpenVLA-OFT dependencies conflict
# with .venv or .venv-wan. Use a Python environment containing the upstream
# OpenVLA-OFT dependencies, bitsandbytes and Accelerate.

python scripts/patch_stackcube_openvlaoft_qlora.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/unit_tests/test_stackcube_openvlaoft_train.py -q
python scripts/train_stackcube_openvlaoft.py --qlora --check-only
python scripts/train_stackcube_openvlaoft.py --qlora --steps 1
```

Configuration: 4-bit NF4 double quantization, bfloat16 compute,
gradient checkpointing, LoRA rank 4 on attention q/v projections,
batch size 1, one primary camera, no proprio input, and offline W&B.
The second camera remains in RLDS but is not used in this smoke test.
For checkpoint compatibility, the existing continuous L1 regression
action head remains enabled. This head is **not** itself quantized and may
still cause a 16 GiB CUDA OOM.

The first actual run may download many gigabytes for the upstream
`openvla/openvla-7b` base checkpoint. Confirm free disk space before
running. QLoRA does not guarantee 16 GiB is sufficient.

**Important:** ManiSkill `pd_ee_delta_pose` actions are normalized using
StackCube training statistics, but an end-to-end action interpretation and
simulator rollout have not yet been validated. This is a one-step
optimization **smoke test**, not a successful policy or GRPO training run.
