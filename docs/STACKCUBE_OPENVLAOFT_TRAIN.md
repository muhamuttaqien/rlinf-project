# OpenVLA-OFT first policy training step

The StackCube RLDS dataset loads correctly through upstream OpenVLA-OFT with
eight action targets per observation. Use its existing SFT implementation,
not RLinf's generic VLA SFT worker (which does not register OpenVLA-OFT).

## Preflight

From the RLinf root:

```bash
git pull origin branch-world-model
python scripts/train_stackcube_openvlaoft.py --check-only
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
    tests/unit_tests/test_stackcube_openvlaoft_train.py -q
```

## Single LoRA optimization step

The command uses the upstream `moojink/openvla-oft` training script with the
prebuilt `stack_cube_builder` RLDS dataset, two camera images, action
chunk length 8, L1 action head, one process, offline W&B logging and LoRA.

**Memory warning:** The standard 7B OpenVLA-OFT bf16 recipe normally
requires more than a single 16 GiB GPU. The launcher deliberately refuses
to start on less than ~26 GiB unless explicitly overridden; that override
does not make the training memory-efficient.

```bash
python scripts/train_stackcube_openvlaoft.py --steps 1 \
    --model openvla/openvla-7b
```

No policy training is started by `--check-only`. Before attempting an actual
full training run, verify the checkpoint's action/gripper interpretation; the
training dataset uses raw ManiSkill `pd_ee_delta_pose` actions, with
normalization computed from training episodes. The first policy step is only
a feasibility test, not a usable robot controller. Preserve .venv and
.venv-wan, and install policy-specific dependencies separately.
