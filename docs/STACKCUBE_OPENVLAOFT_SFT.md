# StackCube OpenVLA-OFT supervised baseline

RLinf already provides a ManiSkill-specific OpenVLA-OFT model configuration
(`examples/embodiment/config/model/openvla_oft.yaml`) and an existing
ManiSkill GRPO reference
(`examples/embodiment/config/maniskill_grpo_openvlaoft.yaml`).
The reference points to `RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main` and
uses `unnorm_key: bridge_orig`. That setting is **not validated** for
the locally recorded `pd_ee_delta_pose` StackCube expert actions.

## Prepare an episode-separated baseline dataset

```bash
python scripts/prepare_stackcube_openvlaoft_sft.py \
  --input data/stackcube_world_model_dual \
  --output data/stackcube_openvlaoft_sft \
  --val-fraction 0.1 --seed 42

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/unit_tests/test_stackcube_openvlaoft_sft.py -q
```

The output includes `index.json` (training/validation examples with whole
episodes held out) and `train_action_stats.json` (statistics computed only
on training episodes). Each sample includes the base camera, wrist camera,
one text instruction, and eight aligned seven-dimensional actions.

**Important:** The output is an intermediate dataset; it is *not* an OpenVLA-OFT
training-ready Hugging Face / LeRobot dataset. The official OpenVLA-OFT
processing path, expected proprioceptive inputs, robot embodiment tokens,
normalization statistics and action conventions must be checked before
fine-tuning or rollout. The current 100 demonstrated episodes are suitable
for a preliminary experiment but may be insufficient for robust performance.

**No policy weights are modified or trained by this script.**


## Verified SFT training compatibility

The generic RLinf trainer `examples/sft/train_vla_sft.py` uses
`FSDPVlaSftWorker`. Its `build_dataloader()` does not register
`openvla_oft`, so passing this dataset to that runner is **not supported**.

The upstream OpenVLA-OFT repository (`moojink/openvla-oft`) provides
`vla-scripts/finetune.py`, which trains from RLDS using `RLDSDataset`
and `RLDSBatchTransform`. It cannot consume our intermediate
`index.json` directly.

Next work: implement a TensorFlow Datasets RLDS exporter plus the custom
StackCube OXE registration, verify camera, robot embodiment and action
normalization, and then validate the official LoRA training path. Do not
use `unnorm_key: bridge_orig` for StackCube without confirming conversion.

```bash
python scripts/audit_stackcube_openvlaoft_sft.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/unit_tests/test_stackcube_openvlaoft_audit.py -q
```

This audit is a compatibility gate, **not** policy training.
