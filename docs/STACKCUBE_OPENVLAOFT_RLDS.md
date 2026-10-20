# StackCube RLDS export for OpenVLA-OFT

## Verified integration facts

The upstream `moojink/openvla-oft/vla-scripts/finetune.py` consumes
`RLDSDataset` and `RLDSBatchTransform`, rather than the RLinf
intermediate SFT index. It requires an OXE registration, an action
normalization key and compatible robot action conventions.

Our TensorFlow Datasets exporter generates full **episode-separated** examples
containing `steps/observation/image`,
`steps/observation/wrist_image`, `steps/action` (7-D),
`steps/language_instruction`, and RLDS step flags.

First verify without TensorFlow:

```bash
python scripts/export_stackcube_openvlaoft_rlds.py --dry-run
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/unit_tests/test_stackcube_openvlaoft_rlds.py -q
```

Then, in an isolated environment with TensorFlow and tensorflow-datasets:

```bash
python scripts/export_stackcube_openvlaoft_rlds.py \
  --index data/stackcube_openvlaoft_sft/index.json \
  --output data/stackcube_rlds
```

The resulting TFDS builder class is named `StackCubeBuilder`, and it
currently registers as `stack_cube_builder`, **not** `stackcube_panda`.
An upstream OpenVLA-OFT OXE registration is still required before
`dataset_name` can resolve this dataset; the export alone does not
make it trainable.

**Safety of experiments:** The raw actions are ManiSkill
`pd_ee_delta_pose`, not verified equivalent to `bridge_orig` or
`libero` action targets. Validate rotation and gripper conventions and
normalize using the training-split statistics before any policy rollout.

No OpenVLA-OFT policy training is performed by this exporter.
