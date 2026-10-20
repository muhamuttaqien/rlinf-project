# OpenVLA-OFT StackCube integration

The TFDS export already lives at `data/stackcube_rlds/stack_cube_builder/1.0.0`.

Run from the RLinf project root:

```bash
git clone https://github.com/moojink/openvla-oft.git external/openvla-oft
python scripts/register_stackcube_openvlaoft.py --openvla-oft external/openvla-oft
python scripts/check_stackcube_rlds.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/unit_tests/test_stackcube_openvlaoft_register.py -q
```

The registration adds `stack_cube_builder` to OXE configs, mixture, and
identity transform, with both cameras and 7-D EEF actions. No state is provided,
so `state_encoding` is NONE. It does not change ManiSkill action semantics.

**Not yet training ready:** OpenVLA-OFT's upstream RLDS loader resolves datasets
through TFDS builder discovery; the standalone export uses an unregistered
`stack_cube_builder` class, so registration of OXE metadata alone is not
sufficient to guarantee the upstream `RLDSDataset` can discover it. The
`check_stackcube_rlds.py` command tests the written TFDS data independently.
The action transformation and normalization must also be checked before training.
