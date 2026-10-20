# StackCube world-model preparation (after dual-camera export)

Use the verified output in `data/stackcube_world_model_dual`. Run:

```bash
python scripts/prepare_stackcube_world_model.py \
  --input data/stackcube_world_model_dual \
  --output data/stackcube_wm_ready
```

This produces:

- `initial_frames/traj_N.npy`: an RLinf `NpyTrajectoryDatasetWrapper`-readable first-frame dictionary, with `image` (base camera), `wrist_rgb` (preserved), `delta_action` (7-D zeros), and `instruction`
- `action_stats.json`: empirical `q01` and `q99` for the OpenSora action normalizer
- `dataset_index.json`: episode metadata and source-file references

The loader and both available world-model inference backends currently consume
the base-camera `image` as their visual conditioning. The wrist camera is
preserved but does **not** automatically enable multi-view inference.

**Important limitations:** This prepares reset images and normalization
statistics. It does not train either video diffusion model, make LIBERO
weights compatible with ManiSkill, implement action-conditioned video training,
prepare a reward classifier, or supply a ManiSkill-specific policy checkpoint.
It is not yet possible to launch Wan/OpenSora+GRPO on StackCube with these
artifacts alone.

To test the preparation script:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/unit_tests/test_stackcube_world_model_prepare.py -q
```
