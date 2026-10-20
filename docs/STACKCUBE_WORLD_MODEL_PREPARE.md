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


## Action-conditioned eight-step window indices

The existing RLinf Wan configuration uses five condition frames plus eight
generated frames; OpenSora uses four condition frames plus eight generated
frames. Prepare lazy supervised windows for both:

```bash
python scripts/index_stackcube_world_model.py --backend wan \
  --input data/stackcube_world_model_dual \
  --output data/stackcube_wm_ready/wan_windows.json
python scripts/index_stackcube_world_model.py --backend opensora \
  --input data/stackcube_world_model_dual \
  --output data/stackcube_wm_ready/opensora_windows.json
```

These indices retain both camera views and explicitly align
`condition=o[t-To+1:t]`, `actions=a[t:t+8]`, and
`targets=o[t+1:t+8]`. They are **offline training-window indices**, not
the native training format of DiffSynth Wan or OpenSora STDiT.
Both backends still require appropriate model-training configuration, a
StackCube-trained reward model, and validation of action conventions.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/unit_tests/test_stackcube_world_model_windows.py -q
```


## Prediction evaluation baseline (no model weights needed)

Before training Wan/OpenSora, run the eight-step **persistence baseline** on the
aligned StackCube windows. It predicts that the last observed frame remains
unchanged across the next eight steps. MAE is reported on the 0–255 RGB scale,
separately for the base camera and wrist camera.

```bash
python scripts/evaluate_stackcube_prediction.py \
  --index data/stackcube_wm_ready/wan_windows.json \
  --output outputs/stackcube_prediction/wan_persistence \
  --max-windows 50
python scripts/evaluate_stackcube_prediction.py \
  --index data/stackcube_wm_ready/opensora_windows.json \
  --output outputs/stackcube_prediction/opensora_persistence \
  --max-windows 50
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/unit_tests/test_stackcube_prediction_eval.py -q
```

Output: `metrics.json`, `base_persistence_comparison.png` and
`wrist_persistence_comparison.png`. The images compare last observation,
ground-truth final frame, and unchanged-frame prediction.

**This is not Wan or OpenSora inference.** It establishes a verifiable metric
and visualization pipeline before implementing and evaluating actual model
predictions. The official LIBERO checkpoints are not validated for StackCube.
