# Milestone 5 — Simulation checkpoint training provenance (read-only)

**Purpose:** resolve which *dataset* and *action convention* generated the actual `global_step_1084` actor weights, before any FR3 normalization or command mapping. A successful GPU forward and a seven-channel output **cannot** establish units, frame or gripper behavior.

## Verified source-code distinction

- `rlinf/models/embodiment/openpi/policies/isaaclab_policy.py` uses `IsaacLabOutputs` to apply `np.sign` to the seventh action (discrete gripper intent in the IsaacLab StackCube adapter).
- `rlinf/models/embodiment/openpi/policies/maniskill_policy.py` and `realworld_policy.py` only extract output channels. That extraction **does not certify** the semantics of the checkpoint that generated them.
- The saved checkpoint filename `stackcube_panda_pi05_full_sft` alone does **not** establish which data transform or controller action format its training dataset used. The actual training config and dataset metadata must be checked.

User confirmed `norm_stats.json` has **18 state channels and 7 action channels**; the candidate real FR3 state contract requires **19 state channels**. No simulation norm stats have been applied and no FR3-ready real action labels exist in the stationary 300-frame recording.

## Code added

`toolkits/realworld_check/fr3_checkpoint_provenance_audit.py` checks that the **local** RLinf `full_weights.pt` exists but never loads it, audits norm JSON for structural consistency, and performs a bounded scan for textual training configs under the *actual* output directory. Matching lines are reported for human review, NOT automatically used as proof of physical action semantics. It has no ROS imports and never issues commands.

Run exactly one read-only workstation audit:

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
PYTHONPATH="$PWD" python -m toolkits.realworld_check.fr3_checkpoint_provenance_audit \
  --checkpoint "outputs/stackcube_recap/full_sft/results/stackcube_panda_pi05_full_sft/checkpoints/global_step_1084/actor/model_state_dict/full_weights.pt" \
  --norm-stats "outputs/stackcube_recap/openpi_norm/stackcube_panda/norm_stats.json" \
  --experiment-root "outputs/stackcube_recap/full_sft"
```

Expected fields: `normalization.state_dim=18`, `normalization.action_dim=7`, plus `experiment_metadata.files` with any locally preserved config snippets. **If no configs were stored, the report must say so**; the next source of evidence is the underlying LeRobot training dataset's `meta/info.json` and original data conversion/control-mode definition. Do not infer semantics from filenames, static example configs, or shape alone.

The test only reads local file metadata and small JSON/YAML text files; it does NOT load the model, normalize recorded FR3 states, or publish to any ROS domain.

Optional focused tests, only if code needs verification:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" python -m pytest -q tests/unit_tests/test_fr3_checkpoint_provenance_audit.py`

**Milestone 5 stays in progress:** actual policy training data, FR3 action labels, appropriate normalization, controller authority, safety sign-off are still missing. No physical execution.
