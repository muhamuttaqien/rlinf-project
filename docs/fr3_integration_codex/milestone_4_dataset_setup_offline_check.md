# Milestone 4 — Offline dataset setup smoke (not an experiment)

The project owner requires **setup readiness**, not real manipulation experiments.

## Implemented

- \`fr3_dataset_setup_smoke.py\`: CPU-only deterministic check that an explicit, independently supplied NUC-minus-workstation monotonic-clock offset and uncertainty can align observation timestamps to controller ticks. Refuses absent offset, excessive uncertainty, mismatched clocks, unordered timestamps, and duplicate tick association. The smoke uses a **synthetic known offset**, not a fabricated real clock calibration.
- The same command writes and reloads a temporary **synthetic LeRobot episode** through the repository's existing LeRobot compatibility function, using a two-camera RGB + 19D state + 7D action feature schema. It validates the recovered sample with the existing OpenPI input-contract checker and deletes the temporary dataset. No real training labels are created.
- Optionally audits a previously recorded real raw episode with the existing \`fr3_lerobot_preflight.audit_raw_episode\` read-only validator. It does not synchronize nonoverlapping historic NUC and workstation recordings.

## One workstation command (offline only)

Use the Python environment that has \`lerobot\`, \`numpy\`, and the repository's dependencies. No ROS launch, robot motion, or GPU is needed.

\`\`\`bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
PYTHONPATH="$PWD" python -m toolkits.realworld_check.fr3_dataset_setup_smoke --raw "$HOME/fr3_demo_20261107_084244"
\`\`\`

Expected: \`synthetic_clock_alignment.passed=true\`, \`synthetic_lerobot_roundtrip.passed=true\`, and \`raw_episode_preflight.passed=true\`. If missing \`lerobot\` or incompatible runtime dependencies, **the smoke reports a failure**, not a fabricated pass. If those checks pass, the offline software *smoke* is verified; a complete production export is **not**.

Optional focused tests:
\`PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/unit_tests/test_fr3_dataset_setup_smoke.py\`

## Still future experimental work

- Calibrate the actual NUC/workstation monotonic clocks and camera exposure timing during **simultaneous** capture; the prior 300-frame camera run and 4,412-tick NUC run occurred at different times.
- Obtain real, accepted Cartesian/gripper actions before constructing physical 7D action labels. Current stationary \`training_action=null\`; synthetic zeros are never real action labels.
- The real-controller JSONL schema \`fr3_controller_capture_raw_v1\` is not the proposed offline validator's \`fr3_controller_target_telemetry_proposal_v1\`. Map and validate explicitly before consuming it.
- Run actual OpenPI **official SFT dataloader** and RECAP two-view loader end-to-end once their relevant dependencies and action/episode semantics are configured; the current smoke verifies local LeRobot roundtrip + OpenPI input **shape contract**, not end-to-end training compatibility.
- Formal moving-arm safety sign-off is separate and required before physical experiment.

Do not misstate the limited synthetic smoke as proof that a training-ready real FR3 dataset, true cross-host sync or complete RECAP compatibility exists.
