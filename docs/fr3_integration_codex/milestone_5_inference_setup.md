# Milestone 5 — FR3 OpenPI pi0.5 inference integration (offline input boundary)

## Goal

Integrate actual OpenPI pi0.5 inference with the FR3 + Robotiq interface, initially with **no actuation**. This document records the *first* offline input-side step only; Milestone 5 is **IN PROGRESS**, not complete.

## Existing code reused

- Upstream \`pi05_franka_pnp\` registered in \`rlinf/models/embodiment/openpi/dataconfig/__init__.py\`: pi0.5, horizon 5, \`action_dim=32\`, \`discrete_state_input=False\`, with \`LeRobotRealworldDataConfig\`.
- \`rlinf/models/embodiment/openpi/policies/realworld_policy.py\`: actual \`RealworldInputs\` mapping for two images, 19D state, and no third camera (mask false).
- \`toolkits/realworld_check/fr3_policy_contract.py\`: candidate physical 19D FR3 state, **not approved as final normalization/production schema**.
- Existing raw episode \`~/fr3_demo_20261107_084244\` with 300 FR3/Robotiq/base+wrist camera samples (no operator events).
- \`RealworldOutputs\` extracts first seven decoded action channels, but no published ROS2 controller action mapping exists.

## Implemented

\`toolkits/realworld_check/fr3_pi05_inference_preflight.py\` loads one actual recorded observation from disk, converts both BGR views to RGB, builds the candidate 19D state, invokes **real** \`RealworldInputs(ModelType.PI05, action_dim=32)\`, and verifies resulting padded state, camera names, masks, and RGB fidelity. No checkpoint is loaded, no autonomous control, no ROS publishers and no hardware access.

It separately implements \`audit_prediction_chunk\` to reject malformed/nonfinite decoded (H,7) outputs while **always returning execution_approved=false**. This is shape auditing, not a safe-action adapter.

## One workstation run (offline)

Use the existing workstation Python environment with \`torch\` and \`openpi\` importable.

\`\`\`bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
PYTHONPATH="$PWD" python -m toolkits.realworld_check.fr3_pi05_inference_preflight --raw "$HOME/fr3_demo_20261107_084244"
\`\`\`

Success: \`passed: true\`, state \`19 -> 32\`, image masks \`base_0_rgb=true, left_wrist_0_rgb=true, right_wrist_0_rgb=false\`. If OpenPI dependencies are missing, this must fail clearly instead of reporting verification.

Optional focused CPU-only tests:
\`PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/unit_tests/test_fr3_pi05_inference_preflight.py\`

## Remaining Milestone 5 work

1. Choose/check a real available \`pi05\` checkpoint and normalization assets, test a single **offline** policy inference with this two-camera/19D input.
2. Review the resulting decoded (H,7) action units and frame semantics against the actual NUC controller and Robotiq command interface. The currently proposed target representation is not approved for actuation.
3. Implement a strictly bounded, default-disabled ROS 2 action bridge with controller-enforced safety, freshness, exclusive authority, and local safety approval *before any physical inference test*.
4. Verify host/NUC transport and operator enable/disable handoff without issuing any policy-generated motion. Defer real robot evaluation to Milestone 6.

Do not assert policy inference, checkpoint loading, transport or safe robotic command execution merely because the input transform passed.
