# Milestone 5 — Policy-to-controller handoff: offline preview only

The real FR3 arm controller accepts **normalized six-axis SpaceMouse intent**
on ROS domain 70, while a future FR3 π0.5 training policy proposes **seven
metric channels**, with the seventh referencing Robotiq knuckle position.
Those domains are **not equivalent**. This addition prepares the unit-testable
mathematical bridge and fail-closed control-handoff logic, **not** a physical
ROS publisher, robot controller plugin or safety certification.

## Implemented

- `toolkits/realworld_check/fr3_policy_handoff_preview.py`
  - Imports and reuses the existing strict offline `CandidateAction` contract.
  - Explicit source-reviewed mathematical scales; **no default physical limits**.
  - Single offline owner, monotonically ordered timestamps, heartbeat age,
    expiry, replay detection, disabled/armed/fault-latched states.
  - Six *illustrative* normalized channel values. No implicit clipping.
  - Seventh component permits only the source-observed binary Robotiq open
    (0.0 candidate rad) / close (0.7929 candidate rad) endpoints; intermediate
    angles are rejected. Does not issue an actual gripper action.
  - Rejects StackCube origin or unverified frames through the existing contract;
    its `send_to_robot` method always raises `PermissionError`. Zero ROS imports.
- `tests/unit_tests/test_fr3_policy_handoff_preview.py`: focused tests for
  mapping math, no dispatch, gripper endpoints, expiry, heartbeat, replay, owner,
  and explicit bounds. **Not yet run on the workstation.**

## Workstation verification — one command, CPU only

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit_tests/test_fr3_policy_handoff_preview.py
```

## Remaining boundary

The controller's source-reviewed values 0.007 m / 0.03 rad per normalized
axis are *heuristic per-update increments*, **not** safe limits.
Rotation-vector components may not be interchangeable with the deployed
controller's incremental rotation convention. Cross-host timing, authority,
tool-frame and gripper calibration and handling of lost commands remain
unverified on physical hardware. Do **not** wire the preview's numbers to
`/franka_controller/target_cartesian_velocity_percent`, nor treat its
"armed_preview_only" mode as physical robot enablement.

**Milestone 5 can finish offline preparation only after this focused check.**
A separate, supervised NUC-side exclusive control interface with real safety
limits, E-stop availability, and rollback needs authorized review before
policy-driven physical motion in Milestone 6. This file intentionally contains
no physical enable switch, ROS publisher or action client.
