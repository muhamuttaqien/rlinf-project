# Milestone 4C.2 — proposed 19D state and 7D action contract

**Status: candidate implementation for offline review; production contract NOT approved.**

The existing OpenPI `RealworldInputs` requires exactly 19 state values,
while RLinf's Franka env representation is 20D. Neither the raw recorder
nor the ROS SpaceMouse stream provides trustworthy (N,7) *training*
actions. Do not truncate 20D to 19D or treat normalized `Twist` as
metric motion.

## Explicit candidate 19D state (preview only)

A physically interpretable option based on fields measured by the 4A
FR3+Robotiq recorder:

| Indices (zero-based) | Source | Representation and units |
| --- | --- | --- |
| 0–2 | measured TCP pose | x/y/z, metres, frame to verify |
| 3–6 | measured TCP pose | quaternion x/y/z/w, dimensionless, frame to verify |
| 7–13 | FR3 measured joint positions | joints 1–7 in radians |
| 14–16 | measured external wrench force | x/y/z in newtons, source frame to verify |
| 17 | measured Robotiq left knuckle position | radians (NOT aperture width) |
| 18 | measured external wrench torque | L2 norm of 3D torque in N·m |

This yields **7 + 7 + 3 + 1 + 1 = 19**. The deliberate compromise is
retaining the Robotiq feedback, TCP quaternion and all joint positions
while reducing torque direction to magnitude and omitting velocities
and effort. This is an **engineering proposal**, not a validated
best-performing policy state. Selecting the force/torque features
and mapping gripper angle are open research design decisions.

The preview enforces finite physical readings, exact widths, and an
approximately unit-length TCP quaternion. It does not silently
normalize, invent gripper width, or make data training-ready.

## Proposed future 7D action semantics (no labels generated)

| Indices | Proposed meaning |
| --- | --- |
| 0–2 | base-frame TCP translation displacement (metres) |
| 3–5 | base-frame TCP rotation-vector displacement (radians) |
| 6 | commanded Robotiq gripper aperture width target (metres) |

This is a **candidate future target contract**, not a mapping from
SpaceMouse axes. It requires controller-side evidence of accepted
metric EE/gripper setpoints, reference and rotation frames, tool/TCP
calibration, exact action interval and a verified timebase. The
present domain-70 `JointImpedanceIKController` takes normalized
SpaceMouse inputs and computes IK pose offsets (0.007 m, 0.03 rad
scales); domain-71 Robotiq has a distinct control path. Neither is
an observed executed 7D action label. In particular, measured pose
differences are motion observations and must not be mislabeled as
accepted controller commands.

Action samples must eventually be aligned to observations at an
explicitly approved sampling interval, with actual controller acceptance
and gripper command provenance, not merely timestamped input intent.

## Offline usage (workstation, no ROS or hardware access)

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" \
python -m pytest -q \
  tests/unit_tests/test_fr3_policy_contract.py \
  tests/unit_tests/test_fr3_lerobot_preflight.py \
  tests/unit_tests/test_franka_pi05_contract.py

python -m toolkits.realworld_check.fr3_policy_contract \
  --raw "$HOME/fr3_raw_buttons_20261106_183614" \
  --output "$HOME/fr3_4c2_contract_$(date +%Y%m%d_%H%M%S)"
```

The command revalidates the whole raw dataset, creates a **new**
`contract_candidate.json` and `state_candidate_first.npy`
(19D first observation only), and reports unresolved approvals.
No `actions` are exported, and the result must not be passed to a
policy trainer. The original raw capture is untouched.

## Real workstation candidate validation — November 7, 2026

The operator ran focused CPU-only unit suites in the `aist-gamma`
Python 3.12.3 environment:

- `test_fr3_policy_contract.py`: **11 passed**
- `test_fr3_lerobot_preflight.py`: **7 passed**
- `test_franka_pi05_contract.py`: **7 passed**
- **Total: 25 passed in 0.22 seconds**

The offline candidate CLI successfully revalidated the previously
recorded `fr3_raw_buttons_20261106_183614` dataset, containing
**100** genuine raw observations, and created a review-only
`fr3_robotiq_policy_contract_candidate_v1` report.
It reported `state_dim=19`, `float32`, and source-header
`frame_id=base` for both TCP pose and external wrench.

The output also explicitly indicated:
`status=candidate_requires_review`,
`action_interval_s=null`, `executed_action_source=null`,
`gripper_width_calibration_verified=false`,
`action_labels_available=false`,
`le_robot_export_ready=false`, `sft_ready=false`, and
`recap_ready=false`.

**Interpretation:** candidate code and the first-state projection
are offline validated, but the CLI only projects the first
observation to 19D; the separate raw preflight checks all 100
state/image records. Do not claim that a full 100-frame projected
policy-state dataset or a usable 7D-action training set exists.

Next: inspect actual candidate values and frame-specific semantics,
verify gripper measurement calibration, and qualify the action
definition independently of normalized SpaceMouse intent.

## Approval gates before 4C.2 can be marked complete

- [ ] Human selection or revision of candidate 19D feature names,
      ordering, frame definitions, unit conventions and gripper feedback.
- [ ] Verify TCP frame id and physical tool offset and quaternion order.
- [ ] Confirm normalized force and torque semantics and whether torque
      magnitude is useful compared with directional components.
- [ ] Verify Robotiq width conversion/limits and accepted control targets.
- [ ] Choose the actual 7D action source and interval; confirm that
      metrics refer to accepted controller commands, not SpaceMouse intent.
- [ ] Verify frame/action cross-host timing and documented tolerances.
- [ ] Validate real human-labeled episodes, then end-to-end OpenPI SFT
      and two-camera RECAP loaders in 4C.3/4C.4.

No source code is added to the robot controller, and this module cannot
write to robot-control topics.
