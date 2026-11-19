# Milestone 5 — Candidate FR3/Robotiq policy-command contract (offline only)

**Outcome:** an explicit typed review boundary is implemented. It cannot send arm or gripper commands. This is not an autonomous controller adapter, motor mapping, or workcell safety approval.

## Observed interfaces (November 7, 2026)

- NUC `ros2 control list_controllers` reported `joint_impedance_ik_controller` (`franka_arm_controllers/JointImpedanceIKController`) **inactive**, with the two state broadcasters **active**. Controller presence and topic subscription are NOT the same as active policy-motion readiness.
- NUC `ros2 topic info -v` on domain 70 reported one `geometry_msgs/msg/Twist` publisher and one `joint_impedance_ik_controller` subscriber with `RELIABLE` vs `BEST_EFFORT` QoS. The publisher identity was `_NODE_NAME_UNKNOWN_`; **exclusive ownership is NOT verified**.
- The first SSH audit reported zero subscriptions and `Node not found`; the later detailed NUC query found one subscription and an inactive controller. Treat graph-discovery snapshots as non-atomic; do not infer safety, activation or successful motion from them.
- Domain 71 reported `/robotiq_gripper_controller/gripper_cmd` as `control_msgs/action/GripperCommand`. Type discovery does NOT validate continuous joint-angle semantics, gripper motion/feedback, or control-authority handoff.
- The arm's existing NUC controller takes **six normalized SpaceMouse Twist inputs** (not physical m/s and rad/s). Source-reviewed IK increments `0.007 m` and `0.03 rad` are implementation scaling values, NOT independent velocity/workspace safety certification. A controller timeout of `0.25 s` was previously observed, but current parameter discovery returned `Node not found` while inactive; current setting remains unverified. Timeout holds an impedance reference, NOT an E-stop.
- The actual StackCube Panda π0.5 checkpoint's seven output channels are NOT semantically mapped to real FR3. Its normalization state 18D does not match the proposed FR3 state 19D. The actual model predictions remain quarantined and never become candidate commands.

## Offline candidate review contract

`toolkits/realworld_check/fr3_policy_command_contract.py` defines:

| Field | Offline proposed convention (NOT hardware-qualified) |
| --- | --- |
| Channel 0–2 | TCP Cartesian translation *delta* in metres |
| Channel 3–5 | TCP rotation-vector *delta* in radians |
| Channel 6 | Candidate Robotiq knuckle *target* scalar in radians, NOT proven ROS action-server units |
| Frames | `fr3_base_candidate_unverified` and `base_frame_rotation_vector_candidate_unverified` |
| Source | `synthetic_contract_fixture_only`; real model predictions are rejected |
| Timing | One-host monotonic nanoseconds for observation, issue, and action interval; no cross-host extrapolation |
| Authority | An offline `PreviewOwner` name + expiry and monotonic sequence, NOT a hardware lease |
| Limits | Caller must provide `ReviewLimits` explicitly; **NO** default physically safe thresholds |
| Dispatch | Unconditionally raises `PermissionError`; no `rclpy`, ROS publish, GripperCommand goal or enable switch |

It checks structure, finite seven-channel values, translation/rotation vector norms, candidate gripper bounds, same-host timestamp ordering and age, action interval, owner matching, and monotonic sequence/replay. It returns `review_checks_passed` for a **synthetic test case only**; `real_robot_execution_approved` always remains `false`.

The test values (5 mm, 0.02 rad, 0 to 0.5 candidate rad) exist ONLY in the unit-test fixture. They do **not** represent selected robot safety limits, certified Robotiq stroke or recommended deployment values.

## One focused test

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" \
python -m pytest -q tests/unit_tests/test_fr3_policy_command_contract.py
```

Expected passing scenarios: valid synthetic preview (still cannot command), wrong origin, invalid frames/units, missing/incompatible limits, stale/future time, replay, ownership mismatch, and impossible dispatch. **Test has not yet run on the operator's workstation.**

## Remaining approval / engineering gates

1. Select/collect a genuine FR3 19D observation representation and normalized, labeled 7D action targets; establish policy training and real-robot domain validity.
2. Inspect **deployed** NUC controller API and action-acceptance provenance; decide whether to create a separate, mutually exclusive metric command controller rather than adapting six-axis SpaceMouse Twist.
3. Verify TCP frames/tool offsets, per-axis limits/workspace, gripper setpoint units and external safety provisions with responsible operators.
4. Design a NUC-enforced physical authority handoff, freshness and command-loss interlock, not merely client-side validation; confirm controller active/inactive state in a supervised window.
5. After approval only, develop and qualify a **separate** physical adapter; preserve this review-only module as a never-actuating boundary. No physical robot motion during Milestone 5 offline development.
