# Milestone 4C.2T — Proposed controller-target telemetry (offline only)

**Status:** Offline design and synthetic validator implemented. The actual NUC controller and Robotiq bridge are unchanged. There is no real telemetry capture under this schema and no accepted 7D demonstration action dataset.

## Why this exists

The deployed JointImpedanceIKController holds its IK solution in an internal RealtimeBuffer<Target>, and its velocity-limited q_desired_ remains internal to the 1 kHz update. Neither variable was found exported in the inspected .cpp/.hpp. The Franka robot-state broadcaster's last_desired_pose and desired_joint_states are not proven equivalent to these custom internal references. The existing normalized SpaceMouse Twist is operator intent, not a metric robot action.

## Proposed records

The new **transport-neutral JSONL proposal** uses schema fr3_controller_target_telemetry_proposal_v1, with one event type per record. It is an *offline validation format*, not a ROS message definition or a production logging implementation.

| Event | Required information |
| --- | --- |
| arm_tick | Same-NUC monotonic clock_id, seq, tick_ns, activation_ns, intent_recv_ns, target_state_ns, target_command_ns, target_activation_ns, sample_ns, ik_solve_ns; 6D normalized input; 7D IK joint target; q_desired_ before/after; measured 7D q; 7 joint velocity limits; dt_s, timeout; six boolean gates; reference_advanced; cumulative telemetry_drops_total |
| gripper_goal | Same-NUC clock_id, seq, goal_id; request/sent/acceptance/result timestamps and status; normalized binary request; Robotiq target joint angle, max effort; optional timestamped measured joint angle; replaced queued-request count |

All joint angles use **radians**; the gripper controller targets joint robotiq_85_left_knuckle_joint, 0 rad open and 0.7929 rad close. The gripper bridge's queue holds at most one pending goal and may replace earlier requests. An action goal acceptance/result does not establish successful grasp or a physically unobstructed closure.

ARM tick gates mirror the currently deployed implementation: moving, intent_fresh, target_valid, activation_matches, state_fresh, command_fresh. Freshness is evaluated against activation and timeout in the tick's monotonic timebase. The validator checks expected q_desired_ after applying the per-joint rate-limited update when all gates pass; otherwise it requires reference hold. It rejects malformed/nonfinite fields, inconsistent timestamps/gates, duplicate sequences, incorrect gripper endpoint mapping, and inconsistent action lifecycle. Sequence gaps and recorder drops are reported.

**Instrumenting these fields is future work.** The validator cannot prove a record came from the controller, cannot infer skipped RT ticks, and cannot verify that the 1 kHz loop has room for a publisher. A later implementation must use bounded nonblocking telemetry outside the realtime path, count drops, and undergo lab safety review, load testing, deployment approval, and rollback planning.

## Candidate 6D Cartesian training-action definition — not implemented

At a future explicitly approved observation time t_k and action horizon H, a possible reference-based target is:

- T_obs(t_k): calibrated measured TCP pose in robot base frame, with defined camera/robot clock alignment.
- T_ref(t_k+H): calibrated FK of the **logged post-rate-limit q_desired_** at the specified reference time, using the exact FR3 tool/TCP frame and calibrated offsets.
- Proposed action[0:3]: p_ref(t_k+H) minus p_obs(t_k), expressed in the verified base frame, **metres**.
- Proposed action[3:6]: Log_SO3(R_ref(t_k+H) times transpose(R_obs(t_k))), expressed as a **base-frame rotation vector in radians**. Define quaternion order, shortest-rotation branch and singularity handling before implementation.
- Proposed action[6]: associated Robotiq **controller goal joint position in radians** (not measured opening width or raw 0/1 operator request), with a documented hold policy and accepted-goal provenance.

This is a research proposal, not automatically equivalent to the controller's requested Cartesian offset, nor to actually executed TCP motion. A reference-based target has different semantics from the raw per-callback IK pose offset and from physical measured displacement. Select one policy action definition and horizon explicitly before implementing any conversion. Do not derive real action labels from the old stationary raw capture or from broadcaster topics.

## Clock and readiness gates

Domain 70 and domain 71 run on the NUC, where a common monotonic clock could support correlation if recorded correctly. ROS header time and workstation monotonic time are **not automatically comparable** to NUC monotonic timestamps. Establish cross-host clock mapping, drift/tolerance, stable camera exposure times, controller reference timestamps, the Robotiq action-goal lifecycle and loss metrics before labeling frames.

The validator **always** reports cross_host_clock_alignment_verified=false, tcp_fk_and_frame_semantics_verified=false, arm_cartesian_action_labels_available=false, 7d_training_actions_available=false, lerobot_export_ready=false, sft_ready=false, recap_ready=false. A passing schema test does not override any gate.

## Local offline tests (workstation)

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" python -m pytest -q tests/unit_tests/test_fr3_target_telemetry_contract.py
```

Tests use **synthetic records only**, never connect to ROS, command hardware or access captured images. The isolated implementation test run completed **11 passed**; verify again on the workstation checkout after pulling.

Optional parser for **future correctly instrumented** JSONL (not the existing states.jsonl):

```bash
python -m toolkits.realworld_check.fr3_target_telemetry_contract --jsonl /path/to/new_proposed_telemetry.jsonl
```

## Follow-up

1. Obtain an approved instrumented controller design and compare logged IK target against the actual post-limit q_desired_ at tick level; do not patch or restart the deployed controller during this offline milestone.
2. Choose the exact Cartesian action semantic/frame/horizon and verify FK and tool offsets with controlled tests.
3. Establish NUC/workstation timebase mapping, gripper-goal provenance, and loss accounting before any LeRobot, SFT or RECAP export.
