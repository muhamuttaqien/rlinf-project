# FR3 Mapping Calibration Operator Procedure

Prepare and inspect one bounded calibration case at a time, then review its evidence against the [17-check qualification checklist](pi05_physical_mapping_qualification.md). This interface is verified offline only. All physical checks remain pending, and whether scaling 1.0 resolves weak movement remains a physical question. [中文](pi05_calibration_operator.zh.md)

## Prepare the Installation Evidence

Start from the already running, operator-approved controller. These tools never launch, activate, restart, tune or recover it. Read the existing [controller investigation](fr3_controller_investigation.md) and [pending record](pi05_mapping_verification.pending.json) first. The previous +X reports establish receipt/IK/reference activity but lack a resolved, synchronized physical direction trace. The archived Release build is not evidence of live deployment. Reuse any subsequently collected valid trace before requesting another movement.

Copy the actual mapped controller library, selected launch/config/URDF, captured runtime parameters, mounting and tool-calibration evidence into a local evidence directory. Match `/proc/<actual-pid>/maps` and the live configuration manually; copying or hashing files does not inspect the current NUC process. Create the configuration manifest from those real files:

```bash
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence config-manifest \
  --output /tmp/fr3_calibration/config_manifest.json \
  /tmp/fr3_calibration/controllers.yaml \
  /tmp/fr3_calibration/live_parameters.yaml \
  /tmp/fr3_calibration/robot.urdf \
  /tmp/fr3_calibration/tool_calibration.json
sha256sum /tmp/fr3_calibration/copied_controller.so /tmp/fr3_calibration/config_manifest.json
```

Create `/tmp/fr3_calibration/deployment.json` from the [deployment template](pi05_calibration_deployment.example.json). Its null fields and false attestations intentionally block execution. Enter the real operator, frame names, diagnostic logger, Robotiq joint/bridge identity and domain-71 action status topic. Fill the two artifact hashes from the command above. Specify operator-approved translation and rotation guards, each positive and at most **0.01 m / 0.01 rad**. The translation cap preserves the existing calibration cap; the new rotation guard adds a software bound without changing controller limits.

All three transforms use `[x,y,z,qx,qy,qz,qw]`: `base_to_pose` maps the controller base to the pose topic's header frame; `ee_to_flange` maps the reported EE to the controlled flange; `ee_to_tcp` maps that EE to the Robotiq peg TCP. Determine them from the actual installation and calibration. The runtime checks `f_t_ee * ee_to_flange` against identity and monitors EE, flange and TCP travel. It cannot authenticate physical tool geometry. Review historic errors, clear the workcell, check the approved hardware stop and match the live installation before setting the corresponding attestations true. Preserve gains, limits, weights and SpaceMouse configuration.

## Preview and Check Without Publishing

The default command prints the exact request slots and imports no ROS. Preview every proposed case with it:

```bash
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_calibration \
  --case linear_x_positive
```

The five 100 ms arm slots use a single signed axis with magnitudes `0.030, 0.024, 0.018, 0.024, 0.030`; all other axes remain zero. Distinct slots let the sender trace expose incorrect held-sample timing. They are normalized requests with no additional gain, HID mapping or SI conversion. The maximum remains the proposed 0.03 request. CLI bounds retain the existing maximum magnitude 0.05 and duration 0.5 s; duration must be a whole number of 100 ms slots. Gripper open/close needs at least three samples to produce a goal. Never increase a request or repeat a case automatically after weak motion.

For the later operator session, source Jazzy and the matching message definitions, including `franka_msgs` and `controller_manager_msgs`, in a fresh workstation shell. Stop competing publishers through their normal owning interfaces. The following preflight subscribes and calls only the read-only controller-list service; it creates **zero command publishers**:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=70 ROS_LOCALHOST_ONLY=0
PYTHONPATH=. /usr/bin/python3 -m toolkits.realworld_check.fr3_mapping_calibration \
  --case linear_x_positive \
  --deployment /tmp/fr3_calibration/deployment.json --preflight
```

Require one correctly typed subscriber on each command topic, no competing writers, fresh pose/joints/full robot state and gripper position, an active joint-impedance controller claiming seven effort interfaces, safe robot mode/current errors, and two fresh diagnostic reports. `FR3_DIAG_V1` must include the existing joint-error/PD/total-effort peak fields. Missing message definitions, diagnostics or telemetry block execution; the tool does not install, activate or modify anything to fix them. A command-topic recorder adds a subscriber and fails the strict graph check; capture sender events and controller diagnostics instead.

## Execute One Case Only in a Separately Approved Session

Begin with +X and review its measured response before any further direction. The following is a future operator command, **not executed during preparation**:

```bash
PYTHONPATH=. /usr/bin/python3 -m toolkits.realworld_check.fr3_mapping_calibration \
  --case linear_x_positive \
  --deployment /tmp/fr3_calibration/deployment.json \
  --output /tmp/fr3_calibration/x_positive_01 --execute
```

The terminal displays a fresh `CALIBRATE <case> <digest>` token bound to this plan and deployment. Read the preview and match the installation before typing it exactly. Piped approvals are rejected. Confirmation covers one bounded case and its neutral cleanup, including 100 Hz refreshes; every new movement or gripper case requires a new interactive confirmation. A local exclusive lock and live DDS checks prevent competing calibration ownership.

After confirmation the tool reacquires at least 2.1 s of continuous fresh baseline, then executes up to 0.5 s of requests and at least 2.1 s of neutral observation. ENTER/EOF, SIGINT or SIGTERM revokes new motion/goals; pressing ENTER during the active sample collects the stop transition. Abort paths attempt five neutral arm publications independently of evidence storage. Neutral holds the last joint reference and may permit residual tracking. It does not cancel an in-flight gripper goal or replace the approved hardware stop. An in-flight middleware publication cannot be recalled.

The loop rechecks ownership, telemetry, full orientation and tool geometry. Pose/joints/robot/gripper state must be no older than 200 ms; controller-list status expires after 1 s and diagnostics after 2 s. Action status uses its event-stream QoS rather than a periodic heartbeat; post-goal acceptance/completion events still require review. A transport gap above 50 ms or an expired 100 ms slot aborts instead of catching up. Evidence storage uses a bounded queue; overflow or write failure aborts while neutral cleanup remains independent. The tool never retries a fault or restarts a case.

Use the following single-case names after resolving +X; issue separate commands and confirmations, with a new output directory for each. At mounting RPY `[0,0,0]`, linear directions follow the controller base; angular directions use the current flange's right-hand axes. Pure rotation can translate the peg through its lever arm. The default maximum IK offsets are 0.21 mm and 0.0009 rad (0.052°), not guaranteed physical travel. Predeclare measurement resolution, off-axis tolerances and acceptable residual motion; unresolved response is inconclusive.

| Axis | Positive case | Negative case |
| --- | --- | --- |
| Base X | `linear_x_positive` | `linear_x_negative` |
| Base Y | `linear_y_positive` | `linear_y_negative` |
| Base Z | `linear_z_positive` | `linear_z_negative` |
| Flange X | `angular_x_positive` | `angular_x_negative` |
| Flange Y | `angular_y_positive` | `angular_y_negative` |
| Flange Z | `angular_z_positive` | `angular_z_negative` |

`gripper_open`, `gripper_close` and `gripper_hold` are separate cases with a neutral arm. Open/close publishes one goal after three confident 10 Hz samples; stable latched samples do not repeat goals, and hold uses request 0.5. Arm refreshes add no gripper confirmations. Baseline/post observation separates sequential runs by more than the existing 0.8 s goal interval. Record the domain-71 action status and measured joint; bridge publication alone is not acceptance. Inspect independent Robotiq stop/completion behavior. The `neutral` case only publishes zero arm requests; it cannot by itself qualify stopping from motion.

## Inspect and Review Genuine Measurements

Each run creates `session.json`, `events.jsonl`, `telemetry.jsonl`, `summary.json`, copied installation inputs and a hashed `manifest.json`. The capture preserves the controller's `/rosout` diagnostics, pose, measured/desired joints, full robot state, read-only controller status and domain-71 gripper state/status. Full robot-state evidence is sampled at up to 100 Hz, retaining a receipt counter; it is not a complete 1 kHz recording. Host monotonic receipts align local streams; ROS header clocks remain separate.

```bash
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence verify-session \
  /tmp/fr3_calibration/x_positive_01
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence analyze \
  /tmp/fr3_calibration/x_positive_01
```

These commands verify hashes and report actual sender intervals, signed frame motion, counter deltas and peaks. They **never pass physical checks automatically**. Fake captures cannot approve mapping. `FR3_DIAG_V1` provides aggregate receipts and independent maxima; it does not measure per-command controller receipt gaps, accepted hardware torque or exact request-to-reference correspondence. Keep `timing_10hz_100hz` pending until the operator has adequate genuine controller-side timing evidence. The utility adds no controller instrumentation to supply missing evidence.

Prepare a real review from the [review template](pi05_calibration_review.example.json). Bind it to the analysis's session-manifest and scope hashes, name exactly one of the 17 checks, record actual measurements/criteria, a past UTC review time after the capture, notes, and hashed nonempty attachments. Choose `pass` only after a physical review; otherwise use `inconclusive` or `fail`. Scope mismatches and altered nested files are rejected. One valid trace can support several separately reviewed checks, including a signed direction and an operator stop, without repeating motion.

For `gripper_open_close`, also provide `counterpart_session_manifest: {"path": "close_session/manifest.json", "sha256": "<actual hash>"}` pointing to the complete other case. Both sessions must be genuine ROS captures with the same installation scope, measured/status data and goals separated by at least 0.8 s on the same host boot clock. Attach acceptance/completion evidence for both goals. For frames, attach actual TCP calibration; for acceptance, compare measured/desired joint effort and mode/errors with diagnostics and motion; for neutral/stop, attach residual-motion and approved hardware-stop evidence.

Apply one review to a new partial record with another explicit `REVIEW <check> <digest>` confirmation:

```bash
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence apply-review \
  --record docs/experiments/openpi_recap_franka/pi05_mapping_verification.pending.json \
  --session /tmp/fr3_calibration/x_positive_01 \
  --review /tmp/fr3_calibration/x_positive_review.json \
  --output /tmp/fr3_calibration/mapping_partial_01.json
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence audit-record \
  /tmp/fr3_calibration/mapping_partial_01.json
```

The new record has a portable evidence bundle; later reviews use the latest partial record. Existing outcomes and evidence files are never overwritten. Partial records retain absent final approval even when an individual reviewed check passes. Only after all 17 genuine reviews can the final operator enter `APPROVE MAPPING <digest>`:

```bash
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence finalize \
  --record /tmp/fr3_calibration/mapping_partial_17.json \
  --operator '<actual operator name>' \
  --output /tmp/fr3_calibration/mapping_verified.json
.venv/bin/python -m toolkits.realworld_check.fr3_mapping_evidence verify-record \
  /tmp/fr3_calibration/mapping_verified.json
```

This validates completeness and transitive integrity; it cannot authenticate a person or certify physical safety. Finalizing enables only the record gate. One later scaling-1.0 rollout still requires live installation matching and its own `RUN VERIFIED <digest>` consent, using the command in the qualification checklist. Stop and review that single outcome before collecting more rollouts.

## Offline Verification Boundary

Run `test_fr3_mapping_calibration.py` and `test_fr3_mapping_evidence.py` in separate CPU pytest processes, plus the existing request-adapter, executor, outcomes, deployment-audit and X-pulse suites. The new tests drive the real state machine and ROS publication methods at fake hardware boundaries. Synthetic review fixtures stay in pytest temporary directories and never become physical records. The branch omits upstream CI workflows; restore path-filter coverage before upstream submission. An AI assistant prepared and tested this interface. No controller operation, physical movement, new inference, gain/limit/weight change or SpaceMouse edit was performed.

The completed offline run passed **379 tests across nine component files**, plus two fake-hardware cases using actual Jazzy message classes under Python 3.12. EN/ZH builds, local links and command parity, Ruff lint/format and whitespace checks passed. The global docs check retains 24 existing symbol findings and 61 RST warnings (zero errors) in unchanged pages. See the [offline evidence record](pi05_calibration_offline_evidence.json) for source hashes and limits.
