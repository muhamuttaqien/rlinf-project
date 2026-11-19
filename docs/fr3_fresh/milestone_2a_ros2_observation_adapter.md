# Milestone 2A — FR3 ROS 2 Observation Adapter

Read measured FR3 and Robotiq 2F-85 state through native RLinf robot parts while
the existing ROS controllers retain hardware ownership. This milestone implements
subscription-only attachment, local IPC, validation, freshness diagnostics, and a
bounded live check. An AI assistant performed the implementation and verification
for the human submitter on 2026-11-19.

## Baseline and Interface Decision

Work used only `branch-openpi-recap-franka-fresh` in `rlinf-project-fresh`, starting
with a clean working tree and the completed
[Milestone 1 audit](milestone_1_hardware_interface_audit.md). No previous
experimental branch, implementation, bridge source, or documentation was read,
compared, copied, or reused. The required read-only open PR and issue searches on
`RLinf/RLinf`, using `FR3 ROS observation`, returned no matches.

The inspected native contracts were `Connection`, `RobotPart`, `PartGroup`,
`ArmState`, `BaseArm`, `FrankaRobotState`, `BaseGripper`, and `FrankaRobot`:

| Contract | Decision |
|---|---|
| `Connection` | Reuse deferred connect, disconnect, rollback hooks, and native placement metadata. |
| `RobotPart` | Subclass for observations without action or control methods. |
| `PartGroup` / `FrankaRobot` | Compose these parts directly; retain native per-owner snapshot reads and independent lifecycles. |
| `BaseArm` / `FrankaRobotState` | Reuse canonical measurement names, but avoid their full-state defaults. TCP velocity and Jacobian are unavailable. |
| `BaseGripper` | Its metre-based opening-width and motion contract cannot represent uncalibrated raw angle/normalized closure faithfully. Use a read-only part. |
| `FrankaRobot.build()` / `FrankaEnv` | Leave motion backends and task paths unchanged; bypass them for observation checks. |

The adapter does not register as a controllable arm or gripper backend. This keeps
the read-only interface explicit and prevents its accidental selection by a
motion-capable environment. Direct `FrankaRobot(arm=..., end_effector=...)`
composition uses the existing robot tree without an arm IP or serial connection.

## Implementation and Observation Contract

| File | Responsibility |
|---|---|
| `rlinf/robotics/parts/ros2_observation.py` | `FR3ROS2Observation` and `RobotiqROS2Observation`; native parts, selected state, completeness, and receipt freshness. |
| `rlinf/robotics/parts/transports/ros2/observation.py` | `ROS2ObservationTransport`; compatible worker launch, bounded IPC, timeout, diagnostics, and cleanup. |
| `rlinf/robotics/parts/transports/ros2/worker.py` | Standalone ROS subscriber process with fixed state-topic profiles. |
| `rlinf/robotics/parts/transports/ros2/protocol.py` | Versioned JSON protocol and standard-message validation; standard library only. |
| `toolkits/realworld_check/observe_fr3_ros2.py` | Bounded live check using only the observation parts. |
| `tests/robot_mocks/ros2.py` | Process-edge ROS SDK fake, with no control API or live network connection. |
| `tests/unit_tests/test_robotics.py` | Adapter tests alongside the native component tests. |

Each part exposes `connect()`, `get_state()`, `get_observation()`,
`wait_for_observation(timeout=...)`, and `disconnect()`. `reset()` inherits the
native no-op. Constructor settings include explicit domain ID, compatible Python
and ROS setup paths, startup/IPC timeout, and maximum receipt age. Native
`is_connected` describes an open handle; `ros2.bridge_alive` and per-stream
`connected` report process and publisher state separately.

`get_observation()` returns `state` and `ros2` dictionaries. The arm state contains
seven joint positions, velocities, and efforts ordered by explicit
`fr3_joint1`–`fr3_joint7` names; measured pose in metres plus quaternion x,y,z,w;
and separate base/stiffness external force/torque estimates. The stiffness values
use native `tcp_force`/`tcp_torque` names with their actual headers retained.
Neither wrench is transformed or reinterpreted as a direct force sensor reading.

The gripper reports raw `knuckle_position_rad` and `normalized_closure`
(0=open, 1=closed) independently. It filters the two domain-71 `/joint_states`
representations by exact joint name and ignores unrelated joint messages.
Driver-supplied zero velocity is not exposed as measured velocity. Custom typed
status, calibrated opening width, and grip force remain explicitly unavailable.

Every stream retains the ROS header's `sec`, `nanosec`, and `frame_id`, a separate
`received_monotonic_ns`, and a local sequence counter. Data dimensions, finite
values, unique/required joint names, unit quaternions, and normalized range are
validated. Empty standard JointState arrays are missing fields; invalid values
clear the affected stream rather than preserving an apparently valid reading.
Missing values are `None`, not manufactured zeros. Desired/reference topics are
excluded from the fixed subscription profile.

`ros2.complete` means **all fields in the selected subset** have valid, fresh
samples and matching publishers. It does not mean the native control state is
complete. Missing, invalid, stale, and permanently unavailable fields are named
separately. Stale samples remain visible with explicit flags; callers must inspect
completeness before using them. Receipt age uses only the host monotonic clock;
it is neither ROS source age nor acquisition latency. Samples from different
streams are independently received, not synchronized.

## ROS Process and IPC Ownership

Each part owns one child process. The default worker uses `/usr/bin/python3`
(Python 3.12) after sourcing `/opt/ros/jazzy/setup.bash`. The RLinf process uses
Python 3.11. Python path/virtual-environment variables are removed before the child
sources ROS; `rclpy` and message imports occur only inside that worker. No foreign
binary package was installed into RLinf. Each worker receives both explicit
`ROS_DOMAIN_ID` and an explicit rclpy context domain.

Workers verify publisher types before subscribing and recheck discovery every
0.5 seconds, including late publisher arrival. Subscriptions use best effort,
volatile durability, and keep-last depth 5, following the verified audit probes.
Only one decoded sample is retained per logical stream. The shared gripper topic
has one subscription and two independently selected retained readings.

A private inherited Unix `SOCK_SEQPACKET` socket carries version-1 JSON, capped at
64 KiB per packet. Requests are serialized; `snapshot` returns current retained
state, `shutdown` requests exit. There is no unsolicited state stream or growing
message queue. Parent stderr retention is capped at 16 KiB. Startup and individual
IPC requests default to a five-second timeout. Protocol/IPC failure closes the
session and reports diagnostics. Reconnect is explicit via disconnect/connect.

Shutdown requests close the executor, node, context, socket, and process. A bounded
terminate/kill/reap fallback handles an unresponsive worker. Startup failures roll
back owned resources. Parent EOF also exits the worker. Temporary ROS log
directories are removed on close.

The worker creates no command publishers, action clients, service clients, or
hardware-control handles. Standard rclpy node infrastructure can expose parameter
events and local type-description services; these are distinct from application
control endpoints. No existing controller, driver, bridge, serial session, FCI
connection, or machine-wide Ray process was changed or stopped.

## Verified Live Interfaces and Results

A fresh, bounded workstation discovery on domains 70 and 71 independently
confirmed the expected publisher types. This did not read controller source or
reuse the existing cross-domain bridge.

| Domain | Selected Topic | Verified Type | Live Frame |
|---|---|---|---|
| 70 | `/franka/joint_states` | `sensor_msgs/msg/JointState` | `base_link` |
| 70 | `/franka_robot_state_broadcaster/current_pose` | `geometry_msgs/msg/PoseStamped` | `base` |
| 70 | `/franka_robot_state_broadcaster/external_wrench_in_base_frame` | `geometry_msgs/msg/WrenchStamped` | `base` |
| 70 | `/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame` | `geometry_msgs/msg/WrenchStamped` | `fr3_link2` |
| 71 | `/joint_states`, raw knuckle | `sensor_msgs/msg/JointState` | `base_link` |
| 71 | `/joint_states`, normalized closure | `sensor_msgs/msg/JointState` | Empty header frame |

These publishers advertised reliable/transient-local QoS. The best-effort/volatile
subscriptions received all six selected streams. Mixed Humble/Jazzy discovery
still reports unknown remote node names, as recorded in Milestone 1. Publisher
identity attribution beyond domain/topic/type/name selection remains unverified.

A preliminary depth-1 check returned one incomplete gripper observation in 227
reads per device. The final implementation uses the audit's depth 5. That short
preliminary result does not establish the cause of the incomplete reading.

The final five-second check used:

```bash
PYTHONPATH="$PWD" RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET \
  python toolkits/realworld_check/observe_fr3_ros2.py --duration 5
```

This attaches only to existing state publishers, waits for valid initial state,
reads both parts, reports metadata, and disconnects in `finally`. The command
returned status 0, with **228 reads per device and zero incomplete observations**.
Maximum sampled local callback-receipt ages were:

| Stream | Maximum Receipt Age |
|---|---|
| FR3 joints | 1.503 ms |
| FR3 pose | 1.901 ms |
| Base wrench | 3.111 ms |
| Stiffness wrench | 3.043 ms |
| Raw gripper | 2.516 ms |
| Normalized gripper | 2.460 ms |

These are sampled receipt ages on this workstation, not latency bounds, loss
measurements, hardware refresh rates, or load qualification. A preliminary
system-Python/worker check also verified that `rclpy` was absent from the RLinf
parent's imported modules. All workers exited after their bounded checks.

## Offline Verification

The CPU regression command used the existing RLinf Python 3.11 environment:

```bash
PYTHONPATH="$PWD:$PWD/tests/unit_tests" \
  pytest -q tests/unit_tests/test_robotics.py -m 'not placement' --disable-warnings --tb=short
```

Result: **319 passed, 8 deselected**. The 27 new cases cover native composition,
shuffled joints, separate gripper representations, source/receipt timestamps,
frame retention, unavailable fields, partial and invalid data, late discovery,
publisher loss, staleness, startup failure, IPC timeout, process death, parent EOF,
explicit reconnect, configuration validation, and packet size/version bounds.
Only the external ROS SDK is faked; the parts, subprocess worker, transport, and
protocol are real. The eight native scheduler placement tests were excluded;
remote placement, accelerator tests, motion e2e, collection, and training were
not run. No `ray stop` was issued.

The first test invocation failed because this sandbox prevents socket operations
on inherited child descriptors, and a test helper used the wrong import path.
After correcting the helper import, the tests passed with local socket access.
The initial documentation build similarly encountered a sandbox restriction in
the sitemap extension's local multiprocessing manager; builds use local socket
access for verification.

Final checks passed:

| Check | Result |
|---|---|
| Adapter tests after the startup-cleanup adjustment (`pytest ... -k ros2`) | 27 passed, 300 deselected |
| `ruff check --preview` on all changed Python files | Passed |
| `ruff format --check` on all changed Python files | Passed |
| `python tests/unit_tests/check_missing_init.py` | Passed |
| `build_docs.py --venv /tmp/fr3-m2a-docs-venv` | English and Chinese: zero warnings under the repository harness's default autodoc filtering |
| `check_rst_markup.py` on both guides and changed indexes | No findings |
| `check_doc_symbols.py` on both guides | Passed, allowing the independently verified live names `fr3_joint1`, `fr3_joint7`, and `fr3_link2` |
| `git diff --check` | Passed |

The English and Chinese usage guides are linked from their Launch & Scale indexes.
Documentation commands, public APIs, field names, and topic semantics were
cross-checked against the implementation and live evidence. No doc-code or EN-ZH
consistency issues found in checked scope. No CI workflow/filter file exists in
this fresh checkout, so there is no existing path filter to extend here.

## Remaining Limits

The selected subset is usable for read-only inspection. It deliberately leaves
measured TCP velocity, Jacobian, calibrated gripper opening width, grip force, and
custom gripper fault/object/current status outside this milestone. Their absence
is surfaced in `unavailable_fields`; it cannot be interpreted as zero or healthy
status. Joint effort remains measured joint torque, not TCP wrench.

Frame equivalence, stiffness-frame interpretation, tool calibration, physical
jaw width, source-clock alignment, cross-stream skew, and sustained freshness
under training load still need independent qualification. Standard best-effort
subscriptions may drop messages; fixed buffers and explicit incomplete/stale
results are the intended behavior. Fresh headers do not prove fresh serial-device
measurements.

No motion, control publisher, action goal or cancellation, state-changing service
call, activation, reset motion, controller switch, recovery, serial/FCI connection,
camera start, data collection, or training was performed. The existing
motion-capable `FrankaEnv` was neither instantiated nor modified. No commit,
push, PR, issue, or review comment was created.

See the [English usage guide](../source-en/rst_source/guides/fr3_ros2_observation.rst)
and [Chinese usage guide](../source-zh/rst_source/guides/fr3_ros2_observation.rst)
for the public API and runnable observation check.
