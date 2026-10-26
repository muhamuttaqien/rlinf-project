# Milestone 1B.1 — ROS 2 Baseline and Robotiq Read-Only Qualification

## 1. Objective

This report qualifies the existing observation path's ROS settings and evaluates activation-free Robotiq feedback on **2026-10-26 JST**. It builds on the [Milestone 1A audit](milestone_1a_audit.md) and [Milestone 1B report](milestone_1b_report.md), preserving the implemented pipeline and distinguishing their October 23 hardware evidence from today's results.

The baseline configuration is documented and the subscription helper starts successfully. **Fresh Franka data reception is PENDING:** today's NUC has no observed Franka controller/state-broadcaster process or publisher for the three required topics. **Robotiq integration remains BLOCKED:** candidate status-reading code exists, but its hardware connection and feedback lifecycle are not qualified. These are separate findings; neither establishes cross-distribution data incompatibility.

Work stayed on `branch-openpi-recap-franka`, starting at `63b79c8f`. The existing untracked `assets/pi05_stackcube_panda_joint` was preserved. No controller, gripper, NUC configuration, camera code, SpaceMouse code, or learning pipeline was changed.

## 2. Previously Verified Baseline

The October 23 run remains evidence that the selected Humble/Jazzy state path can work with these standard message types. It is not a measurement of current hardware availability.

| Component | Prior evidence | Qualification today |
| --- | --- | --- |
| Physical Franka state | VERIFIED on October 23: measured joints, EE pose and external wrench in the unified observation; physical hardware plugin inspected. | Current publisher availability checked separately below. |
| Dual RealSense capture | VERIFIED on October 23: 30 observations, two distinct `(480,640,3)` BGR `uint8` frames; maximum host-receipt skew 46.44 ms. | Implementation unchanged; capture not repeated. Physical base/wrist assignment remains PARTIALLY VERIFIED. |
| SpaceMouse | VERIFIED on October 23: explicit HID path, idle axes/buttons; mapping/lifecycle tested at the SDK boundary. | Implementation unchanged; physical all-axis/button exercise remains PENDING. |
| Focused CPU tests | PASSED previously: 14 tests, 293 deselected. | Not rerun: no production or test code changed. |
| Robotiq | BLOCKED previously: no qualified state source; RLinf connection activates hardware. | Reassessed against both current NUC implementations below. |

The upstream open-PR search for `franka ROS2` again returned no matches. [RLinf issue #759](https://github.com/RLinf/RLinf/issues/759) still overlaps the Python-version split. The existing system-Python helper already addresses that process boundary; this milestone introduces no replacement.

## 3. ROS 2 Environment Findings

The user's baseline was applied only to the qualification processes. The Workstation helper inherited the following explicit environment after sourcing Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=70
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

These settings select domain 70, subnet discovery and Fast DDS for the subscription process. They do not launch a NUC publisher. The existing `RosStateSubscriber.start()` fixes its child domain to 70 and inherits middleware/discovery settings from its caller; it does not set the other two variables itself.

| Process/environment | Status | Observed settings |
| --- | --- | --- |
| Workstation existing `franka_ros2_observer.py` | VERIFIED | `/proc/<pid>/environ`: Jazzy, domain `70`, RMW `rmw_fastrtps_cpp`, discovery range `SUBNET`; helper ready record confirms Jazzy/domain/RMW. |
| Workstation helper DDS libraries | VERIFIED | Process maps load `libfastrtps.so.2.14.6`, `libfastcdr.so.2.2.7`, `librmw_fastrtps_cpp.so` and its shared support library. |
| Workstation discovery overrides | VERIFIED for the sampled helper | `ROS_LOCALHOST_ONLY`, `ROS_STATIC_PEERS`, `ROS_DISCOVERY_SERVER`, both Fast DDS profile-file environment variables and `RMW_FASTRTPS_USE_QOS_FROM_XML` were unset. This records environment overrides, not every possible middleware configuration source. |
| Temporary NUC graph observer | VERIFIED | Humble, domain `70`, RMW `rmw_fastrtps_cpp`, `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`, `ROS_LOCALHOST_ONLY=0`; actual RMW identifier confirmed by `rclpy`. |
| Existing NUC Franka publisher environment | PENDING | No matching controller/launch process was present to inspect. The October 23 process environment cannot be reconstructed from its old PID. The graph observer's settings are not presented as publisher settings. |
| NUC interactive shell | VERIFIED at file level | `.bashrc` still exports domain `42`. Explicit exports for the bounded observer affected only that process. |

Installed package versions were:

| Package | Workstation Jazzy | NUC Humble |
| --- | --- | --- |
| `rcl` | `9.2.11` | `5.3.13` |
| `rmw_fastrtps_cpp` | `8.4.4` | `6.2.10` |
| `fastrtps` | `2.14.6` | `2.6.12` |
| `fastcdr` | `2.2.7` | `1.0.29` |

Jazzy's installed `rcl/discovery_options.h` and [Jazzy rcl implementation](https://github.com/ros2/rcl/blob/jazzy/rcl/src/rcl/discovery_options.c) support `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`. The inspected Humble headers lack that interface; its `librcl.so` contains `ROS_LOCALHOST_ONLY` but neither `ROS_AUTOMATIC_DISCOVERY_RANGE` nor `ROS_STATIC_PEERS`. Setting the new variable on this Humble installation does not configure its discovery scope. Keep Humble's `ROS_LOCALHOST_ONLY=0` for network participation; document Jazzy's subnet option separately. See the [Jazzy discovery documentation source](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Tutorials/Advanced/Improved-Dynamic-Discovery.rst).

No distribution, middleware, network, profile file, shell startup file or daemon was replaced or reconfigured.

## 4. ROS 2 Communication Verification

One bounded concurrent check addressed current availability and the unresolved discovery warning. The unchanged Workstation helper subscribed for **12.217 seconds** while a temporary NUC graph observer inspected endpoints for approximately 10 seconds, both on the stated baseline. Neither process invoked services/actions or created robot command publishers.

| Required topic | Type | Workstation messages | NUC-discovered publishers |
| --- | --- | --- | --- |
| `/franka_robot_state_broadcaster/measured_joint_states` | `sensor_msgs/msg/JointState` | 0 | 0 |
| `/franka_robot_state_broadcaster/current_pose` | `geometry_msgs/msg/PoseStamped` | 0 | 0 |
| `/franka_robot_state_broadcaster/external_wrench_in_base_frame` | `geometry_msgs/msg/WrenchStamped` | 0 | 0 |

The helper emitted its ready record, reported no protocol error and exited cleanly with code 0 after SIGTERM. Readiness proves subscriber initialization, not state availability. The independent NUC process scan likewise found no Franka controller/launch process. The controller observed on October 23 was not started, stopped or replaced by this work.

The NUC graph contained the three topic names because the remote helper subscribed to them. A topic name alone is not evidence of a publisher. Its node list contained the local qualification node and the separate `yubi_gripper_driver`, but omitted the Workstation helper name. NUC stderr emitted `sequence size exceeds remaining buffer`; Workstation helper stderr was empty. This reproduces incomplete node metadata alongside discovery of remote topic endpoints. It does not establish a failure to deserialize the three standard state messages.

**Interpretation:** the current sample has no observed Franka source from which to receive data. It cannot measure how the warning affects an active state stream. October 23's successful physical reception remains valid evidence against declaring Humble/Jazzy universally incompatible; today's warning is not declared harmless or repaired. Restore availability through the existing operator-approved owner before the remaining targeted reception check. Do not start a second controller to obtain evidence.

The subscriber uses `qos_profile_sensor_data`: `BEST_EFFORT`, `VOLATILE`, keep-last depth 5. The October 23 measured-joint endpoint was `RELIABLE`/`TRANSIENT_LOCAL`, compatible with those requested policies. Current NUC broadcaster source creates all three selected publishers using `rclcpp::SystemDefaultsQoS()`. That source declaration is not a fresh endpoint measurement; actual offered QoS for pose/wrench remains PENDING while publishers are absent.

## 5. Robotiq Hardware and Driver Findings

The adapter and available code were inspected without opening the serial port. The user-provided label remains “Robotiq 2F Adaptive Gripper C-49010.” Robotiq's [serial-format reference](https://blog.robotiq.com/knowledge/what-are-the-two-parallel-locking-pins-for-on-the-2f85-and-2f140) identifies `C-4XXXX` as a generation/serial format, not a selection between 85 mm and 140 mm variants.

| Finding | Status | Evidence and limit |
| --- | --- | --- |
| Adapter | VERIFIED | FTDI USB `0403:6015`, serial `DA95C62Y`; stable path `/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DA95C62Y-if00-port0` resolves to `/dev/ttyUSB1`. This identifies the adapter, not the device beyond it. |
| Permissions/ownership | PARTIALLY VERIFIED | Mode `0660`; SSH user belongs to `dialout` and has read/write permission. `fuser` reported no owner at the inspected instant. This is not an exclusive serial lease. |
| Model, wiring, power and actual serial settings | PENDING | Exact mounted variant and remote endpoint configuration were not established. `115200`, 8N1, slave `9` are source defaults, not a hardware readback. |
| Existing Robotiq ROS feedback | BLOCKED | No qualified Robotiq publisher or process was found in the bounded graph/process checks. Yubi and `/openpi/gripper_target` belong to the separate setup and remain excluded. Franka Hand interfaces are not substitutes. |
| Workstation RLinf driver | VERIFIED at source | `rlinf/robotics/parts/end_effectors/grippers/robotiq.py` uses `pymodbus`; `_open()` calls `_activate()`. `_read_status()` is private and needs the connection. Calling `connect()` remains prohibited for this audit. |
| Current NUC RLinf driver | VERIFIED at source | `/home/aist-franka/RLinf/rlinf/robotics/parts/end_effectors/grippers/robotiq.py` is locally modified at checkout `d34d4c32`. It uses PySerial, FC04 feedback, response CRC/slave/header checks and local-echo handling. `_open()` still activates before starting optional background reads. Public `get_status()` does not provide an activation-free connection lifecycle. |
| NUC dependencies | VERIFIED | Existing NUC venv has `pymodbus 3.15.0` and `pyserial 3.5`. No dependency was installed or upgraded. |
| Existing C++ library | PARTIALLY VERIFIED | `~/robotiq/robotiq-gripper-interface-main`: `connect()` opens/configures serial without invoking reset/activation. `get_feedback()` sends a fixed FC03 status request; reset/activation/motion methods are separate. This is a candidate, not a qualified live source. |

The NUC driver differs materially from the Workstation copy. Its SHA-256 at inspection was `9b914b8a01ab2824b4d3800f90150a18e9fc597be59013800dc2386a3a7395ee`; syntax compilation passed without importing it. Its comments claim adapter echo and previous hardware behavior, but those comments were not treated as newly verified hardware results. Neither checkout was synchronized or modified.

Robotiq's [manufacturer control reference](https://assets.robotiq.com/website-assets/support_documents/document/online/2F-85_2F-140_TM-OMRON_InstructionManual_HTML5_20190503.zip/2F-85_2F-140_TM-OMRON_InstructionManual_HTML5/Content/4.%20Control.htm) distinguishes FC03/FC04 status reads from FC16 output writes. The C++ status request `09 03 07 D0 00 03 04 0E` selects three registers starting at `0x07D0`; it is a status read, not activation. A Modbus read still transmits a serial request and requires a qualified endpoint and owner.

The C++ implementation cannot yet supply trusted observation feedback unchanged: `write_read()` reads until timeout or byte `0x0A`, although RTU responses are binary. `get_feedback()` checks response length but does not validate response CRC, slave address, function code or byte count before decoding; errors return a default feedback object. These are concrete feedback-integrity gaps, separate from actuation. The reviewed NUC Python implementation handles several of these checks, but its activation-coupled lifecycle still prevents using it directly here.

**BLOCKED resolution:** confirm the endpoint/settings and sole serial owner, then qualify an explicit activation-free read lifecycle in the existing driver rather than bypassing private fields or copying a second Modbus implementation. Preserve the current command lifecycle as the default. Validate framing, echo handling, exceptions and unavailable-state behavior with SDK/transport-boundary tests before a bounded hardware read. Treat any polling/watchdog effects separately; the NUC `get_status()` implementation explicitly documents that reads can clear communication-timeout fault `0x09`.

No serial connection, Modbus transaction, activation, reset, motion, force command or gripper hardware test was performed.

## 6. Implemented Changes

This milestone adds qualification evidence rather than replacing working sensor code. Production and test sources are unchanged.

| File | Change | Status |
| --- | --- | --- |
| `docs/experiments/openpi_recap_franka/milestone_1b_1_report.md` | Baseline settings, current availability, discovery/QoS findings, driver comparison and remaining acceptance conditions. | IMPLEMENTED |
| `docs/experiments/openpi_recap_franka/robotiq_reference.md` | Link the preparation notes to the dated hardware/driver findings. | IMPLEMENTED |

Temporary qualification scripts and logs stayed under Workstation `/tmp`; the NUC graph script ran over SSH standard input. No persistent NUC file was created. The existing monitor continues to accept an explicitly qualified JointState gripper topic; no unverified source was configured.

## 7. Observation Contract

The existing raw dictionary remains the contract. Review traced the subscription callbacks, cache snapshot and final observation assembly without opening sensors again.

| Field | Current contract and qualification |
| --- | --- |
| `robot.states.joint_state` | Original measured joint positions, velocities and efforts; source header/frame retained, units rad, rad/s, N·m. Missing/stale/stopped input has `source=None`. |
| `robot.states.tcp_pose` | Original stamped EE pose, metres and quaternion x/y/z/w. The source frame is preserved; Robotiq TCP geometry remains unqualified. |
| `robot.states.external_wrench` | Original stamped force/torque and frame, N and N·m. |
| `robot.states.gripper_state` | Remains unavailable with `source=None`; no requested position, cached default or Franka Hand state is substituted. |
| `cameras.base`, `cameras.wrist` | Existing separate real-image arrays, explicit SDK identities, BGR encoding and host receipt timestamps; capture/freshness/pairing code unchanged. |
| `operator_intent` | SpaceMouse axes/buttons remain separate from robot observations/actions; no forwarding or command publisher. |
| `timing` | Source ROS clock, host monotonic receipts and assembly wall clock remain distinct. Receipt skew is not exposure synchronization or measured network latency. |

For a future qualified Robotiq source, retain raw `gPO` measured position separately from `gPR` requested-position echo, status bits, raw fault byte and decoded fault, current, transport identity, validity and receipt time. A metric width needs the confirmed model/fingertips and a documented calibration; the existing linear maximum-width formula is not an independently measured jaw opening. Until calibrated, expose raw position and leave metric width unavailable. Neither driver-derived `is_open` nor a button press proves a successful grasp.

The optional JointState source can retain an approved measured joint coordinate, but cannot by itself carry all native Robotiq fault/status fields. Such fields need a qualified accompanying source before claiming full integration. No new schema or speculative publisher was added. No 19D state projection, 7D action conversion or OpenPI/RECAP transform was changed.

## 8. Focused Test Results

Only checks answering this milestone's open questions were performed. Historical camera/SpaceMouse capture and the 14 CPU tests were not repeated.

| Check actually run on October 26 | Result |
| --- | --- |
| SSH NUC process, adapter, permission, source and package inspection | PASSED: adapter/dependencies confirmed; absence of observed Franka/Robotiq publisher processes documented. |
| Existing Workstation helper, explicit baseline, bounded 12-second run | PASSED initialization/environment/cleanup: ready record, actual Fast DDS libraries, no helper errors, exit 0. |
| Fresh three-topic Franka reception in that run | FAILED availability check: zero messages on all three topics, zero publisher endpoints observed from NUC. Live qualification remains PENDING; this is not classified as a demonstrated transport incompatibility. |
| Concurrent NUC graph/QoS inspection, approximately 10 seconds | PASSED diagnostic collection: remote topic endpoints visible, zero required publishers, omitted remote helper node name; buffer warning reproduced. |
| Installed discovery support inspection | PASSED: Jazzy header/source supports subnet range; inspected Humble library/header exposes the legacy localhost setting. |
| Current NUC Robotiq driver `compile(source, path, 'exec')` | PASSED syntax only; no import, connection or hardware execution. |
| Gripper hardware feedback verification | BLOCKED; not run. |
| Camera/SpaceMouse hardware tests and CPU regression suite | Not rerun; unchanged implementation and prior results retained. |

The existing subscription helper was invoked with only its three topic arguments after exporting the baseline; it was terminated by the temporary parent after the bounded interval. The NUC observer created a graph-only `rclpy.Node` with rosout and parameter services disabled. It queried publisher metadata and sensor-data QoS, then destroyed only its own node. No ROS daemon was started/stopped and no controller service was called.

Two initial inspection limitations were corrected: an SSH sandbox socket denial was followed by an approved read-only host probe, and unavailable NUC `rg` was replaced by standard tools. They were not hardware failures. Official documentation pages/PDFs that were inaccessible were replaced with readable ROS source and the manufacturer's HTML manual; no unverified third-party register map was used.

Documentation received a focused `docs-check`/`refine-docs` review against the unchanged implementation. Relative-link validation PASSED for all seven local targets across the two changed Markdown files; the report's ten requested sections were also checked. Sphinx, RST markup and EN/ZH build harnesses were not run: only English experiment Markdown changed, consistent with the requested scope.

## 9. Remaining Blockers

Current acceptance separates documentation/review results from unavailable live evidence.

| Acceptance item | Status | Required evidence |
| --- | --- | --- |
| Existing state reception under documented baseline | PARTIALLY VERIFIED | October 23 actual reception and October 26 helper baseline are established separately. Obtain a fresh three-topic sample plus publisher environment/QoS once the existing approved owner is available. |
| Preserve ROS distributions/middleware | VERIFIED | Existing Jazzy/Humble and Fast DDS retained. |
| Robotiq safely integrated or evidence-based BLOCKED result | BLOCKED, documented | Confirm endpoint/settings/ownership; qualify a driver read lifecycle and validated responses. Exact model/calibration is additionally required for metric width. |
| Dual-camera/SpaceMouse implementation preserved | VERIFIED by change scope | Production sources unchanged; latest physical evidence remains October 23. |
| No robot/gripper commands issued | VERIFIED for this work | Only source/metadata inspection, graph queries and subscriptions were executed. |

The missing publisher prevents fresh ROS reception acceptance today. This does not invalidate Milestone 1B's dated run or make the entire qualification report a failure. Physical camera-role confirmation, deliberate HID exercise and timing calibration remain prior pending items; this milestone does not expand into those hardware exercises.

## 10. Recommended Next Milestone

First finish the two remaining read-only qualifications. Have the operator restore or identify the existing approved Franka publisher without changing controller ownership, then run one bounded three-topic subscription and capture its actual publisher environment/QoS. Keep node-discovery warning evidence separate from receipt counts and source freshness. Confirm Robotiq model and serial endpoint/settings, then qualify the existing driver's activation-free feedback lifecycle with explicit unavailable values on communication failure.

These steps precede a future full raw-feedback integration. Metric gripper calibration, TCP/load, command semantics, operator enable, ownership and command expiry remain prerequisites for a separately authorized control milestone. Teleoperation, gripper actuation, dataset collection, OpenPI SFT and RECAP were not started.

This report was prepared with an AI assistant. **Milestone 1B.1 work stops at this qualification record; fresh arm reception is PENDING and Robotiq feedback is BLOCKED.**
