# Milestone 1B — ROS 2 Integration and Observation Pipeline

## 1. Objective

This report describes the implemented read-only Workstation observation pipeline and its verification on **2026-10-23 JST**. The Workstation received physical Franka feedback, both RealSense images, and SpaceMouse input in one raw observation. **Robotiq feedback remains BLOCKED**, and this milestone does not establish readiness for active teleoperation, dataset collection, SFT, or RECAP.

Work remained on `branch-openpi-recap-franka`. The initial head was `07a8acdc`; a separate human commit, `23838897` (“Complete Milestone 1A”), appeared during this work and was preserved. The existing asset symlink was not staged or changed. No changes were made to `main`.

| Status | Meaning |
| --- | --- |
| IMPLEMENTED | Source changes exist and were checked. |
| VERIFIED | Directly exercised within the stated scope. |
| PARTIALLY VERIFIED | A working path was observed with remaining physical or timing qualifications. |
| PENDING | Not established or exercised. |
| BLOCKED | A prerequisite or permitted safe path is missing. |

## 2. Existing Components Reused

The pipeline composes existing RLinf sensor readers and standard ROS subscriptions. It does not construct `FrankaRobot` or `FrankaEnv`, whose connection/reset lifecycles can control hardware.

| Component | Reuse |
| --- | --- |
| `Camera.of(CameraInfo(...))` / `RealSenseCamera` | Two explicitly identified color streams, existing capture threads, freshness and cleanup. |
| `SpaceMouse` / `SpaceMouseExpert` | Existing six-axis mapping and button readings; the monitor never invokes `action()` or publishes intent. |
| NUC `franka_robot_state_broadcaster` | Existing physical-arm feedback. No replacement publisher or robot driver was written. |
| ROS standard message definitions and `message_to_ordereddict` | Preserve message fields, header stamps, frame IDs, and units. |
| `filelock`, `ExitStack`, subprocess pipes | Existing dependency and standard-library ownership/cleanup; no network service or new data framework. |

The upstream search found no open PR matching the narrow query `franka ROS2`. [RLinf issue #759](https://github.com/RLinf/RLinf/issues/759) overlaps the ROS/learning Python version split. Relevant source was compared with [RLinf](https://github.com/RLinf/RLinf), [Franka ROS 2](https://github.com/frankarobotics/franka_ros2/tree/humble), [Franka SpaceMouse](https://github.com/frankarobotics/franka_spacemouse), and the local implementations referenced by the [Milestone 1A audit](milestone_1a_audit.md). OpenPI and RECAP implementations remain unchanged.

## 3. Implemented Changes

The monitor separates local capture from the ROS runtime because RLinf uses Python 3.11 while the Workstation's installed Jazzy client uses Python 3.12.

| File | Change |
| --- | --- |
| `toolkits/realworld_check/franka_observation.py` | `FrankaObservation.connect/get_observation/disconnect`, bounded CLI monitor, raw observation assembly, receipt-age checks, frame pairing, and metadata summary. |
| `toolkits/realworld_check/franka_ros2_observer.py` | Subscription-only system-Python helper; passes JSON state records to the parent through stdout. ROS logs remain on stderr. |
| `rlinf/robotics/parts/cameras/base.py` | Exposes existing capture-thread receipt timestamps through `get_frame_with_timestamp()`. Existing `get_frame()` and ordinary observations keep their return contracts. |
| `rlinf/robotics/parts/teleop/spacemouse.py` | Adds optional `device_path`, backed by the SDK's `open_by_path()`. The default index path remains available; conflicting path/nonzero-index selection is rejected. |
| `tests/unit_tests/test_franka_observation.py` | Tests raw message preservation, cache isolation, stale/missing/stopped state, and distinct camera identities. |
| `tests/unit_tests/test_robotics.py` and `tests/robot_mocks/teleop.py` | Tests camera receipt timestamps and path-selected SpaceMouse axes/buttons using the existing vendor SDK fakes. |
| `AGENTS.md` | Makes CI workflow guidance conditional on workflow presence. This experiment branch contains no `.github/workflows` tree to extend. |

The Workstation repository venv received `pyrealsense2==2.58.4.10922`, `pyspacemouse==2.0.0`, and its `easyhid-ng==0.1.0` dependency. These are peripheral dependencies already required by RLinf's Franka environment; the learning stack was not upgraded. No NUC package, configuration, or safety parameter was changed by this work.

The monitor takes an advisory lock for the explicit HID path and checks existing device owners with `fuser` before opening it. These checks prevent a second cooperating monitor and reject an already occupied device; they are not a system-wide mandatory lock against arbitrary external programs. Startup failure unwinds owned resources. Shutdown closes the cameras, HID reader, and only the ROS helper subprocess it created.

## 4. ROS 2 Integration Results

**VERIFIED:** Jazzy on the Workstation received the existing Humble broadcaster's standard state messages on domain 70. This establishes the selected data path, not universal cross-distribution compatibility or a repair of all discovery issues from Milestone 1A.

| Source | Message type | Observation |
| --- | --- | --- |
| `/franka_robot_state_broadcaster/measured_joint_states` | `sensor_msgs/msg/JointState` | Seven named FR3 joints, measured positions/velocities/efforts. |
| `/franka_robot_state_broadcaster/current_pose` | `geometry_msgs/msg/PoseStamped` | EE position and quaternion with header frame `base`. |
| `/franka_robot_state_broadcaster/external_wrench_in_base_frame` | `geometry_msgs/msg/WrenchStamped` | External force and torque with header frame `base`. |

The NUC endpoint query reported one joint-state publisher: `franka_robot_state_broadcaster`, namespace `/`, reliability `RELIABLE`, durability `TRANSIENT_LOCAL`. The subscriber uses the standard sensor-data QoS (`BEST_EFFORT`, `VOLATILE`) and received updates successfully. Source timestamps were preserved alongside Workstation wall-clock and monotonic receipt timestamps.

The final snapshot's host-receipt ages were approximately **0.70 ms** for joints, **1.08 ms** for pose, and **0.94 ms** for wrench. These are cache ages at observation assembly, not network latency measurements. Source-clock synchronization and end-to-end latency remain unqualified.

The helper explicitly uses domain 70 for its own process. It does not edit the NUC's interactive domain-42 shell setting, restart a ROS daemon, invoke services/actions, or create a robot command publisher.

## 5. Franka and Robotiq State

The arm feedback source became available independently during implementation. The monitor subscribed to it and left its control ownership unchanged.

| Finding | Status | Evidence |
| --- | --- | --- |
| Physical feedback provenance | VERIFIED | Read-only `robot_description` query names `franka_hardware/FrankaHardwareInterface`, `arm_id=fr3`, `robot_ip=10.42.0.73`; no fake hardware plugin. |
| Existing owner | VERIFIED | NUC launch PID 462396 started at 17:49:14 JST: `franka_arm_controllers joint_impedance_ik_controller.launch.py robot_config_file:=aist_fr3_config.yaml`; controller manager PID 462399. This work did not start it. |
| Existing controllers | VERIFIED | `joint_state_broadcaster`, `franka_robot_state_broadcaster`, and `joint_impedance_ik_controller` were active. The seven joint effort interfaces were claimed; position/velocity interfaces were unclaimed. |
| Arm state subscription | VERIFIED | Seven joint position/velocity/effort values, EE pose and wrench arrived through the new helper during real camera capture. |
| Measured Cartesian twist | PENDING | The inspected broadcaster exposes desired/commanded EE twist. Neither is relabeled as measured velocity; measured joint velocities are retained. |
| Robotiq read-only source | BLOCKED | No verified Robotiq state publisher was found. `RobotiqGripper.connect()` activates the device; no such connection or Modbus transaction was attempted. |
| Gripper identity/calibration | PARTIALLY VERIFIED | User label “Robotiq 2F Adaptive Gripper C-49010”; 85/140 variant, physical width mapping and TCP/load remain unqualified. Yubi remains excluded. |

A proposed bounded broadcaster-only Franka bringup was **rejected by automatic approval review before execution** because the physical hardware lifecycle/connection could have actuation or safety effects. It was not retried or executed indirectly. The independently running publisher subsequently made another bringup unnecessary for this milestone.

Read-only source inspection also confirmed that adding `/opt/openrobots/lib` to a temporary loader environment resolves the inspected Pinocchio dependencies under `ldd`. This was a dependency inspection, not a persistent NUC change or a controller startup by this work.

## 6. Camera and SpaceMouse Results

**VERIFIED:** both cameras delivered distinct real images in the same observations. The SDK serial namespace differs from the USB-descriptor namespace recorded in Milestone 1A.

| Camera/model | SDK serial used by RLinf | USB descriptor serial | Color node |
| --- | --- | --- | --- |
| D435i, configured as base | `337122075473` | `327743060094` | `/dev/video6` |
| D405, configured as wrist | `218622277827` | `323743071035` | `/dev/video10` |

SDK physical-port metadata matched the corresponding USB/V4L devices. Firmware versions were D435i `5.17.3.10` and D405 `5.17.0.10`, both reporting USB 3.2. Use SDK serials with the existing RLinf RealSense backend. The physical base/wrist assignment is still **PARTIALLY VERIFIED**, based on the user's long-camera-at-base description; the CLI requires explicit serial arguments.

The final 30-observation run took **3.72 seconds** after startup. Each observation contained two `uint8` BGR frames of shape `(480,640,3)`. Both streams were configured for 15 fps; the monitor sampled approximately 10 Hz and observed 30 distinct frames per camera, at **9.76 Hz base / 9.72 Hz wrist**. This does not measure the cameras' full delivered frame rate.

Camera receipt skew stayed below the 100 ms limit, with a measured maximum of **46.44 ms**. Pairing advances the older frame within a bounded timeout rather than inserting a dummy or relaxing the limit. These timestamps are host receipts after driver reads, not sensor exposure times or hardware synchronization.

**SpaceMouse:** the SDK opened `/dev/hidraw4` explicitly. Six finite axes and two button fields were read while the cameras and arm subscriptions were active. The physical run was idle: axes were zero and buttons false. All-axis/button physical exercise remains **PENDING**; a CPU SDK-fake test verified the mapping `(-y,x,z,-roll,-pitch,-yaw)` and button preservation. No operator-orientation calibration or metric scaling was introduced.

## 7. Observation Data Contract

`FrankaObservation.get_observation()` returns RLinf's existing dictionary-style `Observation`. It preserves raw measurements and separates operator intent; it does not generate an OpenPI policy vector.

| Field | Contract |
| --- | --- |
| `robot.states.joint_state` | Availability, receipt age, topic, message type, units and full original ROS message. FR3 revolute joint units: rad, rad/s, N·m. |
| `robot.states.tcp_pose` | Original header and pose; metres and ROS quaternion x/y/z/w. Tool calibration is not inferred. |
| `robot.states.external_wrench` | Original stamped force/torque in N and N·m, retaining its source frame. |
| `robot.states.gripper_state` | Explicitly unavailable with `source=None` unless a qualified JointState source is configured. No invented width or unit conversion. |
| `robot.transport` | ROS distro, domain, RMW, helper status and errors. A running helper alone is not proof of available robot state. |
| `cameras.base`, `cameras.wrist` | Real BGR arrays, SDK identity, encoding and host monotonic receipt nanoseconds. |
| `operator_intent` | Six normalized device axes, button states, axis mapping and local read-completion time. HID event source timestamp is explicitly unknown. |
| `timing` | Assembly wall/monotonic timestamps, camera receipt skew and the configured age/skew bounds. |

State cache age is evaluated only in the Workstation monotonic clock domain. ROS header stamps remain in their original source clock. Missing, expired, future-receipt, or stopped-subscriber state becomes unavailable. A missing, malformed, stale or misaligned camera pair fails the observation; it is never replaced with a blank image.

To monitor an already running, qualified publisher from the repository root:

```bash
source /opt/ros/jazzy/setup.bash
ROS_DOMAIN_ID=70 .venv/bin/python -m toolkits.realworld_check.franka_observation \
  --base-serial 337122075473 \
  --wrist-serial 218622277827 \
  --spacemouse-path /dev/hidraw4 \
  --samples 30 \
  --summary-json /tmp/franka_observation_summary.json
```

This opens only the named local sensors, starts the system-Python ROS subscription helper, reads 30 observations, logs a metadata summary, and closes owned resources. It saves neither images nor demonstrations. Confirm physical camera roles before using these arguments for a collection workflow. The optional `--gripper-topic` accepts a qualified `sensor_msgs/msg/JointState` publisher; no such Robotiq topic is currently approved.

For a caller, use the context manager and read the returned arrays directly:

```python
from toolkits.realworld_check.franka_observation import FrankaObservation

with FrankaObservation(
    base_serial="337122075473",
    wrist_serial="218622277827",
    spacemouse_path="/dev/hidraw4",
) as monitor:
    observation = monitor.get_observation()
    base_frame = observation["cameras"]["base"]["frame"]
    arm_state = observation["robot"]["states"]["joint_state"]
```

Entry opens the sensors and subscriber; `get_observation()` returns arrays and source-state availability; exit closes the owned resources even on error. Check `arm_state["status"]` before consuming `arm_state["source"]`. The caller remains responsible for a separately qualified recording or control workflow.

## 8. Verification Results

Verification covered the new logic and a short physical read-only path. No training, robot command, gripper command, controller change, demonstration collection, or model inference was performed.

| Check | Result |
| --- | --- |
| Focused CPU tests in the repository venv | **14 passed, 293 deselected**, 3.91 s. Covers new observation logic, camera timing, freshness/recovery, legacy camera observations and SpaceMouse lifecycle/path/mapping. |
| Real sensor + ROS monitor | **30 observations**, both real images, explicit HID path, three physical-arm state sources; helper shutdown and sensor cleanup succeeded. |
| RealSense SDK inventory | SDK/model/firmware/USB port identities verified; corrected use of USB descriptor serials as SDK identifiers. |
| NUC read-only controller and endpoint queries | Physical plugin, one measured-joint publisher, active controller ownership and interface claims verified. |
| Ruff `0.14.3 check --preview` on changed Python files | Passed. The repository-pinned formatter was used for final formatting. |
| `git diff --check` | Passed before commit. |
| Documentation checks | Focused `refine-docs`/`docs-check` contract and command review. Published RST pages were unchanged; Sphinx/EN-ZH builds were not run under the requested minimal verification and English-only scope. |
| Upstream CI filter update | Not applicable in this branch: no workflow tree exists. Upstream workflow coverage must be restored before upstream submission. |

The focused pytest command was:

```bash
PYTHONPATH=.:tests .venv/bin/python -m pytest -q \
  tests/unit_tests/test_franka_observation.py \
  tests/unit_tests/test_robotics.py \
  -k 'raw_state or invalid_state or camera_roles or frame_with_timestamp or camera_can_reuse or camera_rejects_expired or a_camera_without_depth or a_camera_reports_depth or stalled_camera or disconnecting_a_spacemouse or spacemouse_path_selection'
```

It uses real RLinf components with SDK fakes at hardware boundaries for the selected CPU tests. No Ray cluster was started or stopped.

Development failures were resolved before the final successful run: an unavailable SDK version pin was replaced with an available 2.58 binding; the monitor lock moved out of an unwritable shared claim directory; USB descriptor IDs were replaced with SDK IDs; ROS helper SIGTERM cleanup was made orderly; and initial 160 ms camera skew triggered bounded re-pairing. A previous Ruff version reformatted unrelated test lines; those formatting-only changes were removed and the pinned formatter applied.

## 9. Known Limitations

The read-only path works, with the following qualifications:

- **BLOCKED:** Robotiq feedback, until an activation-free, qualified source is available. The private driver status routine is not a public read-only connection lifecycle; no alternate Modbus driver was introduced.
- **PENDING:** physical camera-role confirmation, deliberate six-axis/button exercise, operator orientation, Robotiq model/calibration and actual TCP geometry.
- **PENDING:** exposure timestamps, camera clock alignment, measured network latency and cross-machine source-clock offset. Host receipt skew is only an observation pairing bound.
- **PENDING:** measured Cartesian velocity. Desired twist is not substituted.
- **PARTIALLY VERIFIED:** cross-distro ROS compatibility for the three tested standard-message streams. Milestone 1A's broader discovery warnings were not comprehensively repaired.
- **PENDING:** extended-duration performance and device-unplug recovery qualification. SpaceMouse returns the SDK's latest reading; the monitor's read timestamp is not proof of a new physical HID event.
- **BLOCKED for control/learning:** command semantics, NUC-local expiry/deadman/ownership safeguards, 19D state selection, 7D actions, SFT/RECAP camera-transform parity and the first qualified real dataset.

The existing active impedance controller was neither activated nor validated for teleoperation safety by this work. Receiving state from it is not approval to forward SpaceMouse or policy commands.

## 10. Next Milestone Recommendation

The next milestone should qualify **Robotiq read-only feedback and observation timing**: identify the exact gripper/calibration, establish an explicit activation-free feedback lifecycle or reuse an approved publisher, confirm physical camera roles, and exercise the input axes/buttons without forwarding commands.

Treat active teleoperation as a separate later milestone. It requires an agreed action contract, verified Robotiq TCP/load, operator enable, command ownership, NUC-local limits and expiry, and loss-of-link behavior. OpenPI and RECAP training remain after real-data and transform qualification.

**Milestone 1B stops here.** This report and implementation were prepared with an AI assistant. No autonomous continuation to teleoperation, data collection, SFT, or RECAP is authorized by this result.
