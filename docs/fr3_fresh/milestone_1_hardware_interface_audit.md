# Milestone 1 — FR3 Hardware and RLinf Interface Audit

This audit identifies the interfaces available for a fresh FR3 integration and
the work needed before native RLinf data collection can use them. It covers the
fresh branch, passive workstation and NUC inspection, communication evidence,
and a bounded scope for Milestone 2. It does not authorize hardware operation.

Audit date: **2026-11-19** (UTC and JST). An AI assistant performed this audit
for the human submitter. Findings describe these snapshots, not continuous
monitoring.

## 1. Scope, Evidence, and Repository State

The fresh branch is the sole RLinf implementation baseline. Current device
metadata, live ROS graphs, installed vendor interfaces, and official references
provide the hardware evidence.

| Label | Meaning |
|---|---|
| **Verified** | Observed in this checkout or independently measured in the current environment. A live parameter confirms its value, not its implementation's behavior. |
| **Source-supported** | Supported by inspected source, installed message definitions, or an official reference; not demonstrated by a hardware command. |
| **Assumed** | A proposed interpretation or design choice that needs confirmation. |
| **Not verified** | Evidence was unavailable, incompatible, or deliberately excluded by the audit constraints. |

**Verified repository state:**

| Item | Result |
|---|---|
| Repository | `https://github.com/muhamuttaqien/rlinf-project` |
| Branch | `branch-openpi-recap-franka-fresh` |
| Starting HEAD | `6be5280dc407b5923a17c3851fc3075e63eeb148` |
| Initial working tree | No tracked changes; an untracked `checkpoints` path existed and was left untouched. |
| Change scope | This standalone audit document only; no RLinf runtime edits. |

Previous RLinf experimental branches were not used as implementation references.
An external process switched the shared checkout after the initial inspection;
a late search exposed unrelated file names. That search was discarded. Final
source verification and documentation checks use the clean, dedicated
`rlinf-project-fresh` worktree, on the required branch and starting HEAD above.
No unrelated implementation was opened or reused. Process metadata revealed
that the running NUC arm controller uses an experimental installation. Its files, scripts,
configuration files, and implementation were excluded. Its live graph,
read-only status services, and live parameters were independently observable.
The existing cross-domain bridge's source was also excluded. Consequently,
command scaling and watchdog behavior remain unresolved even where live
parameter values are visible.

The required upstream overlap search used `gh pr list` and `gh issue list` on
`RLinf/RLinf`, with the query `Franka ROS2` and open state. It returned
[PR #1791, UR5 backend](https://github.com/RLinf/RLinf/pull/1791), and
[issue #759, real-world deployment](https://github.com/RLinf/RLinf/issues/759).
Neither search result establishes an existing solution for this FR3 interface;
their implementations were not used.

## 2. Native RLinf Capability Inventory

RLinf already separates robot parts, task behavior, teleoperation, and recording.
Reuse those contracts while adapting the actual hardware transport.

All repository paths below were inspected in the starting fresh checkout.

| Capability | Existing Implementation | Finding |
|---|---|---|
| Robot composition and placement | [`robot.py`](../../rlinf/robotics/robot.py), [`parts/base.py`](../../rlinf/robotics/parts/base.py), [`robots/franka.py`](../../rlinf/robotics/robots/franka.py) | **Verified:** `Robot` composes named parts through `PartGroup`; arm, end effector, and cameras have separate connection lifecycles. `FrankaConfig` carries backend and node placement settings. |
| Direct Franka control | [`arms/franky.py`](../../rlinf/robotics/parts/arms/franky.py) | **Source-supported:** registered `franky` backend uses libfranka through Franky, accepts `tcp_pose`, and obtains state/Jacobian from the robot. It owns a hardware-control connection and is unsuitable for attaching alongside the existing ROS controller. |
| Franka ROS control | [`arms/franka_ros.py`](../../rlinf/robotics/parts/arms/franka_ros.py), [`ROS transport`](../../rlinf/robotics/parts/transports/ros/ros_controller.py) | **Source-supported:** `franka_ros` uses ROS **1** (`rospy`, `roslaunch`, `dynamic_reconfigure`, SERL messages). Opening it launches impedance control; reset can stop/start controllers and recover errors. It is not a ROS 2 attach backend. |
| Canonical arm state | [`arms/base.py`](../../rlinf/robotics/parts/arms/base.py), [`arms/franka.py`](../../rlinf/robotics/parts/arms/franka.py) | **Verified:** pose, velocity, joint position/velocity, wrench, and Jacobian fields. Native pose order is position followed by quaternion `x,y,z,w`. |
| Robotiq | [`grippers/robotiq.py`](../../rlinf/robotics/parts/end_effectors/grippers/robotiq.py), [`gripper contract`](../../rlinf/robotics/parts/end_effectors/grippers/base.py) | **Source-supported:** Modbus RTU through `pymodbus`, default 115200 baud, slave `0x09`, nominal width 0.085 m. `_open()` activates the gripper. `position` and `move(width)` use opening width in metres. Reuse the contract, not a second serial connection. |
| SpaceMouse | [`teleop/spacemouse.py`](../../rlinf/robotics/parts/teleop/spacemouse.py) | **Source-supported:** direct `pyspacemouse` HID input, six axes and two buttons; produces Cartesian deltas and a gripper action. No ROS subscriber is required for this path. |
| Cameras | [`camera contract`](../../rlinf/robotics/parts/cameras/base.py), [`RealSense`](../../rlinf/robotics/parts/cameras/realsense.py), [`ZED`](../../rlinf/robotics/parts/cameras/zed.py), [`Lumos`](../../rlinf/robotics/parts/cameras/lumos.py) | **Source-supported:** SDK backends. RealSense selects a serial, starts BGR8 color, optionally Z16 depth, and aligns depth to color. Defaults are 640×480 at 15 fps. No ROS 2 camera backend was found in these robotics paths. |
| Real-world environments | [`real/env.py`](../../rlinf/envs/real/env.py), [`real/venv.py`](../../rlinf/envs/real/venv.py), [`FrankaEnv`](../../rlinf/envs/real/franka/base.py) | **Verified:** task registration, Gymnasium environment/wrapper composition, action scaling and safety-box clipping, observation conversion. Construction and reset include motion; they were not run. |
| Collection entry point | [`collect_data.sh`](../../examples/embodiment/collect_data.sh), [`collect_real_data.py`](../../examples/embodiment/collect_real_data.py) | **Verified:** shell entry selects a Hydra config, launches a Ray-backed `DataCollector`, creates `RealWorldEnv`, steps it, and saves accepted trajectories. |
| Trajectory recording | [`embodied_trajectory.py`](../../rlinf/data/schema/embodied_trajectory.py), [`replay buffer`](../../rlinf/data/storage/replay/buffer.py) | **Source-supported:** `TrajectoryAccumulator`, `TrajectoryStep`, and `TrajectoryReplayBuffer`; collector stores `.pt` trajectories and records teleoperation overrides. |
| LeRobot export | [`CollectEpisode`](../../rlinf/envs/wrappers/collect_episode.py), [`LeRobot writer`](../../rlinf/data/storage/lerobot/writer.py), [`frame schema`](../../rlinf/data/schema/embodied_types.py) | **Source-supported:** optional `data_collection.export_format: lerobot`; frame state, executed/intervened actions, images, task, completion/success and intervention metadata. Supports additional camera views. |

**Verified search result:** no `rclpy`/`rclcpp` integration was found in
`rlinf/robotics/` or `rlinf/envs/real/`. The relevant native ROS components require
ROS 1; SDK camera acquisition, HID teleoperation, Modbus control, task wrappers,
and recording are independent of ROS 2. This finding is scoped to the audited
robotics and real-environment code, not every external dependency.

Two existing configs illustrate reusable collection wiring:
[`realworld_collect_data.yaml`](../../examples/embodiment/config/realworld_collect_data.yaml)
and
[`realworld_collect_data_zed_robotiq.yaml`](../../examples/embodiment/config/realworld_collect_data_zed_robotiq.yaml).
Their placeholders, task/reset behavior, backend selection, and camera settings
must be replaced or explicitly qualified before hardware use. They were not run.

## 3. Workstation Inventory

The workstation has suitable native camera and input drivers, but those devices
were not streaming through ROS during inspection.

### Operating System and Runtime

**Verified:** Ubuntu 24.04.4 LTS; `/opt/ros/jazzy` is installed. The existing web
monitor process runs with `ROS_DISTRO=jazzy`, `ROS_DOMAIN_ID=70`,
`RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, and
`ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`. No custom DDS XML profile variable was
seen in the inspected device-related process environments. This is a
per-process observation, not a universal shell default.

**Verified installed packages:** `ros-jazzy-realsense2-camera` 4.58.1,
`ros-jazzy-joy` 3.3.0, and `librealsense2-utils` 2.58.3. The package query found
neither `ros-jazzy-spacenav` nor `spacenavd`; no running SpaceMouse acquisition
process was identified.

**Verified RLinf virtual environment:** Python 3.11, `pyrealsense2`
2.58.4.10922, `pyspacemouse` 2.0.0, and `lerobot` 0.3.3; `rclpy` is not installed
there. Jazzy's installed `rclpy` targets system Python 3.12. Attempting to import
it from Python 3.11 failed before graph inspection; the successful workstation
probes used `/usr/bin/python3`. A ROS-compatible process boundary or a
deliberately compatible runtime is required for a future adapter.

### Camera Identity and Advertised Modes

**Verified through USB/sysfs metadata:**

| Device | Serial | USB ID / Link | Device Interfaces |
|---|---|---|---|
| RealSense D405 | `323743071035` | `8086:0b5b`, 5000 Mb/s negotiated USB speed | Stable `/dev/v4l/by-id/` entries resolve to `/dev/video0`–`/dev/video5`; multiple stream/metadata interfaces. |
| RealSense D435i | `327743060094` | `8086:0b3a`, 5000 Mb/s negotiated USB speed | Stable `/dev/v4l/by-id/` entries resolve to `/dev/video8`–`/dev/video11`; additional USB HID interface. |

The table below records a useful subset of **advertised**, independently decoded
UVC frame modes from cached `/sys/bus/usb/devices/*/descriptors`. No camera handle
was opened, no V4L2 format changed, and no RealSense pipeline started. Advertised
formats and rates do not establish SDK stream selection, simultaneous-stream
support, image quality, or sustained capture rate.

| Camera / Descriptor Stream | Resolutions | Advertised Frame Rates (fps) |
|---|---|---|
| D405 Z16 depth and YUY2 color descriptors | 424×240, 480×270, 640×360, 640×480, 848×480 | 5, 15, 30, 60, 90 |
| D405 Z16 depth and YUY2 color descriptors | 1280×720 | 5, 15, 30 |
| D405 Z16 depth descriptor | 256×144 | 90 |
| D435i Z16 depth descriptors | 424×240, 480×270, 640×360, 640×480, 848×480 | 6, 15, 30, 60, 90 |
| D435i Z16 depth descriptor | 1280×720 | 6, 15, 30 |
| D435i YUY2 color descriptors | 320×180, 320×240 | 6, 30, 60 |
| D435i YUY2 color descriptors | 424×240, 640×360, 640×480, 848×480, 960×540 | 6, 15, 30, 60 |
| D435i YUY2 color descriptors | 1280×720, 1920×1080 | 6, 15, 30 |

**Not verified:** which serial is physically the base camera or wrist camera;
mounting, extrinsics, intrinsics, depth scale, exposure settings, SDK timestamp
domain, and a working simultaneous color/depth profile. D435i-as-base and
D405-as-wrist are **Assumed candidate assignments**, not audited physical facts.

**Verified graph distinction:** domain 70 contains subscriptions to
`/camera/d435i/color/image_raw/compressed` and
`/camera/d435i_first/color/image_raw/compressed`, both
`sensor_msgs/msg/CompressedImage`, owned locally by
`/franka_dual_camera_web_monitor` with best-effort/volatile QoS. Each had **zero
publishers**. These names do not identify the attached cameras or prove streams
exist. No camera publication rate, resolution, encoding, or camera timestamp
could be measured from those topics.

### SpaceMouse and Device Ownership

**Verified:** USB `256f:c63a`, product `3Dconnexion SpaceMouse Wireless BT`, with
`/dev/hidraw0`, `/dev/input/js0` and `/dev/input/event2` interfaces and stable
joystick links. Both cameras bind to the `uvcvideo` kernel driver. A SpaceMouse
USB serial number was not exposed in the inspected descriptor metadata. The installed
`pyspacemouse/devices.toml` contains `SpaceMouseWirelessNew` for this exact USB
ID and two buttons; its `open()` accepts `device_index`, as RLinf expects.

**Source-supported native semantics:** RLinf maps input to
`[-y, x, z, -roll, -pitch, -yaw]` before task action scaling. These are input
axes/deltas, not measured SI robot velocities. Left requests close, right
requests open; the native device does not supply a ROS timestamp or deadman
contract. Physical direction, button behavior, permissions, and disconnect
behavior still need an isolated input-only qualification.

**Verified ownership observation:** readable `/proc/*/fd` entries showed no
camera video/USB or SpaceMouse input/HID owner at the snapshot. The web monitor
had no such device handle. A process scan cannot prove permanent availability or
see every inaccessible descriptor; do not use this observation to justify a
competing connection later. No HID or camera reader was opened in this audit.

**Not verified:** a live SpaceMouse ROS input publisher or publication rate. The
NUC node named `/spacemouse_input_domain70` subscribes to a gripper request
topic; its name alone does not prove that it acquires a SpaceMouse.

## 4. NUC Hardware, Controllers, and Ownership

The authorized `nuc-franka` SSH alias provided read-only access. The NUC is the
current control owner; RLinf must attach to approved interfaces without taking
over its hardware connections.

**Verified:** Ubuntu 22.04.5 LTS, ROS 2 Humble, separate controller-manager
processes on domains 70 and 71, and `ROS_LOCALHOST_ONLY=0` on those processes.
Domain 71 explicitly selects `rmw_fastrtps_cpp`; the arm process has that library
loaded even though its environment leaves `RMW_IMPLEMENTATION` unset. The
gripper process advertises subnet discovery. Domain 42 appears in unrelated
older process environments; short graph probes on both machines found only
the audit node there. Domain 0 was also empty apart from the workstation probe.

**Verified installed vendor package metadata:** Franka `franka_hardware`,
`franka_msgs`, `franka_robot_state_broadcaster`, `franka_bringup`,
`franka_gripper`, and `franka_semantic_components` are version 2.1.0;
`franka_description` is 1.3.0. Robotiq driver, controller, description, and message
packages report 1.3.0. Package version strings do not prove unmodified upstream
sources or binary/source identity.

### FR3 Arm

**Verified software identity:** live `arm_id=fr3`, seven `fr3_joint*` joints, and
active `franka_hardware/FrankaHardwareInterface`. This confirms the deployed FR3
software configuration; the physical nameplate, robot firmware, and robot serial
were not independently inspected.

| Domain 70 Component | Live State / Ownership |
|---|---|
| `joint_impedance_ik_controller` | Active, type `franka_arm_controllers/JointImpedanceIKController`; exclusively claims `fr3_joint1/effort` through `fr3_joint7/effort`. |
| `joint_state_broadcaster` | Active; publishes `/franka/joint_states` and `/dynamic_joint_states`. |
| `franka_robot_state_broadcaster` | Active; publishes detailed Franka state and derived state topics. |
| `FrankaHardwareInterface` | Active hardware component. Joint position/velocity and Cartesian pose/velocity command interfaces are available but unclaimed. Availability is not a ROS topic or permission to claim them. |
| `joint_state_publisher` | Two discovered instances publish aggregate `/joint_states`; use the named arm broadcaster instead. |

**Verified live parameters:** controller-manager update rate 1000 Hz;
`ik_update_rate=50.0`, `command_timeout=0.25`, `load_gripper=false`,
`arm_mounting_orientation=[0,0,0]`, and `use_sim_time=false`.
The controller's joint stiffness/damping arrays were readable, but they were
neither changed nor qualified as safe gains. Controller-local `update_rate=0`
does not establish a stopped controller; fresh state arrived at about 1 kHz.

**Verified command endpoint:**
`/franka_controller/target_cartesian_velocity_percent`,
`geometry_msgs/msg/Twist`, best-effort/volatile subscription owned by
`/joint_impedance_ik_controller`, with **zero discovered publishers** during
the snapshots. No Cartesian-pose or joint-trajectory ROS command topic/action
was discovered. Unclaimed ros2_control interfaces must not be confused with
an active command server.

**Not verified:** this custom Twist interface's numeric bounds, percent scale,
linear/angular velocity gains, coordinate frame, saturation, stale-command
handling, and exact stop behavior. `Twist` has no header; neither a frame nor a
sender timestamp can be inferred from the message. The timeout parameter is
visible, but its enforcement was not tested or established from the excluded
active controller source. No motion, zero-command test, timeout experiment, or
recovery request was sent.

**Source-supported state semantics:** installed `FrankaRobotState.msg` labels
`o_t_ee` as measured end-effector pose in the base frame, separates desired and
measured joint state, and distinguishes base-frame and stiffness-frame wrench.
Joint position/velocity use rad/rad·s⁻¹; joint effort is torque in N·m; pose
translation is in metres and wrench uses N/N·m. These are standard Franka
semantics, supported by the
[official RobotState reference](https://frankarobotics.github.io/libfranka/latest/structfranka_1_1RobotState.html).
They do not establish the custom command topic's units.

### Robotiq Gripper

**Verified configuration and ownership:** launch arguments and live parameters
select `2f_85`, hardware mode, and 115200 baud. The NUC USB-to-RS485 by-id link
resolves to `/dev/ttyUSB1`, owned by the domain-71 `ros2_control_node` process.
No other serial connection was opened. The workstation's separate FTDI device
is not evidence of a second gripper.

| Domain 71 Component | Live State / Ownership |
|---|---|
| `robotiq_gripper_controller` | Active; claims `robotiq_85_left_knuckle_joint/position`. |
| `robotiq_activation_controller` | Active; claims the two reactivation interfaces. It was not invoked. |
| `robotiq_gripper_status_broadcaster` | Active; publishes typed fault/object/current state. |
| `robotiq_normalized_position_broadcaster` | Active; publishes `robotiq_normalized_joint` on `/joint_states`. |
| `joint_state_broadcaster` | Active; also publishes raw knuckle state on `/joint_states`. |
| `RobotiqGripperHardwareInterface` | Active, class `robotiq_driver/RobotiqGripperHardwareInterface`; speed/effort-setting interfaces are exposed but unclaimed. |

**Verified action endpoints:**
`/robotiq_gripper_controller/gripper_cmd` and
`/robotiq_gripper_controller/normalized_gripper_cmd`, both
`control_msgs/action/GripperCommand`. The existing cross-domain bridge process
hosts `/spacemouse_input_domain70` and `/robotiq_output_domain71`; graph
inspection shows the former consuming a domain-70 Float32 gripper request and
the latter subscribing to standard-action feedback/status on domain 71.
Its forwarding implementation and scaling were not read or tested. Do not add
a second command client without resolving ownership.

**Verified parameters:** closed joint position 0.7929,
`gripper_profile=2f_85`, `max_effort=50.0`, `allow_stalling=true`,
`goal_tolerance=0.02`, `stall_timeout=0.05`, `object_status_timeout=10.0`,
`action_monitor_rate=100.0`, and manager update rate 500 Hz.
These do not prove the serial device refreshes at 500 Hz or that requested
effort is calibrated grip force.

**Source-supported conversions:** the installed canonical workspace's
`robotiq_driver/include/robotiq_driver/gripper_scaling.hpp` maps register counts
through the selected device profile to joint units. The shipped 2F knuckle is
revolute, so the standard action's position is a **joint angle in radians**,
not jaw opening in metres. The normalized action uses **0=open, 1=closed**;
`robotiq_normalized_joint` follows the normalized register position. Do not
assume raw angle divided by 0.7929 exactly equals the normalized state: the
profile includes register endpoints and quantization. A physical opening-width
conversion still needs qualification against RLinf's metre-based contract.

**Source-supported status:** installed `GripperStatus.msg` carries
`object_detection` (0 moving, 1 detected opening, 2 detected closing, 3 at target),
`motor_current` in amperes, and fault/severity codes. Unknown fault/severity
values are 255, distinct from zero/no fault. Motor current is not grip force.
The installed driver source sets reported velocity to zero because its status
block has no velocity measurement; do not interpret that field as a measured
stationary condition.

## 5. Live ROS Graph, QoS, and Frequencies

The following tables separate advertised endpoints from streams that actually
delivered messages. Endpoint ownership was clearest when queried on the NUC.

QoS abbreviations: **R** reliable, **BE** best effort, **TL** transient local,
**V** volatile. Discovery returned history `UNKNOWN` and depth 0 for remote
endpoints, so effective queue depth is **Not verified**. Relevant queried
endpoints reported automatic liveliness and effectively infinite deadline and
lease (`9223372036854775807` ns). These settings do not supply a finite safety
watchdog.

| Domain | Topic | Message Type | Owner / QoS | Observation |
|---|---|---|---|---|
| 70 | `/franka/joint_states` | `sensor_msgs/msg/JointState` | `joint_state_broadcaster`, R/TL | Fresh measured arm state; reorder by joint name. |
| 70 | `/dynamic_joint_states` | `control_msgs/msg/DynamicJointState` | `joint_state_broadcaster`, R/TL | Advertised; not sampled. |
| 70 | `/franka_robot_state_broadcaster/robot_state` | `franka_msgs/msg/FrankaRobotState` | `franka_robot_state_broadcaster`, R/TL | Sampled on NUC; custom type unavailable to workstation probe. |
| 70 | `/franka_robot_state_broadcaster/current_pose` | `geometry_msgs/msg/PoseStamped` | Same broadcaster, R/TL | Sampled on both machines; frame `base`. |
| 70 | `/franka_robot_state_broadcaster/measured_joint_states`, `/franka_robot_state_broadcaster/desired_joint_states`, `/franka_robot_state_broadcaster/external_joint_torques` | `sensor_msgs/msg/JointState` | Same broadcaster, R/TL | Advertised, not sampled. |
| 70 | `/franka_robot_state_broadcaster/last_desired_pose` | `geometry_msgs/msg/PoseStamped` | Same broadcaster, R/TL | Desired pose is not a command input. |
| 70 | `/franka_robot_state_broadcaster/desired_end_effector_twist` | `geometry_msgs/msg/TwistStamped` | Same broadcaster, R/TL | Desired twist is not measured TCP velocity. |
| 70 | `/franka_robot_state_broadcaster/external_wrench_in_base_frame`, `/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame` | `geometry_msgs/msg/WrenchStamped` | Same broadcaster, R/TL | Advertised, not sampled. |
| 70 | `/franka_controller/target_cartesian_velocity_percent` | `geometry_msgs/msg/Twist` | `joint_impedance_ik_controller` subscriber, BE/V | Zero publishers; never published by audit. |
| 70 | `/gripper_client/target_gripper_width_percent` | `std_msgs/msg/Float32` | `spacemouse_input_domain70` subscriber, R/V | Zero publishers; units/range unverified. |
| 70 | `/franka_gripper/joint_states` | `sensor_msgs/msg/JointState` | Two aggregate-state subscribers, R/V | Zero publishers; not the Robotiq state interface. |
| 70 | `/joint_states` | `sensor_msgs/msg/JointState` | Two `joint_state_publisher` instances, R/V | Aggregate stream, approximately 60 Hz combined. |
| 70 | Two camera compressed topics listed above | `sensor_msgs/msg/CompressedImage` | Web-monitor subscriber, BE/V | Zero publishers; publication rate unavailable. |
| 71 | `/joint_states` | `sensor_msgs/msg/JointState` | Raw and normalized broadcasters, both R/TL | Approximately 1000 arrivals/s combined; two different representations. |
| 71 | `/robotiq_gripper_status_broadcaster/status` | `robotiq_msgs/msg/GripperStatus` | Status broadcaster, R/TL | Approximately 500 arrivals/s on NUC. |
| 70, 71 | `/robot_description`, `/tf`, `/tf_static` | `std_msgs/msg/String`, `tf2_msgs/msg/TFMessage` | Per-domain `robot_state_publisher`; R/TL, R/V, R/TL respectively | Advertised; no camera-to-robot calibration verified. |
| 70, 71 | `/parameter_events`, `/rosout`, controller transition events | Standard ROS infrastructure types | Multiple nodes | Infrastructure presence does not indicate device acquisition. |

**Verified services and actions:** both domains advertise controller-manager
list, load, configure, switch, unload, and hardware-state services, plus standard
parameter services. Only list/status and get/list-parameter requests were used.
Domain 70 advertises `/action_server/error_recovery`
(`franka_msgs/action/ErrorRecovery`) and `/service_server/` services for
Cartesian/joint stiffness, force/torque/full collision behavior, load, stiffness
frame, and TCP frame. Domain 71 advertises the two gripper actions, including
goal, result, cancellation, feedback, and status endpoints. None of these
state-changing services or action goals/cancellations was called.

Short state-only subscriptions used BE/V, depth 5, approximately three seconds
for discovery and five seconds for sampling per domain. Rates below are arrival
rates `(count-1)/(last-first)`, not a hardware timing guarantee. NUC and
workstation windows were separate, so their counts are not loss comparisons.

| Stream | NUC Arrival Rate | Workstation Arrival Rate | Observed Header Age at Workstation |
|---|---|---|---|
| Arm `/franka/joint_states` | 1000.9 Hz | 1000.5 Hz | 0.128–6.926 ms |
| Arm current pose | 1000.3 Hz | 999.5 Hz | 0.152–4.730 ms |
| Full Franka state | 1000.0 Hz | Not decoded: missing custom type | Not measured |
| Domain-70 aggregate joints | 60.1 Hz | 60.1 Hz | 0.186–2.530 ms |
| Domain-71 combined joints | 1000.6 Hz | 1000.1 Hz | 0.111–1.339 ms |
| Robotiq typed status | 500.0 Hz | Not decoded: missing custom type | Not measured |

**Verified sample semantics:** `/franka/joint_states` arrived in the order
`joint1, joint3, joint6, joint7, joint2, joint4, joint5` (all prefixed `fr3_`).
Its frame was `base_link`; current pose used `base`; full-state top-level frame
was empty while its pose submessage used `base`. Equivalence of these frame names
was not established. The full-state sample reported robot mode 2 (MOVE), command
success rate 1.0, and no true current-error fields. MOVE is a control mode, not
evidence that the audit requested motion or a guarantee of physical stillness.

The gripper status sample reported object state 3, current 0 A, fault 0, and
severity 0. A raw knuckle sample was 0 rad while the normalized stream's sample
was about 0.0117647. Domain-71 combined joints had roughly half as many distinct
timestamps as messages, consistent with two broadcasters. Repeated fresh ROS
headers do not prove new serial measurements on every publication.

## 6. Network and Synchronization

Cross-machine reception is established for standard state messages, while
interoperability and timing remain narrower than a complete control guarantee.

**Verified:** authorized SSH communication and ROS 2 discovery/data delivery
from NUC to workstation on domains 70 and 71. Both used Fast DDS for the probes.
The workstation saw standard state topics and received changing header stamps;
the NUC also discovered workstation camera-monitor subscriptions. No raw network
addresses, DDS GUIDs, credentials, or machine logs are included here.

**Verified limitation:** mixed Humble/Jazzy discovery emitted
`sequence size exceeds remaining buffer` and exposed many remote node names as
unknown, despite standard message delivery. This is evidence of incomplete graph
interoperability, not proof of a specific root cause. Custom type compatibility,
action execution, and workstation-to-NUC command delivery were not tested.
ROS explicitly does not guarantee cross-distribution communication; see the
[official ROS 2 distribution documentation](https://raw.githubusercontent.com/ros2/ros2_documentation/jazzy/source/Releases.rst).
Prefer a distribution-compatible ROS endpoint for the adapter.

**Source-supported QoS choice:** BE/V state subscribers received the reliable
transient-local state publishers in the measured configuration. A future adapter
must explicitly select compatible QoS per endpoint; reliable delivery alone
does not establish freshness. The
[official ROS QoS reference](https://raw.githubusercontent.com/ros2/ros2_documentation/jazzy/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst)
describes reliability/durability matching. Domain separation also prevents
accidental conflation of the two controller managers and `/joint_states` topics;
each ROS context must retain its domain identity.

**Verified time synchronization:** both hosts run chrony and report
`NTP=yes`, `NTPSynchronized=yes`, normal leap state, and stratum 3. Representative
`chronyc tracking` snapshots reported:

| Metric | Workstation | NUC |
|---|---|---|
| System correction relative to selected NTP estimate | 55.733 µs slow | 20.418 µs slow |
| RMS offset | 0.312451 ms | 0.236710 ms |
| Root delay | 12.203299 ms | 14.384573 ms |
| Root dispersion | 1.082733 ms | 1.098677 ms |

The hosts selected different upstream references. These values are local
estimates, not a measured workstation-minus-NUC offset or an accuracy bound.
No PTP or shared camera hardware synchronization was established. No clock
adjustment, network reconfiguration, ping load test, or timing daemon restart
was performed.

**Source-supported timestamp origin:** the canonical Franka broadcaster assigns
the controller update's ROS time to its header and propagates it to nested
messages. `use_sim_time=false` was readable on the queried controllers. Franka's
`time` field is robot-relative time, not Unix time. Header-to-receipt age above
combines clock error, broadcaster scheduling, DDS transport, and callback delay;
it is not sensor exposure age or one-way transport latency.

**Not verified:** action latency, round-trip control latency, jitter under
training load, camera exposure timestamps, sensor-clock alignment, USB capture
latency, and camera/arm/gripper skew. Native `BaseCamera` caches local monotonic
receipt times but returns arrays without the SDK hardware timestamp. Native
SpaceMouse state is also an untimestamped latest reading. `CollectEpisode`
records step-aligned data and configured dataset fps; it does not establish
physical sensor synchronization or preserve all ROS acquisition stamps by
default. Define observation age/skew limits before collecting demonstrations.

## 7. Compatibility and Minimal Adaptation

The compatibility decision follows the observed transport and lifecycle, rather
than the presence of a matching vendor name.

| Component | Existing Interface | RLinf Support | Required Adaptation |
|---|---|---|---|
| Franka FR3 | NUC Humble/domain 70; full and standard state; custom velocity-percent Twist subscriber; existing effort controller owns FCI | Native `FrankaRobot`, arm/state contracts, `franky`, ROS 1 `franka_ros` | New ROS 2 attach backend/session. Resolve native absolute `tcp_pose` versus velocity-percent semantics, frame/scales, freshness and ownership first. Reuse composition; do not open Franky or launch ROS 1 controllers. |
| Robotiq | NUC Humble/domain 71; Modbus RTU owned by ros2_control; raw-angle and normalized actions; typed status | Native `BaseGripper` and direct Modbus `RobotiqGripper` | New ROS 2 gripper backend using the approved existing action owner, explicit profile/width conversion, state selection and faults. Do not open the serial port or reactivate on connect. |
| SpaceMouse | Workstation USB HID/input; exact USB ID supported by installed pyspacemouse; no ROS acquisition publisher | Native HID `SpaceMouse` and teleop composition | Prefer native input after input-only qualification; no ROS adapter currently justified. Verify axes, buttons, deadman/disconnect and permission behavior. |
| Base Camera | One of the two USB RealSense serials; physical role not confirmed; no live ROS camera publisher | Native serial-selected `RealSenseCamera` | Confirm role/profile; prefer native SDK acquisition once ownership is checked. ROS camera adapter only if an external publisher will own acquisition. |
| Wrist Camera | Other USB RealSense serial; physical role not confirmed; no live ROS camera publisher | Same native RealSense backend | Confirm mount and serial, compatible color/depth mode and timestamp handling; qualify color order. No new camera driver indicated yet. |
| Data Collection | Existing `collect_data.sh` → `DataCollector` → `RealWorldEnv`, trajectory replay and optional `CollectEpisode` | Native trajectory and LeRobot recording | Dedicated fresh hardware config and task/lifecycle behavior; retain recorder. Resolve executed-action definition and observation timing before acceptance. |

**Source-supported collection hazards:** constructing `FrankaEnv` connects the
robot and interpolates toward a reset pose. `reset()` can move Cartesian/joint
targets, reconfigure task compliance, and task-specific code can command the
gripper. A normal collection step converts scaled Cartesian deltas into absolute
pose targets and sends end-effector actions before the arm action. Neither a
zero policy action nor disabling automatic reset makes this path read-only.

Additional **confirmed integration gaps** are:

1. **Action contract:** native arm pose targets cannot be remapped directly to
   the custom velocity-percent topic. A pose-tracking conversion requires an
   independently verified protocol and bounded control design; alternatively,
   a task/action path must explicitly support verified velocities. Select one
   contract rather than silently changing recorded action meaning.
2. **Observation completeness:** full Franka state supplies measured pose,
   joints, and wrench. The inspected graph supplies desired TCP twist, not a
   verified measured TCP velocity or Jacobian stream. Native arm state expects
   those fields. Choose a documented computation/source or a narrower valid
   state contract; do not fill missing measurements with plausible zeros.
3. **Runtime compatibility:** RLinf Python 3.11, Jazzy Python 3.12, and Humble
   Python 3.10 are distinct runtimes. Reuse native placement where possible,
   with a ROS-compatible helper process if needed. Do not inject a foreign
   `rclpy` extension into the RLinf interpreter.
4. **Gripper state:** filter by domain, joint name, and intended broadcaster.
   Convert calibrated/profile-aware state into RLinf opening metres; preserve
   raw joint and normalized closure meanings in diagnostics.
5. **Image semantics:** RealSense requests BGR8, while `FrankaEnv._frames_from`
   reverses channels with comments describing RGB display/BGR policy input.
   That path needs an image-based color qualification through environment and
   export. Array shape alone does not confirm color semantics.
6. **Configuration and limits:** the native arm helper contains Panda-named
   joint limits. Qualify FR3 limits, tool transform, workspace, gains, reset
   poses and collision settings independently; do not inherit generic task
   defaults as hardware safety boundaries.
7. **Timing and export:** the existing frame schema records state/actions/images
   without the ROS source stamps. Decide whether side metadata or a narrowly
   scoped schema extension is needed for age/skew auditing; dataset fps alone
   is insufficient. Dataset `robot_type` metadata should describe the actual
   setup rather than retaining a Panda example label.

## 8. Unresolved Information and Safety Boundaries

The audit is complete within available access. The following items block
qualified live data collection, rather than completion of a read-only audit.

| Unresolved Item | Evidence Needed Before Hardware Use |
|---|---|
| Active custom arm command protocol | An allowed authoritative description/source for scales, limits, coordinate frame, callback/control behavior, saturation, and timeout enforcement. The experimental implementation was excluded. |
| Ownership handover | Named owner for arm commands and gripper goals; policy for the existing bridge, duplicate writers, reconnect, cancellation, and unexpected disconnect. Zero discovered publishers is not an ownership lock. |
| Hardware safety configuration | Qualified FR3 joint/workspace limits, collision thresholds, TCP/load configuration, emergency-stop access, and safe hold/stop behavior. No Desk or safety settings were read or changed. |
| Reset and lifecycle policy | Explicit attach, start, reset, fault, disconnect behavior consistent with externally managed controllers; no automatic recovery, activation, or reset motion during connection. |
| Camera roles and stream settings | Physical serial-to-role check, intrinsics/extrinsics and required depth/color profile; initial isolated capture qualification when separately authorized. |
| Gripper physical calibration | Verified model/nameplate if needed, opening-width conversion, expected contact/force behavior, and distinction between requested force and measured current. |
| State completeness and frames | Source of measured TCP velocity/Jacobian where required; mapping among `base`, `base_link`, TCP and stiffness frame, plus deterministic joint order. |
| Timing and load behavior | Clock-offset visibility, camera timestamp source, allowable age/skew, stale state handling, and bounded end-to-end latency under expected workload. |

**Verified safety procedure:** the audit used filesystem/sysfs metadata, process
metadata, discovery, read-only status/parameter queries, and short state-only
subscriptions. It issued no arm or gripper command, action goal/cancellation,
controller switch/load/configure request, recovery request, parameter write,
device-stream start, serial connection, data collection, or training. Existing
hardware workspaces and running device processes were left unchanged. Temporary
audit nodes added normal ROS discovery/infrastructure endpoints, not motion
publishers. Their executors and contexts were closed after each probe.

## 9. Recommended Milestone 2 — ROS 2 Adapter

Milestone 2 should make the existing interfaces usable through RLinf's part
contracts, starting with offline and observation-only qualification. It should
not replace the externally managed control stack.

1. **Resolve the protocol blockers first.** Independently establish the active
   arm Twist contract, watchdog/hold semantics, permitted safety limits, gripper
   units, and command ownership without consulting excluded experimental
   integration materials. Record the chosen pose-versus-velocity contract.
2. **Implement an attach-only ROS 2 session and part backends.** Use explicit
   domain-70/domain-71 contexts, appropriate message packages, compatible QoS,
   deterministic joint mapping, timestamped state caches and freshness errors.
   Run ROS bindings in a compatible interpreter; reuse RLinf `BaseArm`,
   `BaseGripper`, composition and placement. Connecting/disconnecting must not
   launch, restart, switch, recover or activate hardware controllers.
3. **Integrate task lifecycle deliberately.** Add the smallest dedicated task
   subclass or explicit lifecycle option required to avoid inherited startup
   and reset motions. Keep hardware commands disabled during adapter bring-up;
   model reset/hold/stop as explicit operations after their contracts are known.
4. **Reuse native cameras and teleoperation.** Assign confirmed serials to
   named views, qualify a common stream profile and color order, and verify
   SpaceMouse input with robot output disabled. Add ROS sensor adapters only
   if acquisition ownership changes to existing ROS publishers.
5. **Add a fresh collection config.** Wire the new registered parts and task to
   `examples/embodiment/collect_data.sh`, set actual hardware placement, camera
   names/main view, bounded rates, episode controls and LeRobot metadata.
   Preserve the native trajectory/episode writer and record executed-action
   meaning and observation timing explicitly.
6. **Validate before any live collection.** Add contract tests in the existing
   robotics component test file, with vendor/ROS process-edge fakes for joint
   order, units, domains, stale state/input, reconnect, ownership, fault handling
   and lifecycle side effects. Validate recording alignment, color and metadata
   with synthetic observations. A subsequent explicitly authorized hardware
   qualification must establish safe stopping and control behavior before
   demonstrations are collected.

Expected minimal implementation scope is new ROS 2 transport/part code, a small
task/lifecycle integration, a fresh collection config, relevant component tests,
and documentation. The recorder and native camera/SpaceMouse drivers are
reusable subject to the qualifications above. No Milestone 2 code or config was
implemented or launched during this milestone.

## 10. Evidence and Documentation Checks

The evidence below supports reproduction without publishing raw machine logs,
private addresses, or credentials. Diagnostic scripts were created afresh in
`/tmp` for this audit and were not committed or installed in hardware workspaces.

Repository and package commands actually used included:

```bash
git branch --show-current
git status --short
git rev-parse HEAD
git remote -v
gh pr list --repo RLinf/RLinf --search 'Franka ROS2' --state open
gh issue list --repo RLinf/RLinf --search 'Franka ROS2' --state open
cat /etc/os-release
ls /opt/ros
lsusb
chronyc tracking
chronyc sources -n
timedatectl show -p NTPSynchronized -p NTP
```

The Git commands established the baseline and untracked entry; USB/package and
time commands identified installed devices and synchronization state. Network
addresses returned by diagnostics were omitted from this report.

For ROS inspection, the fresh probes used the following read-only API surface
instead of starting/restarting the ROS CLI daemon:

| Evidence | Method |
|---|---|
| Workstation and NUC graphs | Short-lived `rclpy` nodes; `get_node_names_and_namespaces`, `get_topic_names_and_types`, `get_service_names_and_types`, `get_action_names_and_types`; workstation domains 0/42/70/71, NUC 42/70/71. |
| QoS and endpoint owners | `get_publishers_info_by_topic` and `get_subscriptions_info_by_topic`; local NUC graph used to resolve missing cross-distribution node metadata. |
| Controller state/claims | Only `ListControllers`, `ListHardwareInterfaces`, and `ListHardwareComponents` against each domain's `/controller_manager`. |
| Parameter values | Only `ListParameters` and `GetParameters` for the controller managers and audited controllers/broadcasters; sensitive/address fields excluded from output. |
| Rates and header ages | State-only BE/V subscribers, depth 5, monotonic arrival-time deltas, wall-time minus message header; fresh stamps counted separately. |
| Camera identity and modes | `/dev/v4l/by-id`, cached USB sysfs IDs/serial/speed/descriptors; UVC frame intervals converted from 100 ns units with `fps=10^7/interval`. No device open. |
| Process ownership and domain | Readable `/proc/*/fd`, ROS-specific `/proc/*/environ`, executable metadata and arm process loaded RMW library. No broad environment or credentials saved. |
| Vendor semantics | Canonical installed Franka package manifests/messages/broadcaster source, and Robotiq manifests/messages/scaling/controller source; experimental arm installation and bridge scripts excluded. |

The report is a standalone Markdown audit outside `docs/source-en` and
`docs/source-zh`; neither public-language tree or navigation is changed. The
documentation build harness checks both existing trees and filters missing
autodoc imports in its docs-only environment. Its result does not claim this
standalone report was included in Sphinx or that hardware behavior was tested.

| Check | Result |
|---|---|
| `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/fr3-fresh-uv-cache python3 .agents/skills/docs-check/build_docs.py --venv /tmp/fr3-fresh-docs-venv` | **Passed in the dedicated fresh worktree:** EN 0 retained warnings; ZH 0 retained warnings. Docs-only autodoc filtering applies as described above. |
| `/tmp/fr3-fresh-docs-venv/bin/python .agents/skills/docs-check/check_rst_markup.py` | **Existing findings:** 0 build-breaking errors and 61 silent-rendering warnings in unchanged Chinese RST pages. This audit adds no RST changes. Those unrelated warnings were left untouched. |
| Manual doc-to-code checks | **Passed:** all 27 local links resolve; native backend/lifecycle/color-format assertions and the scoped absence of `rclpy`/`rclcpp` were rechecked against the fresh HEAD. |
| Markdown parsing/rendering | **Passed:** CommonMark parsing with tables, one H1, and the command fence; a temporary HTML preview was generated outside the repository. |
| Sensitive-content review | **Passed:** no private network addresses, credentials, DDS GUIDs, raw machine logs, or device secrets in the report. Camera serials are retained as requested hardware identifiers. |
| `pre-commit run --files docs/fr3_fresh/milestone_1_hardware_interface_audit.md` | Ruff hooks skipped Markdown as expected. The initial signoff hook failed on the unsigned starting HEAD; the new audit commit uses the configured human identity and `git commit -s`. |
| Runtime / hardware tests | **Intentionally not run:** this change is documentation only. No controller, device acquisition, collection, or training test was launched. |

The first docs dependency download failed inside the sandbox due to DNS
restriction, and the sandboxed Sphinx build could not create its local
multiprocessing socket. Authorized reruns outside the sandbox completed the
fresh-worktree build. These were audit-environment limitations, not source fixes.

The final diff review is restricted to this new report. The signed commit uses
`docs(fr3): complete fresh hardware and RLinf interface audit` and is intended
for `origin/branch-openpi-recap-franka-fresh`. No source, configuration, generated
HTML, temporary diagnostic script, or hardware-workspace file is included.
