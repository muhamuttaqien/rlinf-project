# Milestone 1A — Workstation–NUC–Franka Hardware and ROS 2 Audit

## 1. Objective

This report records the read-only hardware and integration audit performed on **2026-10-23 (JST)** for OpenPI pi0.5 SFT and RECAP on Franka with a Robotiq gripper. The audit is complete within the permitted inspection scope; hardware deployment readiness is **BLOCKED** by the findings below.

The Workstation remained on `branch-openpi-recap-franka`, commit `07a8acdc5094203538c38af64fd0cc515e7291a8`. `git fetch origin` succeeded, and the branch matched the fetched remote branch. No fast-forward was necessary. The pre-existing untracked symlinks `assets/pi05_stackcube_panda_joint` and `checkpoints` were preserved. This report is the only repository change made for Milestone 1A.

The audit read `AGENTS.md` and all seven existing files in this experiment directory, inspected local and NUC source/configuration, queried device metadata and ROS graphs, and ran CPU-only transform checks. No robot or gripper connection method, control session, activation, homing, reset, command publisher, installation, training, normalization computation, or dataset collection was performed. NUC files and configuration were not edited. Temporary audit scripts and logs were kept under `/tmp` on the Workstation; remote scripts were supplied through SSH standard input.

Statuses mean:

| Status | Meaning |
| --- | --- |
| VERIFIED | Directly observed by a command or source check; the stated scope matters. |
| PARTIALLY VERIFIED | Supporting evidence exists, but an essential runtime or physical check remains. |
| PENDING | Not established by this audit. User reports are identified explicitly. |
| BLOCKED | An observed mismatch or missing prerequisite prevents the proposed integration from being accepted. |

## 2. Workstation Environment

The Workstation can enumerate its GPUs and attached peripherals, and the existing learning environment imports successfully. Its ROS and learning runtimes use different Python versions.

| Item | Status | Evidence |
| --- | --- | --- |
| Host and OS | VERIFIED | `pca03395`; Ubuntu 24.04.4 LTS; kernel `6.8.0-134-generic`, x86_64. |
| ROS installation | VERIFIED | `/opt/ros/jazzy`; sourcing its setup provides ROS 2 Jazzy with system Python 3.12. |
| Initial audit-shell ROS configuration | VERIFIED | `ROS_DISTRO`, `ROS_DOMAIN_ID`, and `RMW_IMPLEMENTATION` were unset. The inspected `.bashrc`/`.profile` contained no matching ROS setup/domain entries. This does not describe every open terminal. |
| GPUs | VERIFIED | Four NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition GPUs, each reporting 97,887 MiB; driver `590.48.01`. Other GPU activity was left untouched. |
| PyTorch CUDA access | VERIFIED | With `CUDA_VISIBLE_DEVICES=1`, PyTorch reported CUDA available, one visible GPU, and the expected device name. No model or tensor benchmark was run. PyTorch also emitted `Can't initialize NVML`; `nvidia-smi` independently succeeded outside the sandbox. |
| Learning environment | VERIFIED | Repository `.venv`: Python 3.11.15, Torch `2.11.0+cu130`, torchvision `0.26.0+cu130`, NumPy `1.26.4`, Ray `2.59.0`, LeRobot `0.3.3`, JAX/JAXlib `0.5.3`. |
| OpenPI installation | VERIFIED | Importable `openpi` comes from installed `rlinf-openpi==0.1.1`. Stock `transformers` metadata reports `4.57.6`; `rlinf-transformer-openpi==4.53.2` is also installed, and the imported `transformers` module reports `4.53.2`. Shared package ownership means distribution metadata alone does not establish model compatibility. |
| Import checks | VERIFIED | `torch`, `rlinf`, `openpi`, `lerobot`, `cv2`, `openpi.transforms`, and `rlinf.models.embodiment.openpi.dataconfig` imported. These checks did not load weights or establish training readiness. |
| Device SDKs in repository `.venv` | BLOCKED | `pyspacemouse`, `easyhid`, `pyrealsense2`, `pymodbus`, `rclpy`, and `franky` were absent. `pyserial==3.5` was present. |
| System peripheral libraries | PARTIALLY VERIFIED | Python 3.12 user site has `pyspacemouse==1.1.4` and `easyhid==0.0.10`. System packages include librealsense `2.58.3` and hidapi `0.14.0`; system Python did not find `pyrealsense2` before ROS setup. |

The installed SpaceMouse `open()` signature uses `DeviceNumber`, while RLinf calls `open(device_index=...)` and declares `pyspacemouse>=2.0.0` in `requirements/embodied/envs/franka.txt`. The detected Python 3.12 package cannot be treated as the required RLinf SDK merely because it imports.

Sibling Workstation directories already contain `franka_ros2_ws`, `franka_spacemouse_ws`, and `franka_hdf5_recorder`. They were inspected as existing work, not launched or rebuilt. The recorder README assumes Franka finger feedback and camera ROS topics; its presence does not establish a Robotiq-compatible recording pipeline.

## 3. NUC Environment

SSH access through `ssh nuc-franka` is **VERIFIED**. The NUC has built Franka ROS 2 workspaces and a real-time kernel, but its default interactive ROS domain differs from the experiment plan.

| Item | Status | Evidence |
| --- | --- | --- |
| Host and OS | VERIFIED | `aistfranka-NUC13ANK-B`; Ubuntu 22.04.5 LTS; kernel `6.8.0-rt8`, `PREEMPT_RT`. |
| ROS and Python | VERIFIED | `/opt/ros/humble`; ROS 2 Humble with system Python `3.10.12`. |
| Interactive shell domain | VERIFIED | `~/.bashrc` sources Humble and exports `ROS_DOMAIN_ID=42`. Noninteractive SSH initially had ROS variables unset. |
| ROS middleware packages | VERIFIED | `rmw_fastrtps_cpp==6.2.10`; controller manager `2.54.0`. Workstation Fast DDS RMW package version is `8.4.4`. |
| Franka ROS 2 workspace | VERIFIED | `~/Desktop/aist-gamma/franka_ros2_ws`; source revision `76a3418`; `franka_msgs` package version `2.1.0`. Packages become discoverable after sourcing the overlay. |
| SpaceMouse/controller workspace | VERIFIED | `~/Desktop/aist-gamma/franka_controller_ws`; source revision `1d041da`; `franka_arm_controllers` package version `0.2.0`, with local changes and custom controller files. |
| Existing RLinf environment | PARTIALLY VERIFIED | `~/RLinf/.venv`: Python `3.11.14`, Ray `2.58.0`, NumPy `2.2.0`, `pymodbus==3.15.0`, `franky-control==2.0.0`; `rclpy`, `pyspacemouse`, and `pyrealsense2` absent there. Hardware SDKs were not connected. |
| Existing NUC checkout | VERIFIED | `~/RLinf` is on `main`, commit `d34d4c32`, with 522 status entries. It is a separate, heavily modified checkout and was left unchanged. |
| Existing distributed processes | VERIFIED | Ray processes reference the Workstation at `10.42.0.1:6379`. They were not stopped or reconfigured. The observed Ray versions differ across machines. |

## 4. Franka FCI and Controller Findings

The NUC configuration targets an FR3, and the configured robot IP responds to ICMP. A live FCI state/control session was deliberately not opened.

| Finding | Status | Evidence and limit |
| --- | --- | --- |
| Robot generation | PARTIALLY VERIFIED | `aist_fr3_config.yaml` and the modified example use `arm_id: fr3`, `fr3/fr3.urdf.xacro`, and seven `fr3_joint*` joints. This establishes the software target, not an independently read physical model/firmware identity. |
| Robot network reachability | VERIFIED | NUC route to configured `10.42.0.73` is direct through `enp86s0`; two ping replies, 0% loss, approximately 0.177 ms RTT. This does not establish FCI readiness. |
| libfranka | VERIFIED | Built/installed `libfranka.so.0.18.0`; hardware build references the workspace installation. Separate libfranka source is also tagged `0.18.0`. |
| Runtime library resolution | BLOCKED | After sourcing Humble and both NUC overlays, `ldd` still reported `libpinocchio_parsers.so.4.1.0` and `libpinocchio_default.so.4.1.0` as unresolved. Both files exist under `/opt/openrobots/lib`; ROS also has Pinocchio 4.0 libraries. The inspected clean launch environment does not resolve the linked 4.1 dependencies. No loader configuration was changed. |
| Controller binaries | VERIFIED | Installed library symbols include `JointImpedanceIKController` and custom `OpenPIJointPositionController`. Controller YAML also declares joint-state and Franka-state broadcasters and a joint-trajectory controller. |
| Live controller state | PENDING | No Franka controller process or controller-manager service was discovered in the inspected processes/domain graphs. Compiled plugins and YAML declarations do not establish loaded, active, or claimed interfaces. No controller was launched to obtain this information. |
| Cartesian command contract | VERIFIED at source only | `geometry_msgs/msg/Twist` on `franka_controller/target_cartesian_velocity_percent`; normalized input is scaled by `0.007` translation and `0.03` rotation into pose increments. `update()` applies the stored increments and ignores the supplied period. These are not verified SI velocity commands. |
| Cartesian timeout/enable | BLOCKED | The inspected controller retains the most recent increments; no explicit command-age timeout, operator deadman, or remote-owner lease was found in that source. Loss-of-command behavior was not tested. |
| Tool configuration | BLOCKED | The Cartesian controller hardcodes `fr3_hand_tcp`; `aist_fr3_config.yaml` has `load_gripper: "true"` and includes Franka gripper joint states. Robotiq geometry/TCP/load configuration remains unverified. |
| Custom joint controller | VERIFIED at source only | Subscribes to `~/commands` as `std_msgs/msg/Float64MultiArray`, requires seven finite joint targets, and interpolates to the last target. It is joint-space control and does not supply the requested Cartesian teleoperation contract. |
| Local hardware-driver changes | VERIFIED at source only | The NUC modifies joint-position rate limiting in `franka_hardware/robot.hpp` and `robot.cpp`. These changes must be included in a later compatibility review; a clean upstream checkout does not describe the installed system fully. |
| Hardware safety and FCI state | PENDING | Physical E-stop, operator enable, firmware compatibility, active FCI ownership, collision settings, workspace limits, tool calibration, and watchdog behavior were not established. User-reported readiness is recorded without treating it as motion authorization. |

The installed `FrankaRobotState.msg` declares timestamped joint state, measured EE pose `o_t_ee`, external wrenches, robot mode, and error fields. It also declares **desired/commanded** EE twists (`o_dp_ee_d`, `o_dp_ee_c`); these must not be mislabeled as measured TCP velocity. Live values and publication rates remain pending.

## 5. Robotiq Gripper Findings

The user identifies the mounted device as **“Robotiq 2F Adaptive Gripper C-49010”** and confirms that Yubi is a separate setup. Yubi is excluded from the integration plan.

| Item | Status | Evidence |
| --- | --- | --- |
| Mounted family/label | PARTIALLY VERIFIED | User-supplied label above. Robotiq documents `C-4XXXX` as a serial-number format; this label alone does not identify the 2F-85 versus 2F-140 opening variant. |
| Serial hardware | VERIFIED | NUC USB device `0403:6015`; `/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DA95C62Y-if00-port0` resolves to `/dev/ttyUSB1`. NUC user belongs to `dialout`. |
| Serial ownership snapshot | PARTIALLY VERIFIED | `fuser /dev/ttyUSB1` reported no owner. This snapshot does not prove exclusive ownership or the device at the far end of the adapter. |
| Existing NUC implementation | VERIFIED at source only | `~/robotiq/robotiq-gripper-interface-main` contains a built C++ Modbus RTU library and examples. Its README describes 2F-85, 115200 baud, 8N1, slave ID 9. These defaults were not read from the mounted gripper. |
| RLinf reuse | PARTIALLY VERIFIED | Existing `RobotiqGripper` supports the 2F family over Modbus RTU, configurable baud/slave/maximum width, and register-based width/status. NUC already has `pymodbus`; no hardware transaction was attempted. |
| ROS 2 Robotiq state/command interfaces | PENDING | No matching live Robotiq node, state publisher, service, or action was discovered. The existing `openpi_gripper_bridge.py` uses `franka_msgs/action/Move` and Franka finger states, so it does not provide a Robotiq bridge. |
| Physical calibration/status | PENDING | 85/140 variant, real serial settings, measured width calibration, faults, tool offset, limits, power state, and safe connection lifecycle. |

`RobotiqGripper.connect()` reaches `_open()` and activation. It was not called. The C++ activation/open/close examples and RLinf hardware test scripts were not executed. A future bridge must distinguish requested opening, register feedback, and grasp success; RLinf's `is_open` is command-derived.

## 6. SpaceMouse and Camera Findings

Peripheral detection and camera capabilities are **VERIFIED**; live images, six-axis readings, and synchronized timing are **PENDING**.

| Peripheral | Observed identity/interface | Status |
| --- | --- | --- |
| SpaceMouse | 3Dconnexion SpaceMouse Wireless BT, USB `256f:c63a`, `/dev/hidraw4`, `/dev/input/event2`, `/dev/input/js0`; HID path readable by the audit user. | VERIFIED for detection/permissions |
| RealSense D435i | Serial `327743060094`; color `/dev/video6`, depth `/dev/video2`, IR `/dev/video4`. | VERIFIED for device metadata |
| RealSense D405 | Serial `323743071035`; color `/dev/video10`, depth `/dev/video0`, IR `/dev/video8`. | VERIFIED for device metadata |
| Base/wrist placement | User describes a long RealSense at the base. D435i→base and D405→wrist is a provisional inference, not physical confirmation. | PARTIALLY VERIFIED |

Both color nodes use `uvcvideo` and report current **YUYV 640×480 at 15 fps**. Listed 640×480 color rates are D435i: 6/15/30/60 fps; D405: 5/15/30/60/90 fps. These are advertised/current settings, not measured delivery rates. Metadata-only V4L nodes reject video-format queries as expected; this is not a camera failure. No camera stream was started, no frame was saved, and no device settings were changed. No video/HID owner was reported by the inspected `fuser` query.

Use stable serial IDs rather than numeric `/dev/video*` names when configuring RLinf. Its existing `RealSenseCamera`/`CameraInfo` path supports two named cameras, serial selection, resolution, fps, and BGR8 output. Installing the matching Python binding in the chosen capture runtime remains necessary before that path can run. RGB conversion must be defined at the dataset/model boundary.

For SpaceMouse, reuse RLinf's existing Cartesian-delta reader after qualifying its required SDK, or reuse the existing ROS publisher. Select one reader. Do not launch the existing publisher during this milestone: it publishes motion and gripper intent directly to controller topic names.

## 7. ROS 2 Communication Results

IP connectivity is **VERIFIED**. Cross-host ROS discovery on domain 70 is **PARTIALLY VERIFIED**, with a reproducible interoperability warning and no Franka/Robotiq feedback available to qualify.

| Check | Result |
| --- | --- |
| Network | Workstation `eth1=10.42.0.1/24`; NUC `enp86s0=10.42.0.158/24`; SSH works over that subnet. Both hosts also have other interfaces/routes. |
| Persistent domain settings | NUC interactive default is 42; Workstation inspected shell has no domain set. Domain 70 was applied only to temporary audit processes. |
| Domain 70 graph | Both observers discovered topic/service metadata belonging to the separate Yubi setup. NUC discovered Workstation observer parameter-service names. This provides evidence of cross-host discovery, not robot telemetry delivery. |
| Discovery limitation | `get_node_names_and_namespaces()` did not list the remote observer on either host. The Humble observer repeatedly emitted `sequence size exceeds remaining buffer` while the Jazzy observer was present. Both used `rmw_fastrtps_cpp`; the distro/RMW version difference is a candidate cause, not a proven root cause. |
| Franka/Robotiq graph | No `/joint_states` or `/franka_robot_state_broadcaster/robot_state` publishers, controller-manager services, or robot actions were discovered during the final domain-70 window. Action graph was empty. |
| Domain 42 graph on NUC | No nodes listed; topic query returned only `/parameter_events` and `/rosout`. This bounded observation does not rule out systems on other domains or external control clients. |
| QoS and message compatibility | PENDING for Franka/Robotiq: no live publishers existed in the inspected graph to inspect or receive. Installed broadcaster source supplies candidate typed state interfaces only. |
| Time synchronization | Both hosts reported `NTP=yes` and `NTPSynchronized=yes`. Pairwise clock offset, timestamp provenance, and acceptable camera/state skew were not measured. |
| Firewall | NUC `ufw status` requires root and was not escalated to root. SSH and some DDS discovery work; multicast coverage and all required UDP paths were not established. |

No ROS daemon was started or stopped. The audit observers only advertised their standard ROS node metadata/parameter services; they created no robot command publisher and invoked no controller or gripper service/action. An early observer collected separate Yubi telemetry in memory before the user clarified its scope; it failed before reporting samples and was removed from the final observer. No Yubi state is used as Franka/Robotiq evidence.

The minimal interface choice should prefer standard timestamped messages for joint/pose/wrench feedback where available. Use the actual NUC `franka_msgs` definition only when its additional fields are needed, and qualify the same definition on the Workstation. No arm-command message/topic is approved yet.

## 8. Reusable RLinf Components

Reuse the components below once their runtime and hardware contracts are qualified; none requires duplicating the learning algorithms.

| Component | Reuse assessment |
| --- | --- |
| `rlinf/robotics/robots/franka.py` | Existing arm/end-effector/camera composition and independent placement. A new compatible arm backend can fit that contract. |
| `rlinf/robotics/parts/cameras/realsense.py` and `base.py` | Existing two-camera capture by serial, with per-camera resolution/fps and frame-age handling. Hardware capture remains untested. |
| `rlinf/robotics/parts/teleop/spacemouse.py` | Existing six-axis Cartesian intent and gripper buttons; requires the supported SDK and one HID owner. |
| `rlinf/robotics/parts/end_effectors/grippers/robotiq.py` | Existing Modbus implementation, conditional on the exact gripper and safe lifecycle. Its connect path activates hardware. |
| `rlinf/envs/wrappers/collect_episode.py` and `rlinf/data/storage/lerobot/writer.py` | Existing episode collection/export supports state/actions, success labels, and extra image fields. Timestamped raw capture and agreed state/action projection must precede export. |
| `realworld_collect_data_zed_robotiq.yaml` | Reusable split-placement pattern; its ZED assumption, example serial path, and robot/control backend are not the observed deployment configuration. |
| Existing SFT and offline RECAP stages | Reuse training, returns, value fitting, advantage lookup, and CFG stages after dataset/transform alignment. |

The existing `franka_ros` arm backend imports `rospy` and starts impedance control in its connect path; it is ROS 1 and is not a read-only ROS 2 bridge. The `franky` backend opens its own direct robot connection and must not compete with a ROS 2 hardware controller.

The official `franka_spacemouse` implementation targets FR3 + Franka Hand on Humble. Its Cartesian input mapping is useful, but its hardcoded TCP, torque controller lifecycle, unstamped normalized commands, and Franka Hand action client prevent unchanged deployment here. The NUC already contains this implementation with local additions; assess it before adding another controller.

An upstream search for open `franka ROS2` PRs returned no matches in that narrow query. Open [RLinf issue #759](https://github.com/RLinf/RLinf/issues/759) overlaps the Python 3.11 versus ROS Python environment problem. This is an existing integration concern, not evidence of an available ROS 2 backend.

Sources checked include [RLinf upstream](https://github.com/RLinf/RLinf), its [Robotiq driver](https://github.com/RLinf/RLinf/blob/main/rlinf/robotics/parts/end_effectors/grippers/robotiq.py) and [RealSense driver](https://github.com/RLinf/RLinf/blob/main/rlinf/robotics/parts/cameras/realsense.py), [Franka ROS 2 Humble](https://github.com/frankarobotics/franka_ros2/tree/humble), [Franka SpaceMouse](https://github.com/frankarobotics/franka_spacemouse), [OpenPI](https://github.com/Physical-Intelligence/openpi), and the required [RECAP reference](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html). The mounted gripper label was compared with [Robotiq's serial-format reference](https://blog.robotiq.com/knowledge/what-are-the-two-parallel-locking-pins-for-on-the-2f85-and-2f140).

## 9. Integration Gaps

The following gaps prevent acceptance of the proposed recording and learning pipeline.

| Gap | Status | Required resolution |
| --- | --- | --- |
| Actual feedback and controller ownership | BLOCKED | Establish one approved NUC arm owner and obtain timestamped state without silently starting a control session. The current inspected graph supplies no Franka/Robotiq feedback. |
| ROS communication contract | BLOCKED | Resolve incomplete discovery and the Fast DDS buffer warning, then qualify actual message definitions, QoS, namespaces, rates, and freshness. Setting domain 70 alone is insufficient. |
| Runtime environments | BLOCKED | Account for Python 3.12/Jazzy, Python 3.10/Humble, and Python 3.11/RLinf without mixing incompatible `rclpy` binaries. Resolve NUC loader paths and qualify device SDK versions. Align Ray only if using distributed RLinf workers. |
| NUC-local command handling | BLOCKED | Specify operator enable, ownership, source/receipt timestamps, expiry, bounded EE targets, fault handling, and loss-of-link behavior before accepting remote commands. |
| Robotiq transport and tool model | BLOCKED | Confirm the 85/140 variant, serial settings, width mapping, TCP/load, and ownership. Existing Franka Hand configuration/bridge cannot be substituted. |
| Camera assignment/timing | PENDING | Physically confirm base/wrist serials and qualify real frames, channel order, rate, drop behavior, and state alignment. |
| State representation | BLOCKED | `RealworldInputs` requires 19 values. Franka env raw pose/velocity/force/torque/gripper fields total 20 scalars. No approved projection or measured-twist derivation exists. |
| Action representation | BLOCKED | Specify metric coordinates, rotation convention, reference frame, gripper encoding, interval, and horizon. SpaceMouse percentages and custom seven-joint commands are not interchangeable with Cartesian 7D policy actions. |
| SFT/RECAP camera contract | BLOCKED | A synthetic sample retained the wrist image under `RealworldInputs` but masked/replaced it under `FrankaEEInputs`, which current RECAP selects for `robot_type=franka` and `franka_co_train`. |
| Experiment SFT configuration | BLOCKED | The candidate YAML retains placeholder data/model paths. Its selected `pi05_franka_pnp` registration still names `RLinf/pick_red` and example checkpoint/assets paths; it is not yet a registration for this experiment's future dataset. |
| Preflight enforcement | PENDING | `validate_realworld_pi05_sample` is only referenced by its unit tests. It is not automatically invoked by the SFT loader. The file also has no standalone command-line entry point. |
| Real datasets and training assets | PENDING | User reports no experiment dataset. No real sample, norm statistics, returns, advantage files, experiment critic checkpoint, or train/evaluation split was qualified. Existing StackCube symlinks are not evidence of Franka demonstrations. |

The experiment documents correctly flag the 19D/20D and SFT/RECAP mismatches. Their hardware-unavailable descriptions are historical preparation status, now superseded by this audit's detected hardware and unresolved interfaces. Their instruction to “run” the contract file should be made explicit later: import and call the validator on a selected transform-stage sample, or provide an intentional CLI. No existing documentation was rewritten during this audit.

## 10. Recommended Minimal Implementation Plan

This is a plan for a separately authorized later milestone; no step below was implemented here.

1. **Qualify the existing NUC stack first.** Confirm physical FR3 identity/firmware and the Robotiq variant/TCP, document local driver changes, and resolve the intended libfranka/Pinocchio launch environment. Preserve the existing modified workspaces.
2. **Establish a read-only state path.** Resolve ROS discovery interoperability using a supported, explicitly versioned runtime and message set. Receive actual joint/EE/wrench/robot-health and Robotiq feedback with measured freshness. Keep ROS processes in their compatible Python environments rather than inserting system `rclpy` paths into the learning venv. A separate ROS process boundary is justified if these ABIs remain different; choose its local interface only after this qualification.
3. **Reuse local peripheral capture.** Qualify the existing RealSense driver and one SpaceMouse reader in the chosen Workstation runtime. Confirm camera roles and expose capture timestamps without launching the robot environment, which can connect/reset hardware.
4. **Add only the missing ROS 2 integration.** Prefer a small registered RLinf arm backend/client around the qualified NUC interfaces. Reuse the existing Cartesian controller only after reviewing Robotiq TCP and NUC-local enable/expiry/ownership/limits. If no compatible Robotiq ROS service exists, put the existing Modbus driver behind one NUC-owned bridge with explicit activation and read/control lifecycles; do not create a second serial writer or copy the Franka Hand bridge.
5. **Define the capture and learning schema before collection.** Preserve operator intent, accepted commands, measured state, camera timestamps, faults, and episode labels independently. Select the 19D state projection and 7D action semantics explicitly. Reuse the LeRobot writer, then align SFT/value/advantage/CFG transforms around one dataset contract, including both real camera views.

Acceptance for later work requires timestamped real feedback, a verified tool model, measured timing, explicit ownership and loss-of-command behavior, and one real sample processed consistently through the learning stages. Robot motion, data collection, SFT, and RECAP require their own subsequent scope.

## 11. Verification Results

Only the checks below were performed; configuration/source evidence is distinguished from runtime behavior throughout the report.

| Check actually run | Result |
| --- | --- |
| `git status --short --branch`, `git fetch origin`, local/remote `rev-parse` | VERIFIED: intended branch, fetched head unchanged, existing symlinks preserved. |
| Local and SSH `/etc/os-release`, `uname`, Python/package metadata | VERIFIED: environment versions reported above. |
| Host-level `lsusb`, device links/permissions, `v4l2-ctl --list-devices`, per-node `--info --get-fmt-video --get-parm --list-formats-ext` | VERIFIED: peripheral metadata and current/advertised formats only. Metadata-node video ioctls failed as expected. |
| `nvidia-smi` and PyTorch device enumeration on visible GPU 1 | VERIFIED: GPU/driver visibility and CUDA enumeration; NVML warning retained as a limitation. |
| CPU imports of RLinf/OpenPI/LeRobot/OpenCV and data configuration | VERIFIED: imports passed; no model construction or training. |
| SSH network/device/process/package/source/build inspection | VERIFIED: NUC access, serial adapter, compiled plugins, libfranka linkage, existing local modifications. |
| NUC `ldd` after sourcing ROS/workspace overlays | BLOCKED: Pinocchio 4.1 loader resolution, although required files exist. |
| NUC robot IP `ping -c 2 -W 1` | VERIFIED: two replies. FCI protocol/session was not exercised. |
| ROS node/topic/service CLI queries on domain 70 and NUC node/topic queries on domain 42 | PARTIALLY VERIFIED: graph observations above; no robot feedback. The action CLI did not accept `--no-daemon --spin-time`, so final action discovery used `rclpy.action.get_action_names_and_types`. |
| Concurrent 12-second Jazzy/Humble graph observers on domain 70 | PARTIALLY VERIFIED: some cross-host service/topic discovery, incomplete node discovery, reproducible buffer warnings; final scripts exited successfully and action graphs were empty. |
| Temporary observer harness corrections | An initial shell `set -u` conflicted with ROS setup; the next harness used a ROS CLI wrapper action API with a plain `rclpy.Node`. Both audit-script errors were corrected before the final run. They were not robot failures or system changes. |
| Synthetic CPU contract + SFT/value-transform comparison | VERIFIED: state `(32,)`, actions `(5,32)` after padding. SFT masks were `[true,true,false]`, wrist mean 127; RECAP masks were `[true,false,false]`, wrist mean 0 for the same nonzero wrist image. No dataset or statistics were produced. |
| Existing contract unit tests | Earlier in this chat, at the same unchanged branch head: **7 passed** with `.venv/bin/python -m pytest -q --noconftest tests/unit_tests/test_franka_pi05_contract.py`. Not rerun during Milestone 1A. This bypassed repository-wide conftest and is a standalone CPU contract check. |
| Documentation comparison | Focused `docs-check` source/reference audit. Sphinx/EN-ZH build and markup harnesses were not run: no published RST pages changed, and the user requested minimal verification and English-only audit output. |

Initial sandbox USB, GPU, SSH, and netlink probes failed because the sandbox restricted those resources. Approved host-level probes superseded those results; they were not diagnosed as hardware faults. No broad hardware test suite, model inference, ROS bringup, FCI test, gripper test, or collection command was run.

## 12. Pending Items

| Item | Status |
| --- | --- |
| Independent physical robot model/firmware and current FCI ownership/readiness | PENDING |
| Robotiq 85/140 variant, actual serial configuration, width calibration, TCP/load and fault feedback | PENDING |
| Physical D435i/base and D405/wrist assignment | PENDING |
| Live two-camera frames, SpaceMouse axes/buttons, timestamps and delivered rates | PENDING |
| Complete ROS discovery, message/QoS qualification and real arm/gripper state delivery | BLOCKED |
| Reproducible NUC runtime dependency resolution and compatible Workstation device SDKs | BLOCKED |
| Controller ownership, operator enable, command expiry and physical safety verification | PENDING |
| Agreed 19D/7D state/action representation and two-camera SFT/RECAP parity | BLOCKED |
| Real demonstrations, rollout labels, norm statistics and learning-stage validation | PENDING |

**Milestone 1A stops here.** No Milestone 1B implementation, robot motion, gripper actuation, dataset collection, SFT, or RECAP was started. This report was prepared with an AI assistant; it records observations and limitations, not hardware deployment approval.
