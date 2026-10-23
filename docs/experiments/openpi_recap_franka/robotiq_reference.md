# Robotiq mounted on Franka — NUC integration reference

**User-confirmed system:** Franka arm with a **Robotiq gripper**, two Workstation cameras (base + wrist), SpaceMouse EE-space teleoperation and ROS 2 domain 70. No real demonstrations yet.

## Reuse existing RLinf components

- [RLinf Robotiq driver](../../../../rlinf/robotics/parts/end_effectors/grippers/robotiq.py): `RobotiqGripper`, registered as `robotiq` / `robotiq_gripper`. Implements direct **Modbus RTU serial** communication via `pymodbus`, for **2F-85 and 2F-140**. It does **not** require ROS to talk to the gripper.
- [Composable Franka robot](../../../../rlinf/robotics/robots/franka.py): its `declare_end_effector` already supports `gripper_type` and `gripper_connection`, independent of the selected arm backend.
- [Franka + ZED + Robotiq example](../../../../examples/embodiment/config/realworld_collect_data_zed_robotiq.yaml): `gripper_type: robotiq`, example `gripper_connection: "/dev/ttyUSB0"`, `controller_node_rank: 1`. Example is **not** an approved ROS 2 deployment config.
- [Upstream guide](../../source-en/rst_source/examples/embodied/franka_zed_robotiq.rst): documents a multi-node arrangement, SpaceMouse, Robotiq serial control and LeRobot export. Follow the design pattern when applicable, not its assumed ZED camera hardware.

## Major incompatibility with the provided SpaceMouse reference

[Franka SpaceMouse](https://github.com/frankarobotics/franka_spacemouse) ships a `gripper_manager/franka_gripper_client.py` that sends **Franka Hand** `franka_msgs` actions. That is **not** a Robotiq transport. Reuse its 6D EE-space SpaceMouse *ideas* only where compatible with the actual arm/controller. Avoid running the Franka Hand Gripper Manager on this setup.

There are two possible NUC gripper paths, pending inspection:

1. **Existing direct Modbus driver on NUC (preferred for consideration if USB-RS485 2F series):** use the upstream `RobotiqGripper` behind a narrow NUC-side ROS 2 command/state bridge. The bridge and its command interfaces must be discovered/designed after NUC inspection; do not implement an imaginary ROS 2 driver in advance.
2. **Already-running NUC Robotiq ROS 2 driver:** evaluate actual topics/actions/services and reuse it if compatible. Do not deploy duplicate writers to the same gripper.

## Confirm before any activation

- Exact gripper product/model: **2F-85, 2F-140 or other Robotiq**. The existing driver supports the stated 2F series only.
- Physical interface on the **NUC**: USB-RS485 Modbus serial device (e.g. `/dev/ttyUSB*`) or another transport. Device path is a sample, not observed hardware.
- Baud rate, slave address, maximum jaw width, power/enable mechanism, status register semantics and fault reporting.
- Appropriate EE/TCP frame for the actual mounted Robotiq (the FR3 example references `fr3_hand_tcp`, which cannot be assumed correct).
- Gripper command type and exclusive ownership, limit settings, operation timeouts and behavior when Workstation connectivity is lost.

## Safety note — initial audit must be read-only

`RobotiqGripper.connect()` in upstream calls `_open()`, which invokes **the activation sequence**. Do not call it for a read-only hardware audit. Likewise, do not execute `toolkits/realworld_check/test_robotiq_gripper.py` or the guide's open/close test until the operator has confirmed safety and authorized actuation.

At dataset recording time log **both** operator intent and gripper command, plus measured gripper status/width; do not infer successful grasp from a button event or `is_open` property alone. Configure tool offset and gripper width mapping only from verified hardware.

**Status:** verified against repository code and provided hardware description; Robotiq physical model, driver compatibility, ROS 2 interface, and any motion/actuation test remain **PENDING**.
