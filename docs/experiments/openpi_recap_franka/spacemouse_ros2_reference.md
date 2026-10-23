# EE-space SpaceMouse ROS 2 reference

Source (read from upstream `main` on 2026-10-23): https://github.com/frankarobotics/franka_spacemouse

**Important hardware distinction:** upstream states the reference system is a **Franka FR3 + Franka Hand** on **ROS 2 Humble**. Our robot's actual generation, NUC driver, firmware and controller are not yet verified. Do not deploy this controller unchanged to a Franka Panda or assume names such as `fr3_hand_tcp` match.

## Verified upstream behavior

| Component | Upstream code / behavior |
| --- | --- |
| SpaceMouse publisher | `src/spacemouse_publisher/spacemouse_publisher/pyspacemouse_publisher.py` uses `pyspacemouse` on the local host; publishes `geometry_msgs/msg/Twist` at 0.01-s timer period (~100 Hz) |
| Motion topic | relative ROS 2 `franka_controller/target_cartesian_velocity_percent`; with an empty namespace normally resolves to `/franka_controller/target_cartesian_velocity_percent` |
| Joystick axes | `linear=(-y, x, z)`, `angular=(-roll, -pitch, -yaw)`; `operator_position_front=false` flips x/y linear and x/y angular |
| Gripper | `std_msgs/msg/Float32` on `gripper_client/target_gripper_width_percent`; button 0 sets 0.0 (closed), button 1 sets 1.0 (open) |
| Arm controller | `src/franka_arm_controllers/src/joint_impedance_ik_controller.cpp`; subscribes to Twist, maps it to incremental target pose through IK and joint impedance effort control |
| Gripper manager | `src/gripper_manager/gripper_manager/franka_gripper_client.py`; translates width percentages to gripper ROS 2 `Grasp`/Homing action operations |

The topic names above describe **the reference implementation**, not a verified list of interfaces on the NUC. They are *not* automatically RLinf interfaces.

## Critical command semantics

The topic says `velocity_percent`, but the controller multiplies the linear input by `max_linear_pos_update=0.007` and angular input by `max_angular_pos_update=0.03`, then builds incremental position/orientation targets. This is **not a SI `TwistStamped` measurement in m/s and rad/s**. Do not take `Twist` values as direct 7D OpenPI action labels or replay them at a different rate.

Unlike joint-space teleoperation, our desired operator experience is moving and rotating the **tool center point (EE space)** with SpaceMouse 6DoF input, plus opening/closing the gripper with buttons.

RLinf already has `rlinf/robotics/parts/teleop/spacemouse.py` for local device reading and 6D Cartesian-delta intent, so do not run two HID readers simultaneously. Choose **either** that reader with a future ROS 2 output adapter **or** the upstream ROS 2 SpaceMouse publisher on the Workstation; both are candidate paths, not yet installed.

## Safety and integration gates for Codex

- The reference controller reads the latest Twist and updates pose targets in the control loop. In the reviewed code, no explicit command-age timeout or `deadman` check was identified. **Treat inter-PC teleoperation as unsafe until the NUC controller implements and verifies loss-of-command behavior.** A publisher stopping is not a proven stop.
- Confirm actual robot model (Panda vs FR3), ROS 2 distro and libfranka compatibility, TCP frame, controller status, activation/mode selection, velocity/acceleration limits, and emergency-stop chain.
- Publish motion only with operator enable, exclusive command ownership, and NUC-local limits/timeout. Do not relay raw policy output into an unguarded controller.
- Preserve acquisition time, coordinate frame, units/normalization, operator position, button events and robot feedback separately. ROS 2 QoS and precise message types will be finalized only from the real NUC.
- Do not copy the FR3 torque controller or launch files into RLinf before compatibility review.

**Status:** inspected upstream source; **not** tested on our Workstation/NUC/robot.
