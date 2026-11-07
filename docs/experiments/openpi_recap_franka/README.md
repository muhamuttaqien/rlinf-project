# OpenPI pi0.5 SFT + RECAP — Franka (pre-hardware handoff)

**Status:** Pre-hardware preparation for the two-camera, EE-space teleoperation experiment; no demonstrations or rollout datasets exist yet. Not a tested robot deployment.

## Responsibility split

| Workstation (GPU) | NUC (robot) |
| --- | --- |
| OpenPI pi0.5 SFT, inference, RECAP, LeRobot data recording | Franka arm controller and **Robotiq gripper** driver/controller |
| Two local cameras (base and wrist) and SpaceMouse | Hardware safety enforcement, watchdog, controller health |
| ROS 2 client / observations, requested actions | ROS 2 server / state publication, bounded commands |

Both machines use `ROS_DOMAIN_ID=70`; this is only a DDS discovery setting, **not** an interface contract or a guarantee of connectivity. Match ROS 2 distributions, DDS settings, QoS, network routes, firewall, host clocks, and message package versions when hardware becomes available.

Robot-control feedback loops and limits belong on the NUC. The Workstation may issue high-level bounded targets but must never be the sole safety mechanism. A physically accessible emergency stop is required independently of software. Commands must be rejected unless a human-enabled mode is active; loss of connectivity, stale commands, or controller faults must lead to a safe controller-defined state.

## Verified upstream references

- `rlinf/robotics/robots/franka.py`: composable arm + end-effector, default backend `franky`.
- `rlinf/robotics/parts/arms/franka_ros.py` and `rlinf/robotics/parts/transports/ros/ros_controller.py`: **ROS 1** (`rospy`); do not reuse as a ROS 2 driver.
- `rlinf/robotics/parts/end_effectors/grippers/robotiq.py`: **existing Robotiq 2F-85/2F-140 Modbus RTU serial driver**, independent of ROS. Reuse if NUC wiring and model match; verify before use.
- `rlinf/robotics/parts/end_effectors/grippers/franka.py`: Franka Hand **ROS 1** driver, **not** the gripper in this experiment.
- `examples/embodiment/config/realworld_collect_data_zed_robotiq.yaml` and `docs/source-en/rst_source/examples/embodied/franka_zed_robotiq.rst`: existing split-node Franka + Robotiq reference; the ZED camera assumption may not apply.
- `rlinf/robotics/parts/teleop/spacemouse.py`: local `pyspacemouse` reader, Cartesian-delta intent and gripper buttons; existing code assumes local access to its device.
- `rlinf/envs/real/franka/base.py`: environment representation; internally uses a composable robot and frames, not ROS 2 topics.
- `rlinf/models/embodiment/openpi/dataconfig/__init__.py`: upstream `pi05_franka_pnp`, `pi05_franka_state`, and `pi0_realworld` train configs.
- `rlinf/models/embodiment/openpi/policies/realworld_policy.py`: real-world OpenPI adapter.
- `examples/offline_rl/advantage_labeling/recap/` and `examples/offline_rl/policy_optimization/cfg_rl/`: RECAP pipeline stages.

## Experiment decisions (2026-10-23)

- Two cameras on the **Workstation**: **base** and **wrist**. Both observations are required during recording/training; a missing image must not be replaced with a blank image in this experiment.
- Teleoperation uses native SpaceMouse six-axis EE/cartesian motion intent, **not** joint-space teleoperation. **Robotiq** open/close requests use the SpaceMouse buttons; gripper commands and measured width must be logged separately.
- The attached end effector is **Robotiq**, not Franka Hand. Exact Robotiq model, physical connection, and ROS 2 bridge on NUC remain unverified. See [robotiq_reference.md](robotiq_reference.md).
- The [Franka Robotics franka_spacemouse](https://github.com/frankarobotics/franka_spacemouse) ROS 2 implementation is a **reference**, not an automatically compatible backend: its example targets the **FR3**, while actual robot model and NUC controller have not been checked. See [spacemouse_ros2_reference.md](spacemouse_ros2_reference.md).
- Dataset inventory: **none yet**. No pretrained or synthetic data should be presented as real demonstrations. See [dataset_collection_plan.md](dataset_collection_plan.md).
- The non-executable [hardware profile](hardware_profile.example.yaml) documents intended placement without inventing actual topic names or robot IPs.

## Pre-hardware milestones

- **0A:** Record system/data contracts, plus a non-deployable starting SFT config.
- **0B:** Check only hardware-independent OpenPI input assumptions using synthetic data.
- **Later (Codex on Workstation):** inspect the actual NUC driver first; select ROS 2 topic/action/service types, message QoS, controller backend, joint/TCP frames, safety rules; only then implement any ROS 2 bridge.

## Entry points and boundaries

- SFT example: `examples/sft/config/realworld_franka_pi05_sft.yaml` (template; has unresolved dataset/model paths and upstream data registration).
- RECAP source configs: `examples/offline_rl/config/recap_compute_returns.yaml`, `recap_value_model_sft.yaml`, `recap_compute_advantages.yaml`, `cfg_rl_openpi.yaml`. Do **not** launch them unchanged for this task; see [data_contract.md](data_contract.md) and [recap_plan.md](recap_plan.md).
- Never substitute `ROS_DOMAIN_ID=70` for testing message compatibility or freshness.

## Hardware handoff acceptance criteria

1. Workstation and NUC can discover each other on domain 70 and receive real timestamped arm/gripper state (read-only first).
2. Confirm actual arm controller and **Robotiq** interfaces, model, TCP/tool geometry, frames, units, command semantics, QoS, clock sync, safety bounds, and watchdog behavior.
3. Validate both base and wrist camera streams and SpaceMouse locally and synchronize them with robot state.
4. Verify a single authority for robot commands and safe operator-enabled transitions before any motion.
5. Collect a tiny LeRobot dataset, compute OpenPI norm stats and exercise SFT/RECAP transforms with **real data** before training or deploying.

**Not verified in Milestone 0:** ROS 2 driver, NUC access, camera model/count, workspace bounds, action semantics, dataset registration, OpenPI/RECAP end-to-end compatibility, controller timing, or hardware safety.
