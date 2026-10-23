# OpenPI pi0.5 SFT + RECAP — Franka (pre-hardware handoff)

**Status:** Milestone 0A preparation; not a tested robot deployment.

## Responsibility split

| Workstation (GPU) | NUC (robot) |
| --- | --- |
| OpenPI pi0.5 SFT, inference, RECAP, LeRobot data recording | Franka arm and gripper driver/controller |
| Locally attached cameras and SpaceMouse | Hardware safety enforcement, watchdog, controller health |
| ROS 2 client / observations, requested actions | ROS 2 server / state publication, bounded commands |

Both machines use `ROS_DOMAIN_ID=70`; this is only a DDS discovery setting, **not** an interface contract or a guarantee of connectivity. Match ROS 2 distributions, DDS settings, QoS, network routes, firewall, host clocks, and message package versions when hardware becomes available.

Robot-control feedback loops and limits belong on the NUC. The Workstation may issue high-level bounded targets but must never be the sole safety mechanism. A physically accessible emergency stop is required independently of software. Commands must be rejected unless a human-enabled mode is active; loss of connectivity, stale commands, or controller faults must lead to a safe controller-defined state.

## Verified upstream references

- `rlinf/robotics/robots/franka.py`: composable arm + end-effector, default backend `franky`.
- `rlinf/robotics/parts/arms/franka_ros.py` and `rlinf/robotics/parts/transports/ros/ros_controller.py`: **ROS 1** (`rospy`); do not reuse as a ROS 2 driver.
- `rlinf/robotics/parts/end_effectors/grippers/franka.py`: **ROS 1** message interfaces; also needs an alternative for ROS 2.
- `rlinf/robotics/parts/teleop/spacemouse.py`: local `pyspacemouse` reader, Cartesian-delta intent and gripper buttons; existing code assumes local access to its device.
- `rlinf/envs/real/franka/base.py`: environment representation; internally uses a composable robot and frames, not ROS 2 topics.
- `rlinf/models/embodiment/openpi/dataconfig/__init__.py`: upstream `pi05_franka_pnp`, `pi05_franka_state`, and `pi0_realworld` train configs.
- `rlinf/models/embodiment/openpi/policies/realworld_policy.py`: real-world OpenPI adapter.
- `examples/offline_rl/advantage_labeling/recap/` and `examples/offline_rl/policy_optimization/cfg_rl/`: RECAP pipeline stages.

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
2. Confirm actual controller interfaces, frames, units, command semantics, QoS, clock sync, safety bounds, and watchdog behavior.
3. Validate SpaceMouse and camera streams locally and synchronize with robot state.
4. Verify a single authority for robot commands and safe operator-enabled transitions before any motion.
5. Collect a tiny LeRobot dataset, compute OpenPI norm stats and exercise SFT/RECAP transforms with **real data** before training or deploying.

**Not verified in Milestone 0:** ROS 2 driver, NUC access, camera model/count, workspace bounds, action semantics, dataset registration, OpenPI/RECAP end-to-end compatibility, controller timing, or hardware safety.
