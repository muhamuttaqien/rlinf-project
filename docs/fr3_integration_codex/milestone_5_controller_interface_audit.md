# Milestone 5 — Read-only controller command interface audit

## Source-derived contract (not an executable policy mapping)

Previous NUC source and ROS observations establish:

- Arm controller: `franka_arm_controllers/JointImpedanceIKController` on ROS domain **70**; subscribes to `/franka_controller/target_cartesian_velocity_percent` as `geometry_msgs/msg/Twist`. The six inputs are **normalized SpaceMouse intent**, NOT metric m/s or rad/s nor the policy's proposed metric 7D actions.
- Source-level IK heuristic scales each normalized translational axis by **0.007 m** and angular axis by **0.03 rad** to propose a pose offset; joint targets then pass joint constraints and rate-limited impedance references. These are source-evidence values, **not independently certified physical workspace limits**.
- Last inspected `command_timeout=0.25` s. Stale input holds last impedance reference; **not zero torque or E-stop**.
- Robotiq is on separate ROS domain **71**. The previous operational bridge targets `/robotiq_gripper_controller/gripper_cmd` with `control_msgs/action/GripperCommand` and uses open target **0.0** / close target **0.7929**, `max_effort=50`. This does not prove continuous joint target semantics or controller acceptance.
- The repository's `toolkits/robotiq_ros2/src/command.cpp` separately implements a **Modbus command gate** with normalized [0,1] position/speed/force, one-owner lease, feedback freshness and queued dispatch. It is **not established to be deployed as the active action bridge**; do not confuse these two gripper interfaces.
- StackCube Panda π0.5 output has not been trained, normalized or semantically calibrated for this FR3 + Robotiq hardware. The raw 7D model output must remain quarantined.

## Implemented audit (read-only)

`toolkits/realworld_check/fr3_command_interface_audit.py` queries ROS 2 CLI only:

1. ROS domain 70: `ros2 topic info -v` for motion topic (type, publishers, subscriptions, endpoint names).
2. ROS domain 70: `ros2 param get` for arm controller `command_timeout`.
3. ROS domain 71: `ros2 action list -t` for the Robotiq action type.

It launches **no publisher, client goal, controller lifecycle operation, gripper action or command enable**. The graph alone cannot prove one authoritative publisher, safe stopping or physically safe robot policy actions.

### One workstation command (non-actuating)

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
source /opt/ros/jazzy/setup.bash
PYTHONPATH="$PWD" python -m toolkits.realworld_check.fr3_command_interface_audit
```

Optional focused tests (pure parsing): `PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/unit_tests/test_fr3_command_interface_audit.py`.

If your ROS domain 71 is not reachable from the workstation, `gripper_actions` will report an error. **Do not treat unreachable graph as evidence the gripper is absent.** Validate on the NUC under the approved lab procedure if needed.

## Milestone 5 remaining gate

No physical policy interface exists yet. Before building any actuation bridge, inspect the **current deployed NUC source**, agree on real policy action frames, units and gripper semantics, independent controller-enforced limits/authority/freshness, rollout failure handling and supervised go/no-go. Do not convert raw π0.5 channels to normalized SpaceMouse commands simply to make data flow. Physical evaluation belongs to Milestone 6.
