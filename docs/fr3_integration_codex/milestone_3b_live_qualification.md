# Milestone 3B — Live multimodal observation qualification

This is a **read-only** 60-sample live check, not robot teleoperation,
data collection or policy inference. It reuses the Milestone 3A
`assemble_fr3_multimodal` contract and the pre-existing ROS 2
`RosStateSubscriber` and RLinf `Camera` drivers.

## Setup

Keep the working Franka controller on ROS domain 70 and the working
Robotiq ros2_control on domain 71. The cross-domain SpaceMouse command
bridge is optional; the qualification process does not own or open the HID
device, Modbus serial port, controllers, or robot action interfaces.

Workstation: source its ROS 2 Jazzy setup and the existing Python
environment. Two physical camera serials must be specified explicitly.
Earlier inventory listed D435i 337122075473 and D405 218622277827;
their **physical base/wrist assignment must be checked**, not inferred
from the camera model.

## Run

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git checkout branch-openpi-recap-franka
git pull --ff-only origin branch-openpi-recap-franka

export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Verify camera serial numbers before running:
rs-enumerate-devices -s

# Substitute verified serial-to-role mapping.
PYTHONPATH=. python -m toolkits.realworld_check.fr3_live_qualification \
  --base-serial VERIFIED_BASE_SERIAL \
  --wrist-serial VERIFIED_WRIST_SERIAL \
  --summary-json ~/fr3_m3b_live_report.json
```

The two subscriber subprocesses run with **domain IDs 70 and 71**.
The defaults expect:

- Domain 70: `/franka_robot_state_broadcaster/measured_joint_states`,
  `/franka_robot_state_broadcaster/current_pose`,
  `/franka_robot_state_broadcaster/external_wrench_in_base_frame`.
- Domain 71: `/joint_states`, which must include
  `robotiq_85_left_knuckle_joint`.

If your actually deployed names differ, use `--joint-topic`,
`--pose-topic`, `--wrench-topic` or `--gripper-topic`. This
instrument does not auto-select ambiguous ROS topics.

## Acceptance gates

- At least 95% of samples must form fresh, valid multimodal observations.
- At least 60% of sample iterations must show distinct timestamps for
  each of four ROS state sources and both cameras.
- Two camera receipt timestamps must differ by at most 0.1 seconds
  on each accepted sample, and neither may exceed 0.5 seconds old.
- The summary records incomplete runs as FAIL and includes failure reasons.

These are operational sampling thresholds, **not** exposure-time
synchronization, cross-host clock alignment, or a policy-ready OpenPI
representation. FR3 arm joint states are expected to be seven distinct
`fr3_joint1` to `fr3_joint7` readings; gripper feedback is a measured
joint angle, not calibrated jaw width.

The tool writes a JSON report but does not save camera frames or
demonstrations. Hardware control remains entirely with its existing
controllers.
