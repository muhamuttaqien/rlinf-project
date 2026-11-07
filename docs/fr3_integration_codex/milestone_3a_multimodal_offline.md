# Milestone 3A — Offline multimodal observations

The pure-Python assembler in toolkits/realworld_check/fr3_multimodal.py
creates a raw RLinf Observation dictionary without touching any hardware.

Inputs are the existing RosStateSubscriber.snapshot() format for the FR3
arm on domain 70, the Robotiq gripper on domain 71, and two raw workstation
camera records from FrankaObservation.get_observation()["cameras"].

- FR3: measured positions, velocities and efforts for fr3_joint1 to
  fr3_joint7, TCP position/quaternion, and external force/torque.
- Robotiq 2F-85: measured robotiq_85_left_knuckle_joint angle in radians,
  not an inferred opening width.
- Cameras: base and wrist, distinct serials, BGR uint8 raw frames and
  explicit RGB channel-reversed copies.
- Provenance: ROS message headers, per-source host receipt times, camera
  serial IDs, camera receipt skew and explicit clock-alignment limitation.

Missing or unavailable state, wrong dimensions, nonfinite measurements,
reused cameras and over-limit camera receipt skew are rejected.

This output is NOT a 19D OpenPI sample, an action target, a metric
jaw-width estimate, or evidence of synchronized exposure times. No
automatic zero padding, blank camera fallback or policy inference occurs.

## Minimal test

Run from the checkout root after pulling the branch:

    python -m pytest -q tests/unit_tests/test_fr3_multimodal.py

The test fixtures are synthetic and require no ROS, robot, camera or GPU.

## Milestone 3B handoff

Live observation qualification must supply domain-70 arm states,
domain-71 gripper joint states, and both live workstation camera frames
to this contract. The existing SpaceMouse bridge forwards commands only;
do not reuse the legacy RLinf direct-Modbus Robotiq driver while
robotiq-ros owns the serial port.

Check actual topic names, feedback frequency, freshness, jitter and
camera frame drops on the real systems. Do not infer clock synchronization
from different hosts' monotonic timestamps.
