# Milestone 3B — Live multimodal observation qualification

## Qualification result — 2026-11-06 JST

**Status: COMPLETE for the Milestone 3B operational sampling gates.**
The workstation operator ran the read-only qualifier after releasing the
two RealSense ROS publishers from their exclusive camera-device ownership.
The output JSON reported `"passed": true` with all three checks true.

| Measure | Observed result | Gate |
| --- | ---: | --- |
| Valid samples | 60 / 60 (100%) | >= 95% |
| Distinct samples — FR3 joints | 60 / 60 | >= 60% |
| Distinct samples — TCP pose | 60 / 60 | >= 60% |
| Distinct samples — external wrench | 60 / 60 | >= 60% |
| Distinct samples — Robotiq joint | 60 / 60 | >= 60% |
| Distinct samples — base and wrist cameras | 60 / 60 for each | >= 60% |
| Maximum camera host-receipt skew | 32.709273 ms | <= 100 ms |
| p95 camera host-receipt skew | 24.3770542 ms | informational |
| Effective observation rate | 14.961978 Hz | target 15 Hz |
| Elapsed time | 4.010165 s | informational |
| Failure reasons | none | no invalid samples |

Deployed ROS domains were FR3 **70** and Robotiq **71**. Observed
source topics were the Franka state broadcaster's measured joints,
current pose and external wrench, plus Robotiq `/joint_states`.
The camera configuration used base D435i serial `337122075473`
(640x480 @ 30 FPS) and wrist D405 serial `218622277827`
(848x480 @ 30 FPS). The qualifier requested 15 observation samples/s.

Earlier in the same session, **29 / 29 CPU unit tests passed**
(`test_fr3_live_qualification.py`, `test_fr3_multimodal.py`,
`test_franka_observation.py`, `test_franka_pi05_contract.py`).

**Remaining boundaries (not part of 3B's completed sampling gate):**
the output retains `policy_ready: false` and
`clock_exposure_sync_verified: false`. It is a validated *raw*
observation pipeline, not a policy-specific 19D state projection
or verified cross-device exposure synchronization. Any later
training or online-control integration must qualify those separately.


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


## Diagnostics: environment and exclusive camera ownership

The RLinf live runner opens its **own** RealSense SDK camera pipelines.
It cannot open a camera simultaneously owned by a `realsense2_camera` ROS
publisher. Stop only the two camera ROS launches using Ctrl+C before running
the SDK-based qualifier. Leave the Franka and Robotiq controllers running.
Do not force-kill camera processes or modify device permissions.

Test with a compatible project Python environment without allowing ROS
Jazzy's Python 3.12 `launch_testing` pytest plugin to autoload into the
project's Python 3.11 `.venv`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest -q \
  tests/unit_tests/test_fr3_live_qualification.py \
  tests/unit_tests/test_fr3_multimodal.py \
  tests/unit_tests/test_franka_observation.py \
  tests/unit_tests/test_franka_pi05_contract.py
```

Camera profiles observed in the November 6 ROS RealSense launches:
D435i `337122075473`: 640x480x30; D405 `218622277827`:
848x480x30. The qualifier defaults to base 640x480x30 and wrist
848x480x30 with a 15 Hz sample loop. If the physical camera roles
are reversed, swap serials **and** width/height arguments accordingly.

The `/joint_states` gripper reader now selects the physical Robotiq
knuckle joint position and preserves its ROS header. NaN placeholders
for unmeasured velocity/effort of other joints no longer cause JSON
serialization errors; a nonfinite Robotiq position still fails closed.
