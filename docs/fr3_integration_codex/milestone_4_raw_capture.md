# Milestone 4 — Phase A: read-only raw recording pilot

This is an initial **unlabeled raw capture**, not LeRobot export, a safe
teleoperation controller, a dataset of successful demonstrations, or a
policy action specification. It builds on the **completed 3A and 3B**
FR3/Robotiq/dual-camera observation qualification.

## What is captured

`toolkits/realworld_check/fr3_raw_capture.py` uses ROS 2 subscriptions
(domain 70 for FR3, domain 71 for Robotiq) and exclusive direct SDK
RealSense camera reads. Unlike the old
`FrankaObservation` utility, it **never opens a SpaceMouse HID reader**
and does not send motion commands.

- `states.jsonl`: each accepted observation has seven measured FR3 joints
  (position/velocity/effort), TCP pose, wrench and Robotiq measured
  knuckle angle, source/topic/frame/timing provenance and image file paths.
- `frames/base/*.npy` and `frames/wrist/*.npy`: lossless source BGR
  frames at their original configured resolutions.
- `operator_commands.jsonl`: every received ROS SpaceMouse
  `geometry_msgs/Twist` value and gripper open-fraction
  `std_msgs/Float32` event, in a separate asynchronous event stream
  with subscriber receipt stamps and source topics.
- `manifest.json`: task text, device IDs, topic mappings, raw-capture
  status, failure counts, and explicit training-data limitations.

Event buffering protects button events between image samples; a detected
overflow marks the recording incomplete. Missing camera or ROS samples
are dropped with failure reasons, not replaced by fake data.

ROS topic defaults on the existing workstation:
`/franka_controller/target_cartesian_velocity_percent` and
`/gripper_client/target_gripper_width_percent`. Verify the topic names
in the actually running SpaceMouse publisher. Both represent requested
inputs. Neither is a metric robot action or proof of execution.

## Workstation pilot, *without touching the SpaceMouse*

The two RealSense **ROS publishers must be stopped** before SDK capture.
Keep existing FR3 and Robotiq controllers running. Do not start a second
camera or SpaceMouse driver. Source ROS Jazzy and the project environment.
No motion is needed: leave the SpaceMouse idle for this first pilot.

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git checkout branch-openpi-recap-franka
git pull --ff-only origin branch-openpi-recap-franka
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=70
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Offline tests first, avoiding Jazzy's Python 3.12 pytest plugin
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit_tests/test_fr3_raw_capture.py \
  tests/unit_tests/test_fr3_live_qualification.py \
  tests/unit_tests/test_fr3_multimodal.py

python -m toolkits.realworld_check.fr3_raw_capture \
  --output "$HOME/fr3_raw_pilot_$(date +%Y%m%d_%H%M%S)" \
  --task "stationary observation pilot" \
  --base-serial 337122075473 \
  --wrist-serial 218622277827 \
  --base-width 640 --base-height 480 \
  --wrist-width 848 --wrist-height 480 \
  --camera-fps 30 --fps 10 --samples 30
```

The pilot should emit `status: raw_recorded` and produce actual
lossless camera frames. `operator_events` can be zero while the
SpaceMouse publisher is stopped; this means operator-command capture
has **not** been verified yet.

## Verified workstation pilot — November 6, 2026

**Milestone 4A: COMPLETE for a stationary, read-only raw-capture pilot.**

The operator reported **22/22** focused CPU tests passed:
`test_fr3_raw_capture.py`, `test_fr3_live_qualification.py`,
and `test_fr3_multimodal.py`.

The real workstation recorder reported:

| Measurement | Result |
| --- | --- |
| Status | `raw_recorded` |
| Accepted raw observation frames | 30 / 30 |
| Operator ROS events received | 301 |
| Lost operator events | 0 |
| Capture failures | none |
| Duration | 3.005099795 seconds |
| Observation rate | about 9.98 Hz |
| Local data path | `/home/muttaqien/fr3_raw_pilot_20261106_183056` |

The local dataset directory was **not independently inspected**
after the run. The 301 events have not been categorized into
motion versus button messages, nor have executed arm commands or
moving-task success labels been validated.

The recorder correctly reports `lerobot_ready: false` and
`training_action_labels: false`. In particular, normalized
SpaceMouse ROS values are not metric Cartesian action labels.

**Next gate:** inspect the raw command-event types and recorded
frame metadata, then separately qualify supervised FR3 motion,
controller-side command acknowledgement and timeout safety before
recording labeled teleoperation demonstrations.

## Verified button-event pilot — November 6, 2026

**Milestone 4B.2: COMPLETE for ROS command publication and raw event capture.**

The operator used `ros2 topic info -v` to observe one
`spacemouse_publisher` publisher on each of the command interfaces:

- `/franka_controller/target_cartesian_velocity_percent`:
  `geometry_msgs/msg/Twist`, RELIABLE publisher with a BEST_EFFORT
  subscriber (QoS compatible).
- `/gripper_client/target_gripper_width_percent`:
  `std_msgs/msg/Float32`, RELIABLE publisher and subscriber.

A separate read-only `ros2 topic echo` verified the button
publisher sent alternating `0.0` and `1.0` requests (four pairs).
These requests are not by themselves evidence of gripper execution.

The operator then ran a second raw capture. The local output was
`/home/muttaqien/fr3_raw_buttons_20261106_183614`.

| Measurement | Observed |
| --- | --- |
| Recorder result | `raw_recorded` |
| Raw observation records | 100 / 100 |
| Total operator events | 1008 |
| `operator_twist` events | 1002 |
| `operator_gripper_request` events | 6 |
| Gripper button sequence | `[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]` |
| Dropped command events | 0 |
| Capture failures | none |
| Elapsed time | 10.011938318013563 seconds |
| LeRobot readiness / training action labels | false / false |

The user inspected the saved `operator_commands.jsonl` and its
manifest to confirm the six distinct gripper-request events and zero
dropped events. This validates asynchronous command event *recording*,
but does not establish arm motion safety, executed action semantics,
or measured causal effects of individual button requests.

**Next: Milestone 4B.3, read-only controller safety audit.**
Identify the deployed arm controller and verify its loss-of-command
timeout, exclusive operator command authority, and safety limit
configuration before supervised **moving-arm** demonstration capture.
Do not introduce any additional robot command publisher during audit.

## Controller safety configuration audit — November 6, 2026

**Milestone 4B.3: IN PROGRESS — watchdog implementation not yet verified.**

On NUC ROS domain 70, live arm joint-state feedback was observed at
approximately 1000 Hz and `ros2 control list_controllers` reported
three active controllers: `franka_robot_state_broadcaster`,
`joint_state_broadcaster` and
`franka_arm_controllers/JointImpedanceIKController`.

A read-only ROS parameter dump from
`/joint_impedance_ik_controller` reported:

- `arm_id: fr3`
- `command_timeout: 0.25` seconds
- `ik_update_rate: 50.0` Hz
- Seven joints `fr3_joint1` to `fr3_joint7`
- `load_gripper: 'false'` (the Robotiq has a separate controller)

The NUC source location was found at
`/home/aist-franka/Desktop/aist-gamma/franka_controller_ws/src/franka_arm_controllers/src/joint_impedance_ik_controller.cpp`.
`ros2 pkg prefix franka_arm_controllers` returned
`Package not found` in a shell that had sourced only ROS Humble,
suggesting the custom controller overlay was not sourced for that query.

The presence of a **0.25-second configured timeout is not proof** of
safe hold, limit enforcement, or a verified stop on command loss.
Before moving-arm demonstration capture, inspect the deployed C++
callback/update path to identify the timeout branch, its exact
hold/stop semantics, how command timestamps are refreshed, and whether
the running binary corresponds to the inspected source. No live
disconnect or stop test has been authorized or performed.

### Inspected C++ watchdog and hold semantics

Read-only source excerpts supplied by the operator from the NUC controller
implementation establish these **source-level behaviors**:

- The realtime `update()` uses `monotonic_ns()` and rejects intent or
  IK targets older than `command_timeout_s_` (configured to 0.25 s).
  The intended target must be valid and belong to the active
  controller activation generation.
- When command freshness fails, the controller does **not** advance
  `q_desired_`; it continues sending finite joint torques using
  `K * (q_desired - q) - D * dq_filtered + coriolis`.
  This is a **last-reference impedance hold**, not zero torque,
  instantaneous braking, or an independently verified safe stop.
- `spacemouse_callback()` time-stamps receipt on the NUC's monotonic
  clock, accepts six finite normalized axes only if each magnitude is
  <= 1, and converts invalid input to zero motion intent.
- `solve_ik_()` checks freshness of command and robot samples, computes
  a pose offset of 0.007 m per normalized linear axis and 0.03 rad per
  normalized angular axis, and rejects nonfinite / out-of-range IK
  joint solutions. These offsets are **not** SI speed or training
  action labels.
- The 1 kHz loop advances each desired joint by at most
  `velocity_limit * min(period, 0.001)` while commands are fresh.
  The gain-based torque calculation checks finite output, but the
  inspected excerpt does not show explicit torque clamping or
  independent Cartesian workspace/collision limits.
- On activation, desired joint reference initializes to measured joint
  positions; a new activation generation invalidates earlier input.

**Unverified:** the loaded executable/plugin's correspondence to this
source, actual physical response when operator commands cease,
and the independent hardware safety chain. Do not trigger an
unsupervised loss-of-command experiment. Verify deployed binary and
safety equipment before supervised moving-arm demonstrations.

### Controller workspace and process discovery

The operator sourced
`/home/aist-franka/Desktop/aist-gamma/franka_controller_ws/install/setup.bash`
on the NUC; `ros2 pkg prefix franka_arm_controllers` then resolved to
`/home/aist-franka/Desktop/aist-gamma/franka_controller_ws/install/franka_arm_controllers`.

A read-only process listing showed two distinct ROS 2 Humble
`ros2_control_node` processes:

- PID `1259561` has a command line referencing the installed
  `franka_arm_controllers/config/controllers.yaml` plus launch parameter
  files; this is consistent with the FR3 arm controller process.
- PID `1274171` has only temporary launch parameter files in its
  command line; its identity and ROS domain have **not** yet been
  established.

Package resolution and process discovery do not prove that the loaded
controller shared library corresponds byte-for-byte to the inspected
source. Inspect the running process environment and mapped plugin
libraries (read only) to establish the deployed plugin path before
considering any supervised command-loss exercise.

## Still needed to complete Milestone 4

1. Confirm source ROS command topic names and observe actual, supervised
   SpaceMouse events without starting another HID reader.
2. Verify arm teleoperation, exclusive operator command authority and
   NUC-local loss-of-command safety before collecting moving demos.
3. Record an operator-run task with separate executed-controller action
   timestamps, intervention and human-assigned success/failure/terminal labels.
4. Specify a justified 19D state projection, 7D action units, time
   alignment and episode rates. Do not equate raw SpaceMouse percentages
   to physical Cartesian action deltas.
5. Export and validate a small **real** LeRobot dataset end-to-end through
   the OpenPI SFT and RECAP data loaders. Until then,
   `is_lerobot_dataset=false` and `policy_ready=false` are intentional.

The existing direct-Modbus Robotiq driver must not be opened in parallel
with the gripper ROS 2 owner. No hardware control or SDK patches are
changed by this recording stage.
