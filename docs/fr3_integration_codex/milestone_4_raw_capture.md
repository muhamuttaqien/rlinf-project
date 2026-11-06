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
