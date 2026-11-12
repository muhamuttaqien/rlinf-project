# FR3 π₀.₅ request deployment mapping — inspect before physical continuation

## Audited facts

- The 100 demonstrations produced time-weighted normalized SpaceMouse 6D
  request inputs and a *latched button request* for Robotiq; they are not
  controller-accepted robot trajectory actions.
- The request SFT model is trained with 19D state, two RGB cameras,
  10-step horizon and 7D request actions, padded to 32D.
- The inference code uses the **train-only quantile normalization** and
  calls OpenPI `Unnormalize` on predicted actions, retaining the first
  seven channels.
- The current controller subscribes to
  `/franka_controller/target_cartesian_velocity_percent` using Twist
  fields interpreted as *normalized SpaceMouse requests*, **not m/s**.
- Earlier live code multiplied the six arm axes by `0.15`, a post-training
  change. On actual Episode 3, 80 published requests produced maximum
  5.479 mm TCP displacement, almost entirely on Z. This is *not* evidence
  of task progress or proof of controller acceptance.
- The gripper bridge receives a latched open/close request and may move
  even when the arm barely moves.

## Implemented code

`fr3_pi05_request_adapter.py`: offline-testable 7D request validation,
identity arm mapping (same six axis order, no arbitrary gain), and a
three-consecutive-step gripper request filter with 0.8 s minimum interval
between new open/close goals. No values are silently clipped or renormalized.

`fr3_pi05_ros_executor.py`: shares the adapter for action validation and
gripper filtering. Two explicit mapping modes:

- `legacy_limited`: keeps existing experimental 0.15 arm command gain;
  this is **not** identical to the training request representation.
- `identity_verified`: no post-inference arm gain; sends the original
  normalized request axes unchanged. This is **not a safety certification**.

`fr3_pi05_supervised_rollout.py`: default remains `legacy_limited`.
The `--verified-request-mapping` option requests `identity_verified`,
requires the distinct console text `RUN VERIFIED`, and logs the selected
mode. **Do not select this mode until the checks below are independently
verified and physically approved.** The existing episode time, displacement,
fresh pose, exclusivity and human E-stop requirements still apply.

Invalid predictions remain rejected before publication and saved to
`invalid_policy_action.json`; such attempts are marked `aborted`.

## Offline audit — no robot movement

```bash
git pull --ff-only origin branch-openpi-recap-franka

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit_tests/test_fr3_pi05_request_adapter.py \
  tests/unit_tests/test_fr3_pi05_deployment_audit.py \
  tests/unit_tests/test_fr3_pi05_ros_executor.py \
  tests/unit_tests/test_fr3_rollout_outcomes.py

python -m toolkits.realworld_check.fr3_pi05_deployment_audit \
  --episode /data1/Muttaqien/fr3_pi05_real_rollouts_20/episode_03
```

The audit checks the exported train/validation source split, the provenance
of train-only normalization, channel count, training quantile range, and
recorded Episode 3 motion. It intentionally reports
`physical_identity_execution_approved=false` because a static dataset
cannot verify the physical controller.

## Still required before selecting identity deployment

1. Inspect the **deployed NUC controller binary/source**, mounting
   orientation, axis signs, pose-offset conversion and update period.
   Historical documentation disagrees on rotation scale (0.03 vs 0.06).
2. Verify actual command acceptance and measured TCP response using a
   separately approved supervised calibration procedure (not π₀.₅
   open-loop commands); distinguish normalized *requests* from SI velocity.
3. Check observation receipt vs action hold/execution timing. The trained
   label averages input over ~100 ms; the physical controller may reuse its
   most recent request at a different rate. Same FR3 hardware does not
   guarantee equivalent time discretization or coordinate frames.
4. Verify the Robotiq button-to-goal convention and goal acceptance; new
   debounce reduces toggling but does not prove correct pick behavior.
5. Obtain local operator approval of the motion limits, collision
   clearance, exclusion zone, and tested emergency-stop behavior.

A real peg-insertion policy rollout can be evaluated only *after* verifying
these; do not infer task success from ROS command count, inverse
normalization, or the same physical robot.

## Supervised arm-only mapping check (single pulse, not a policy rollout)

The user independently inspected the active NUC process and source:
- `/proc/2676273/maps` identified the running library in
  `~/Desktop/aist-gamma/franka_controller_ws/build/franka_arm_controllers/libfranka_arm_controllers.so`.
- The October 26 build timestamp is later than the controller source timestamp.
  This is evidence of the build order, **not a cryptographic binary/source match**.
- `arm_mounting_orientation=[0,0,0]` and `command_timeout=0.25` seconds.
- The inspected C++ applies `0.007 * normalized_linear_request` and
  `0.03 * normalized_angular_request` to its Cartesian IK target,
  rejecting stale/out-of-range normalized requests.

The prior arm pulse utility now accepts the exact single
`geometry_msgs/msg/Twist` subscriber when ROS Jazzy reports its NUC
node name as `_NODE_NAME_UNKNOWN_`, while still rejecting competing
publishers, extra subscribers, wrong type, stale pose and excessive travel.

On the **workstation**, first run the entirely read-only graph/pose check:

```bash
git pull --ff-only origin branch-openpi-recap-franka
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit_tests/test_fr3_supervised_arm_pulse.py

ROS_DOMAIN_ID=70 ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET \
RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
/usr/bin/python3 -m toolkits.realworld_check.fr3_supervised_arm_pulse
```

Only following a physically verified clear workspace and local operator
approval, a **separate arm-only calibration** can be considered using
the existing interactive `--execute` option. It sends one small,
bounded X request, requires a fresh pose and publisher exclusivity,
publishes neutral requests, and measures the TCP delta over the
post-pulse observation window. The utility does **not** send gripper
commands, use π₀.₅ predictions or increment RECAP rollout counts.

**Do not use this result to enable identity π₀.₅ execution automatically.**
A single short X-axis check cannot establish all axes, signed frame
alignment, timing equivalence, full trajectory safety, or controller
accepted-target tracking. The neutral request is not a safety-rated
stop. The Franka hardware E-stop must remain accessible.
