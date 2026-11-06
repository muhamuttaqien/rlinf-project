# Milestone 4C.2 — proposed 19D state and 7D action contract

**Status: candidate implementation for offline review; production contract NOT approved.**

The existing OpenPI `RealworldInputs` requires exactly 19 state values,
while RLinf's Franka env representation is 20D. Neither the raw recorder
nor the ROS SpaceMouse stream provides trustworthy (N,7) *training*
actions. Do not truncate 20D to 19D or treat normalized `Twist` as
metric motion.

## Explicit candidate 19D state (preview only)

A physically interpretable option based on fields measured by the 4A
FR3+Robotiq recorder:

| Indices (zero-based) | Source | Representation and units |
| --- | --- | --- |
| 0–2 | measured TCP pose | x/y/z, metres, frame to verify |
| 3–6 | measured TCP pose | quaternion x/y/z/w, dimensionless, frame to verify |
| 7–13 | FR3 measured joint positions | joints 1–7 in radians |
| 14–16 | measured external wrench force | x/y/z in newtons, source frame to verify |
| 17 | measured Robotiq left knuckle position | radians (NOT aperture width) |
| 18 | measured external wrench torque | L2 norm of 3D torque in N·m |

This yields **7 + 7 + 3 + 1 + 1 = 19**. The deliberate compromise is
retaining the Robotiq feedback, TCP quaternion and all joint positions
while reducing torque direction to magnitude and omitting velocities
and effort. This is an **engineering proposal**, not a validated
best-performing policy state. Selecting the force/torque features
and mapping gripper angle are open research design decisions.

The preview enforces finite physical readings, exact widths, and an
approximately unit-length TCP quaternion. It does not silently
normalize, invent gripper width, or make data training-ready.

## Proposed future 7D action semantics (no labels generated)

| Indices | Proposed meaning |
| --- | --- |
| 0–2 | base-frame TCP translation displacement (metres) |
| 3–5 | base-frame TCP rotation-vector displacement (radians) |
| 6 | **candidate** Robotiq controller knuckle joint-position target (radians, unit/joint mapping pending confirmation) |

This is a **candidate future target contract**, not a mapping from
SpaceMouse axes. It requires controller-side evidence of accepted
metric EE setpoints, verified Robotiq controller joint-position
semantics, reference and rotation frames, tool/TCP calibration,
exact action interval and a verified timebase. The
present domain-70 `JointImpedanceIKController` takes normalized
SpaceMouse inputs and computes IK pose offsets (0.007 m, 0.03 rad
scales); domain-71 Robotiq has a distinct control path. Neither is
an observed executed 7D action label. In particular, measured pose
differences are motion observations and must not be mislabeled as
accepted controller commands.

Action samples must eventually be aligned to observations at an
explicitly approved sampling interval, with actual controller acceptance
and gripper command provenance, not merely timestamped input intent.

## Offline usage (workstation, no ROS or hardware access)

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" \
python -m pytest -q \
  tests/unit_tests/test_fr3_policy_contract.py \
  tests/unit_tests/test_fr3_lerobot_preflight.py \
  tests/unit_tests/test_franka_pi05_contract.py

python -m toolkits.realworld_check.fr3_policy_contract \
  --raw "$HOME/fr3_raw_buttons_20261106_183614" \
  --output "$HOME/fr3_4c2_contract_$(date +%Y%m%d_%H%M%S)"
```

The command revalidates the whole raw dataset, creates a **new**
`contract_candidate.json` and `state_candidate_first.npy`
(19D first observation only), and reports unresolved approvals.
No `actions` are exported, and the result must not be passed to a
policy trainer. The original raw capture is untouched.

## Real workstation candidate validation — November 7, 2026

The operator ran focused CPU-only unit suites in the `aist-gamma`
Python 3.12.3 environment:

- `test_fr3_policy_contract.py`: **11 passed**
- `test_fr3_lerobot_preflight.py`: **7 passed**
- `test_franka_pi05_contract.py`: **7 passed**
- **Total: 25 passed in 0.22 seconds**

The offline candidate CLI successfully revalidated the previously
recorded `fr3_raw_buttons_20261106_183614` dataset, containing
**100** genuine raw observations, and created a review-only
`fr3_robotiq_policy_contract_candidate_v1` report.
It reported `state_dim=19`, `float32`, and source-header
`frame_id=base` for both TCP pose and external wrench.

The output also explicitly indicated:
`status=candidate_requires_review`,
`action_interval_s=null`, `executed_action_source=null`,
`gripper_width_calibration_verified=false`,
`action_labels_available=false`,
`le_robot_export_ready=false`, `sft_ready=false`, and
`recap_ready=false`.

**Interpretation:** candidate code and the first-state projection
are offline validated, but the CLI only projects the first
observation to 19D; the separate raw preflight checks all 100
state/image records. Do not claim that a full 100-frame projected
policy-state dataset or a usable 7D-action training set exists.

Next: inspect actual candidate values and frame-specific semantics,
verify gripper measurement calibration, and qualify the action
definition independently of normalized SpaceMouse intent.

## Full 100-frame candidate state projection — November 7, 2026

User independently evaluated `candidate_19d_state` on **all 100**
real records from `fr3_raw_buttons_20261106_183614`.
The resulting array had **shape `(100, 19)`** and
`np.isfinite(states).all() == True`.

Reported feature ranges (rounded as printed):

| Index | Feature | Range |
| --- | --- | --- |
| 0 | `tcp_x_m` | 0.50436 to 0.50437 |
| 1 | `tcp_y_m` | 0.11565 to 0.11569 |
| 2 | `tcp_z_m` | 0.58937 to 0.58939 |
| 3 | `tcp_quat_x` | 0.92297 to 0.92298 |
| 4 | `tcp_quat_y` | 0.36862 to 0.36864 |
| 5 | `tcp_quat_z` | 0.10818 to 0.10820 |
| 6 | `tcp_quat_w` | 0.02280 to 0.02281 |
| 7 | `fr3_joint1_rad` | 0.05049 to 0.05052 |
| 8 | `fr3_joint2_rad` | -0.22732 to -0.22729 |
| 9 | `fr3_joint3_rad` | 0.15134 to 0.15139 |
| 10 | `fr3_joint4_rad` | -1.70710 to -1.70705 |
| 11 | `fr3_joint5_rad` | 0.02772 to 0.02773 |
| 12 | `fr3_joint6_rad` | 1.70381 to 1.70385 |
| 13 | `fr3_joint7_rad` | 0.22661 to 0.22664 |
| 14 | `wrench_force_x_N` | -1.40366 to -1.01316 |
| 15 | `wrench_force_y_N` | 0.21502 to 0.40690 |
| 16 | `wrench_force_z_N` | -1.58367 to -1.04178 |
| 17 | `robotiq_left_knuckle_rad` | 0.00000 to 0.78591 |

**Feature 18**, `wrench_torque_magnitude_Nm`, was **not in
the shared terminal excerpt**; its numeric range remains unreported,
although the combined finite check and shape passed.

Interpretation: the 19D candidate can be constructed for the
entire 100-frame raw pilot, TCP and joint positions vary minimally,
and measured gripper knuckle position changes substantially during
the button pilot. This is offline integrity evidence, **not**
validation of gripper opening-width calibration, external wrench
semantics, action mapping, task demonstrations or policy readiness.

## Recorded gripper response and torque-range analysis — November 7, 2026

The operator additionally evaluated the previously omitted projected
feature 18 over the same 100-frame capture:

- `wrench_torque_magnitude_Nm`: **0.68445885181427 to
  0.8850111961364746 N·m** (L2 norm, not a signed torque component).

Using workstation monotonic receipt timestamps, the operator compared
the gripper-request event time to the nearest recorded measured
Robotiq knuckle angle approximately **0.5 seconds later**:

| Normalized requested open fraction | Knuckle angle ~0.5s later |
| --- | --- |
| 1.0 (open) | 0.27594 rad |
| 0.0 (close) | 0.60777 rad |
| 1.0 (open) | 0.27594 rad |
| 0.0 (close) | 0.60777 rad |
| 1.0 (open) | 0.27594 rad |
| 0.0 (close) | No recorded measurement within ±0.2s of target |

The first five events show a **repeatable request/feedback association**
consistent with the expected open/close mapping. The sixth sample is
**unobserved**, not evidence of success or failure. The check used a
single nearest measurement with a ±0.2s window around each 0.5s
post-request target, and does not establish exact actuator response
latency or causal attribution of each request.

**Not established:** gripper aperture width in metres, physical
width-vs-knuckle calibration (including linkage nonlinearities),
command acceptance/execution on the separate domain-71 controller,
timestamp synchronization quality, or gripper end-stop state. Do not
reinterpret normalized 0/1 button requests as physical width targets
or train a 7D action channel from these readings alone.

## NUC gripper-client source audit — November 7, 2026

The operator read the following files directly from the NUC controller
workspace (`~/Desktop/aist-gamma/franka_controller_ws/src`):

- `gripper_manager/gripper_manager/franka_gripper_client.py`
- `franka_arm_controllers/openpi_gripper_bridge.py`

These sources implement **Franka Hand**, not a calibrated Robotiq
knuckle-to-aperture mapping:

1. `franka_gripper_client.py` uses `franka_msgs/action/Homing` and
   `franka_msgs/action/Grasp` with
   `franka_gripper/joint_states`. Its `_get_max_gripper_width`
   computes `2 * msg.position[0]` after homing; this is only
   meaningful for the expected Franka finger-joint representation.
   It maps normalized `target_gripper_width_percent` to an
   action target using that width. This is **not evidence of Robotiq
   metric calibration**.
2. `openpi_gripper_bridge.py` subscribes to
   `/franka_gripper/joint_states`, expects both
   `fr3_finger_joint1` and `fr3_finger_joint2`, and sums their
   positions for current width. It sends `franka_msgs/action/Move`
   on `/franka_gripper/move`, with `MAX_WIDTH=0.08` m.
   No measured `robotiq_85_left_knuckle_joint` conversion is provided.
3. **Convention mismatch:** the known SpaceMouse ROS publisher emits
   `1.0` for **open** and `0.0` for **close**; the existing
   OpenPI bridge computes `width = 0.08 * (1.0 - g)` so it treats
   `g=1.0` as **close**, `g=0.0` as **open**. These are inverted.
   Do not activate or reuse the bridge unchanged for the Robotiq.
4. Reviewing source files does **not** establish whether either
   Franka Hand node is active in the present ROS domains. The
   actual Robotiq driver/ROS2 bridge, model calibration, and aperture
   measurements require separate read-only discovery.

**Gate remains open:** verify the running **Robotiq** software
and its actual position/width semantics before finalizing the
seventh action component. Do not infer a Robotiq maximum width
from `MAX_WIDTH=0.08` or from Franka finger joint names.
No control topic, controller or gripper action was modified.

## Live domain-71 Robotiq topology — November 7, 2026

On the NUC with `ROS_DOMAIN_ID=71` the operator observed these
ROS graph nodes:

- `/controller_manager`, `/joint_state_broadcaster`,
  `/robot_state_publisher`
- `/robotiq_activation_controller`,
  `/robotiq_gripper_controller`,
  `/robotiq_gripper_status_broadcaster`,
  `/robotiq_normalized_position_broadcaster`
- `/robotiq_output_domain71`

The process command line showed a real-hardware Robotiq launch:

```text
ros2 launch robotiq_description robotiq_control.launch.py
  gripper_model:=2f_85
  com_port:=/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DA95C62Y-if00-port0
  baudrate:=115200
  use_fake_hardware:=false
```

The separate cross-domain bridge is running as
`python3 /tmp/robotiq_cross_domain_bridge.py` (observed PID
1276031). A domain-71 `ros2 topic info -v /joint_states` showed
**two RELIABLE publishers**: `joint_state_broadcaster` and
`robotiq_normalized_position_broadcaster`, and one BEST_EFFORT
`robot_state_publisher` subscriber.

**Evidence boundary:** The launch arguments verify the configured
`2f_85` real-hardware driver path and the ROS graph lists gripper
nodes; this read-only discovery does not establish exact active
controller states, numerical aperture calibration, joint-state
publisher payloads, or which outputs the bridge actually forwards.
The historical raw pilot recorded
`robotiq_85_left_knuckle_joint` position in radians, but this
must not be reinterpreted as normalized open fraction or width
without inspecting the live driver and converter.

**Next non-actuating audit:** inspect the text of the running
`/tmp/robotiq_cross_domain_bridge.py` script, domain-71 topic
types/names and a bounded one-message state observation; do not
relaunch the bridge or create a competing Modbus driver.

## Verified NUC binary-gripper bridge semantics — November 7, 2026

The operator inspected the running `/tmp/robotiq_cross_domain_bridge.py`
source. It creates two ROS 2 contexts in one process, with
`spacemouse_input_domain70` subscribing to
`/gripper_client/target_gripper_width_percent` (Float32) in
domain 70 and `robotiq_output_domain71` sending
`control_msgs/action/GripperCommand` goals to
`/robotiq_gripper_controller/gripper_cmd` in domain 71.

Explicit mapping in that script:

| Input | Bridge command | Goal `command.position` |
| --- | --- | --- |
| `>= 0.9` | `OPEN` | `0.0` |
| `<= 0.1` | `CLOSE` | `0.7929` |
| Between 0.1 and 0.9 | ignored | none |

Each goal contains `command.max_effort=50.0`. The bridge has
`queue.Queue(maxsize=1)` and replaces an unprocessed event with
the newest event. While `busy`, it does not send another goal.
It sets `busy=False` after an action result callback, and logs
accepted goals/results, but **does not capture controller-side
acceptance and execution timestamps or measured position alongside
each individual action**.

The source implements **binary endpoint control**; the observed
`GripperCommand.command.position` is a controller joint-position
target, NOT an aperture width in metres. The observed numeric
endpoint `0.7929` is consistent with a knuckle-joint angular
target, but the action server's joint mapping and SI unit semantics
still need verification against the controller configuration/URDF.
Do not assume 0.7929 is a calibrated physical opening width or
that the bridge provides arbitrary intermediate widths.

**Contract revision:** the proposed 7th training action is now
`gripper_target_knuckle_rad` rather than
`gripper_target_width_m`. The draft report version changes from
`fr3_robotiq_policy_contract_candidate_v1` (earlier offline
validation) to **`..._candidate_v2`**. This is only metadata,
not an enabled action encoder or approval to actuate. Historical
test results refer to v1; v2 requires focused workstation retesting.

The `franka_gripper_client.py` and
`openpi_gripper_bridge.py` files are Franka Hand examples and
must not be used to command the actual Robotiq.

## Active Robotiq action-controller configuration — November 7, 2026

The operator inspected the **live domain-71** controller parameters and
URDF from the NUC with read-only ROS commands:

- `/robotiq_gripper_controller`: `gripper_profile=2f_85`,
  `joint=robotiq_85_left_knuckle_joint`,
  `gripper_closed_position=0.7929`,
  `goal_tolerance=0.02`, `max_effort=50.0`,
  `action_monitor_rate=100.0`, `allow_stalling=true`,
  `stall_timeout=0.05`, `stall_velocity_threshold=0.001`,
  and `use_object_status=true`.
- Controller URDF: `robotiq_85_left_knuckle_joint` is a
  **revolute** joint with position limits `[0.0, 0.8]` (radians),
  velocity limit `0.5`, and effort limit `50`.
  Position is exposed as a ros2_control command and state interface;
  velocity, motor current, object status, fault and fault severity
  are exposed as state interfaces.
- `ros2 action info /robotiq_gripper_controller/gripper_cmd`
  showed **one action server** (`/robotiq_gripper_controller`)
  and two advertised action clients
  (`/robotiq_gripper_controller` and
  `/robotiq_output_domain71`). The bridge is thus connected to
  the active, joint-position controller, but this does not establish
  per-command execution timestamps.
- A read-only `ros2 topic echo /joint_states --once` printed the
  named knuckle joint with position
  **0.7859140969162995 rad** and velocity **0.0**. This is near
  the `0.7929` rad closed target and within the configured
  `0.02` rad goal tolerance. However, the sample was not correlated
  with a particular action goal or completion result.
- That same topic observation emitted
  `A message was lost!!! total count change:2`, and its joint
  `effort` field was `.nan`. The observed position is numeric,
  but loss warnings require monitoring, and **effort must not be
  treated as a valid finite gripper state**.

**Resolution:** the observed NUC controller and URDF **confirm that
the bridge goal position and live knuckle feedback are angular
joint positions in radians**. The v2 proposed seventh component
`gripper_target_knuckle_rad` is therefore the appropriate
*controller-target representation*, rather than
`gripper_target_width_m`. Physical aperture-width calibration
is no longer a prerequisite for this *candidate target's units*,
although it may matter for task geometry/metrics.

**Evidence boundary:** the offline v2 contract report cannot
automatically verify a live deployed robot, so its
`goal_unit_and_joint_mapping_verified=false` remains an honest
fail-closed runtime gate. The observed NUC configuration supplies
human-audited evidence, not a universal assurance for another
deployment. Neither successful physical action response nor
`GripperCommand` goal acceptance/result timestamps were recorded
here. In particular, with `allow_stalling=true`, action completion
cannot simply be equated with an unobstructed complete closure or
successful grasp. Actual training labels and end-to-end timing remain
unavailable.

The workstation also passed **11/11** focused
`test_fr3_policy_contract.py` tests for candidate v2
(0.18 seconds, Python 3.12.3).

## Read-only FR3 desired-pose/joint topic discovery — November 7, 2026

On the NUC, with the custom FR3 controller overlay sourced and
ROS domain 70, the operator listed ROS topics and found:

- `/franka_controller/target_cartesian_velocity_percent`:
  `geometry_msgs/msg/Twist` (normalized SpaceMouse intent)
- `/franka_robot_state_broadcaster/current_pose`:
  `geometry_msgs/msg/PoseStamped`
- `/franka_robot_state_broadcaster/last_desired_pose`:
  `geometry_msgs/msg/PoseStamped`
- `/franka_robot_state_broadcaster/desired_joint_states`:
  `sensor_msgs/msg/JointState`
- `/franka_robot_state_broadcaster/measured_joint_states`:
  `sensor_msgs/msg/JointState`

A grep of the locally deployed
`franka_arm_controllers/src/joint_impedance_ik_controller.cpp`
found the internal reference update
`q_desired_[i] += clamp(target.q[i]-q_desired_[i], -step, step)`,
the impedance torque calculation and callbacks for SpaceMouse
input and `solve_ik_()`. The searched excerpt did not show a
`create_publisher` or `publish(` call inside this controller.
This is limited code-search evidence; it does not establish whether
other controller code can export internal target values.

**Potential but not confirmed:** the broadcaster desired-joint and
last-desired-pose streams may expose robot-side desired/reference
states. They must **not** be labeled as the controller's accepted
Cartesian 6D policy actions without checking publisher provenance,
reference-frame definitions, sample timing and numerical behavior
during authorized teleoperation. Robot-side desired states, the
controller's internal `q_desired_`, the IK target and measured robot
pose are distinct notions until equivalence is demonstrated.

**Next safe step:** check publisher ownership/QoS and inspect one
message each from `desired_joint_states` and `last_desired_pose`
with the arm stationary. Do not publish or replay any command,
modify the running controller, or infer training action labels
from stationary telemetry.

## Live FR3 desired-state publisher and stationary samples — November 7, 2026

The operator verified two **domain-70** state streams:

| Topic | Type | Publisher | Subscribers at check |
| --- | --- | --- | --- |
| `/franka_robot_state_broadcaster/desired_joint_states` | `sensor_msgs/msg/JointState` | `/franka_robot_state_broadcaster` (1) | 0 |
| `/franka_robot_state_broadcaster/last_desired_pose` | `geometry_msgs/msg/PoseStamped` | `/franka_robot_state_broadcaster` (1) | 0 |

Both advertised RELIABLE, TRANSIENT_LOCAL publisher QoS.
Read-only `ros2 topic echo --once` produced:

- `desired_joint_states`: `frame_id=base`; seven named
  `fr3_joint1` through `fr3_joint7` positions:
  `[0.3601659238, -0.6930229664, -0.1210644022,
  -2.1624026299, -0.0658097565, 1.4754301310,
  0.3747886121]` rad; seven zero velocity fields.
  Timestamp `1793992136.026197322` (ROS message clock).
- `last_desired_pose`: `frame_id=base`;
  position `[0.3216817975, 0.0566368997, 0.5374777913]` m;
  quaternion xyzw `[0.9442261057, 0.3292437794,
  0.0015154800, -0.0057702735]`.
  Timestamp `1793992136.952256190` (ROS message clock).

These messages were sampled ~0.926 s apart and must **not** be
treated as a synchronized joint/pose pair. The echo command for
each topic printed `sequence size exceeds remaining buffer` and
`A message was lost!!!` once. This is an important CLI/transport
observation but, alone, cannot quantify sustained ROS message loss
or assert that the serialized values are invalid.

**Important provenance boundary:** these topics are emitted by
`franka_robot_state_broadcaster`, **not demonstrated to be**
the `q_desired_` computed by custom
`joint_impedance_ik_controller`, its accepted Cartesian
setpoints, or executed actions. The publisher naming does not
prove semantic equivalence with custom controller targets.
The zero subscriber count at the check means no existing graph
consumer was detected then; reading the topics via CLI
temporarily adds a subscriber. These are telemetry candidates,
not 6D action labels.

**Next non-actuating audit:** inspect the custom controller
implementation around input twist handling, `solve_ik_()`,
the per-update reference step, and any state export to establish
whether its **actual accepted internal target** is observable.
If it is not, plan purpose-built read-only target telemetry for
a separately safety-reviewed future milestone; do not modify the
running controller or synthesize 7D training actions.

## NUC FR3 controller IK and real-time reference audit — November 7, 2026

The operator inspected the deployed
`franka_arm_controllers/src/joint_impedance_ik_controller.cpp`
around real-time `update()`, `spacemouse_callback()`, and
`solve_ik_()`. The following facts come directly from the pasted
source excerpts, not from inferred ROS topic semantics:

1. `spacemouse_callback()` stores a monotonic receive timestamp and
   six normalized Twist axes, rejecting inputs if any is nonfinite or
   outside [-1,1]; `intent.moving` is true only when at least one
   accepted axis is nonzero.
2. `solve_ik_()` samples measured arm joints into `q_init_`, uses
   KDL forward kinematics to obtain a pose, then applies the rotated
   normalized linear offset **0.007 m × axis** and local RPY offset
   **0.03 rad × axis**. The linear axes are rotated using the
   inverse configured mounting orientation. This is a pose-offset
   heuristic, not a calibrated SI velocity action.
3. KDL inverse kinematics computes a candidate `target.q[0:7]`,
   which must be finite and within configured joint limits.
   The proposed target records `state_ns`, `command_ns`,
   `activation_ns` and a validity flag, and is written via
   `target_buffer_.writeFromNonRT(target)`.
4. The 1 kHz real-time update integrates `q_desired_` toward
   `target.q`, **each joint step clamped by**
   `velocity_limits_[i] * min(period.seconds(), 0.001)`.
   It only advances on moving intent, fresh intent, valid IK target,
   matching activation generation, and fresh target state/command
   timestamps. On any gate failure, it retains the previous
   `q_desired_` reference.
5. Joint impedance torque command is computed as
   `k_gains_ * (q_desired_ - q_) - d_gains_ * dq_filtered_ + coriolis`
   and written to joint command interfaces. Source excerpt uses
   `dq_filtered_ = 0.01*dq_filtered_ + 0.99*dq_`.
6. The pasted function excerpts do not contain a ROS publisher
   exporting `target.q`, the pose submitted to IK, or the
   real-time `q_desired_` reference. Earlier grep also did not
   show a `create_publisher`/`publish(` occurrence in that
   controller's searched lines, but a whole-file check remains
   prudent before asserting lack of any exporter.

**Critical semantic difference:** normalized user input,
FK-relative Cartesian offset, IK-computed seven-joint target,
rate-limited `q_desired_`, libfranka's
`franka_robot_state_broadcaster/last_desired_pose` and
`desired_joint_states`, and measured motion are **distinct
quantities**. The observed broadcaster topics must not be relabeled
as accepted controller Cartesian 6D actions without explicit
provenance proof. A held impedance reference during stale/neutral
input is not an emergency-stop guarantee.

**Consequence for 4C.2:** Proposed first six action dimensions
remain **unapproved** despite understanding the code path.
No replay, ROS command publication, LeRobot export, or training
label generation is authorized by this audit. Next step:
a read-only complete-file search for internal-target telemetry
and joint reference/state export, followed by a minimal offline
telemetry/label design if none exists. Any future changes to the
running real-time controller require separate physical safety
review and deployment approval.

## Controller target telemetry search result — November 7, 2026

The operator performed an entire-file grep on the **deployed NUC**
`franka_arm_controllers/src/joint_impedance_ik_controller.cpp`
for `create_publisher`, `publish(`, `export_state`,
`state_interface_configuration`, `target_buffer_` and
`q_desired_`. Only the following matches appeared:

```text
41: state_interface_configuration()
85: target_buffer_.readFromRT()
92: q_desired_[i] += clamp(target.q[i] - q_desired_[i], -step, step)
99: k_gains_.cwiseProduct(q_desired_ - q_) ...
240: q_desired_ = q_
319: target_buffer_.writeFromNonRT(target)
```

There is **no explicit ROS publish or state-export call matching
these patterns in this controller .cpp**. The IK target is exchanged
through an internal real-time buffer and the velocity-limited
`q_desired_` is internal to the 1 kHz controller update. Neither
has an established externally observable, time-stamped topic.
The previously inspected
`franka_robot_state_broadcaster/desired_joint_states` and
`last_desired_pose` cannot be assumed equal to those references.

**Milestone decision:** offline candidate 19D projection and the
Robotiq joint-position action representation are validated against
the observed data/configuration, but **4C.2 production 7D action
contract remains BLOCKED on 6D Cartesian action-source provenance
and clock alignment**. The current raw stationary pilot is not
a task-demonstration dataset. No 7D action labels or LeRobot
export are warranted.

A future approved telemetry design would need to expose both the
controller's IK target and the actual post-limit `q_desired_`
with activation generation, timestamp/freshness, validity, and
association to a specific input intent; and separately log
Robotiq goal acceptance/completion and measured response with
domain clock alignment. **Do not change or restart the running
real-time controller during this offline audit.**

## Final NUC controller header check — November 7, 2026

The operator searched the deployed NUC
`franka_arm_controllers/include` tree for
`q_desired_`, `target_buffer_`, `create_publisher`,
`RealtimePublisher` and `export_state_interfaces`.
Only these lines matched:

```text
joint_impedance_ik_controller.hpp:90:
  Vector7d q_{Vector7d::Zero()}, dq_{Vector7d::Zero()}, q_desired_{Vector7d::Zero()};
joint_impedance_ik_controller.hpp:94:
  realtime_tools::RealtimeBuffer<Target> target_buffer_;
```

Together with the preceding whole-`.cpp` grep, this
establishes that **no publisher or state-export interface for
the controller's `target.q` or `q_desired_` was identified in
the inspected controller implementation and include tree**.
The internal target is shared by a realtime buffer; the
post-limit reference remains internal. This is scoped to the
inspected custom controller, not an assertion about every
ROS component or every diagnostic facility on the NUC.

**4C.2 action-source gate: BLOCKED.** Existing robot-state
broadcaster `desired_joint_states` and `last_desired_pose`
are not proven equivalent to the custom controller reference;
normalized SpaceMouse Twist is input intent rather than an
accepted physical action. Neither can be used as the
authoritative six-dimensional demonstration label yet.

### Proposed offline telemetry specification (design only)

For a future safety-reviewed instrumented controller, record:

- Controller generation/activation ID and an unambiguous
  monotonic timestamp domain; source input receive timestamp,
  sampled robot-state timestamp and IK calculation timestamp.
- Six normalized operator inputs separately from their scaled,
  frame-defined Cartesian offset (do not call them metric
  velocities); IK solve success/failure and joint-limit validity.
- Seven-element solved IK joint target `target.q` **and**
  seven-element rate-limited `q_desired_` actually used by
  the torque update, preserving their difference and update time.
- Accepted-controller update status, freshness-gate decision,
  reference frame/tool transform, and source message association.
- Cross-domain Robotiq goal target, action goal acceptance/result,
  actual measured knuckle angle, and comparable timestamps.
- Bounded telemetry rate, nonblocking real-time publisher/recorder
  design, explicit loss counters and queue-drop behavior,
  no memory allocation or blocking I/O within the 1 kHz loop.

**Separate decisions still required:** whether the policy's first
six action dimensions should encode a Cartesian delta relative
to a clearly defined pose/frame or another controller-level
action quantity; controller-side pose-target logging alone does
not prove the actual RT-applied reference. Define conversion and
action sampling intervals only after comparing recorded
setpoints with the real controller update sequence.

This section is a **non-actuating offline design**, not approval
to patch, restart, command, or deploy on the live robot. Real
controller instrumentation requires an explicit lab safety
review, staging/offline tests, performance tests, and an
approved deployment/rollback procedure.

## Approval gates before 4C.2 can be marked complete

- [ ] Human selection or revision of candidate 19D feature names,
      ordering, frame definitions, unit conventions and gripper feedback.
- [ ] Verify TCP frame id and physical tool offset and quaternion order.
- [ ] Confirm normalized force and torque semantics and whether torque
      magnitude is useful compared with directional components.
- [x] Read-only NUC controller/URDF verification of knuckle-position goal units and configured limits (0–0.8 rad) completed; command acceptance/execution timestamps and runtime verification in the offline contract remain open.
- [ ] Choose the actual 7D action source and interval; confirm that
      metrics refer to accepted controller commands, not SpaceMouse intent.
- [ ] Verify frame/action cross-host timing and documented tolerances.
- [ ] Validate real human-labeled episodes, then end-to-end OpenPI SFT
      and two-camera RECAP loaders in 4C.3/4C.4.

No source code is added to the robot controller, and this module cannot
write to robot-control topics.
