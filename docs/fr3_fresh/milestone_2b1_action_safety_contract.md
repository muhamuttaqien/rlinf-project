# Milestone 2B.1 — FR3 Action and Safety Contract Audit

Use this audit to scope a future ROS 2 action adapter without changing the
controllers that own the FR3 and Robotiq 2F-85. The arm endpoint accepts
**normalized pose offsets, not Cartesian velocities or absolute TCP poses**.
The gripper endpoints accept raw knuckle angle or normalized closure, and the
active position-controller path does **not enforce the action's `max_effort`
through the hardware effort-setting interface**. These distinctions block a
transparent replacement of the existing RLinf Franka drivers.

This audit is complete as a read-only investigation. Physical control remains
blocked pending the qualification and operator approvals below. No action adapter
was implemented, and Milestone 2B.2 was not started.

## 1. Scope and Evidence

Read the classifications as limits on each claim. A source-described watchdog
is not a demonstrated physical stop, and a configured limit is not a measured
safety property.

| Classification | Meaning |
|---|---|
| **Verified runtime** | Observed current process/file identity, ROS graph, read-only service/parameter response, or bounded state subscription. A value proves what was reported at that snapshot. |
| **Source-supported** | Established from fresh-branch RLinf source, independently inspected current NUC deployment/vendor source, build metadata, or an official reference. It does not demonstrate physical behavior. |
| **Proposed** | A future adapter contract or safeguard; not implemented or qualified here. |
| **Unverified / blocked** | Evidence or an approved physical test is missing. The affected capability must remain disabled. |

**Verified runtime:** audit date 2026-11-19; repository
`muhamuttaqien/rlinf-project`; working directory `rlinf-project-fresh`; branch
`branch-openpi-recap-franka-fresh`; initial HEAD `31e33436`; initially clean
working tree. The baseline includes the
[Milestone 1 audit](milestone_1_hardware_interface_audit.md) and
[Milestone 2A report](milestone_2a_ros2_observation_adapter.md). The open upstream
PR/issue searches for `FR3 ROS action safety` returned no matches. An AI assistant
performed the audit and wrote this report for the human submitter.

Only the fresh RLinf checkout, those milestone documents, current NUC interfaces
and standalone deployment/vendor files, and official references were used. No
previous experimental RLinf branch, its code, documentation, patches, or historical
physical-test artifacts were opened or reused. Milestone 1 excluded the deployed
controller implementation. This milestone explicitly authorizes inspecting the
current deployment; the findings below come from that independent inspection,
not a comparison with an excluded branch.

### Evidence Locations and Binary Identity

These aliases identify the inspected NUC files without treating an installation
path or package version as proof of source equivalence:

| Alias | NUC Location |
|---|---|
| `A` | `/home/aist-franka/fr3_controller_diag_production_20261118` — independent current arm deployment, outside the RLinf checkout; its source directory is not a Git worktree |
| `A-src` | `A/src/franka_arm_controllers` |
| `R` | `/home/aist-franka/Desktop/aist-gamma/robotiq_ros2_ws` |
| `R-src` | `R/src/robotiq-ros/grippers`; current clean vendor checkout, revision `f889f447984f922d1f1ae8801cdfb51e3ccdee74` |
| `H` | `/opt/ros/humble/include/gripper_action_controller/gripper_controllers` — installed Humble base controller headers, located through the Robotiq build dependency file |
| `S` | `R/install/robotiq_grippers_sdk_vendor/opt/robotiq_grippers_sdk_vendor/include/Robotiq/gripper` |
| `B` | `/tmp/robotiq_cross_domain_bridge.py` — current standalone bridge file, not a symlink into an RLinf checkout |

**Verified runtime:** arm controller-manager PID 217731 mapped
`A/install/franka_arm_controllers/lib/libfranka_arm_controllers.so`, inode
29507291, device `103:02`, without a deleted-file marker. The current file has
SHA-256 `3b16083954f38d020ffa4f8b465056e3bc5a8933f9af9109bfb6c2582e7752b4`
and ELF build ID `d2a22c4eb68322d90dec5b0863cb3a2b56ea72f0`.
Its process command line names the installed `controllers.yaml` and generated
launch parameter files. Loaded libfranka is `0.18.0`.

**Source-supported:** `A/verified_source_manifest.json` matches the independently
hashed active controller source
`src/joint_impedance_ik_controller.cpp`
(`09ae6d6c7b5b0593a994f2d03dabe154281d5c3409846aa4fde6c42de4e278d4`)
and header (`cc06da1b1482fd75ad03e627d75ec837542076d2563d6aa980992a516f1d42ce`).
`A/production_install_manifest.json` matches the installed ELF hash and selected
configuration. The production CMake cache names `A-src`, Release mode; its build
rule names this `.cpp`, and flags include `-O3 -DNDEBUG`.

**Verified runtime:** Robotiq controller-manager PID 11985 mapped these installed
files with matching current disk inodes:

| Library | SHA-256 | ELF Build ID |
|---|---|---|
| `librobotiq_controllers.so` | `0670aee691f29833cc8cafe3eace0eacb7ab2d014e923262f28db04ae6e37a41` | `56fb6485f1126faecd94a67d1b594e48b90835ce` |
| `librobotiq_driver.so` | `66360abc62d0bd992e2b58dd63fb374bcb9736f447b71094916a2f2b5fd032c6` | `88968d055d125a73fe07a7e6e6da830869ff4b0b` |

**Source-supported:** both arm and Robotiq controller build-output ELFs have the
same `.text` and `.rodata` section hashes as their respective installed copies.
Their full-file hashes differ, and the build copies contain RUNPATH entries that
the installed copies lack. These observations, manifests, and build dependencies
support source attribution; they are not an independent reproducible-build
attestation. Controller algorithm claims therefore retain the **Source-supported**
classification. No binary was rebuilt, replaced, loaded, or reconfigured.

**Verified runtime:** bridge PID 12140 identifies `B` in its command line and
exposes the corresponding graph endpoints. The current on-disk script hashes to
`c1061fede027d08cfd23349e1d0e3c9f96a8de9c7a43620cb5c058d5d752159e`.
**Unverified / blocked:** Python does not retain a mapped source file proving
that these current bytes are exactly those loaded at process start. Bridge
implementation findings below are source-supported, corroborated by endpoint
identity, not runtime demonstrations of forwarding behavior.

## 2. Native RLinf Action Architecture

Trace actions from the environment to the part before deciding what can be
reused. A six-channel environment action and a seven-value arm target belong to
different contracts even when both describe Cartesian motion.

### Data Structures and Transformations

All findings in this subsection are **Source-supported** from the fresh checkout.

| Layer | Contract and Current Behavior |
|---|---|
| [`ControllablePart`](../../rlinf/robotics/parts/base.py) | `Action` is a mapping of names to values. `action_features` declares accepted fields. `send_action()` promises the action actually sent, but concrete drivers frequently return the requested mapping; it is not a hardware acknowledgement. |
| `PartGroup` / [`FrankaRobot`](../../rlinf/robotics/robots/franka.py) | Namespaced actions such as `{"arm": {"tcp_pose": pose}}` and `{"end_effector": {"target": vector}}`; dispatch to independently connected parts. Separate parts can execute concurrently; there is no atomic arm/gripper transaction or rollback of a physical action already dispatched. |
| [`Arm`, `BaseArm`](../../rlinf/robotics/parts/arms/base.py) | Registry/lifecycle and state-selection contracts. `Arm` does not mandate a Cartesian command type. `BaseArm.STATE_FIELDS` can narrow observations; its default includes TCP velocity and Jacobian. Neither base supplies a universal command safety gate. |
| [`FrankaRobotState`](../../rlinf/robotics/parts/arms/franka.py) | TCP pose is xyz plus quaternion x,y,z,w, with separate joints, velocity, force, torque, and Jacobian. Default zero arrays are initialization values, not evidence of measured state. Panda-named helper joint limits in this module are not FR3-qualified limits. |
| [`FrankyArm`](../../rlinf/robotics/parts/arms/franky.py) | Accepts `tcp_pose` absolute xyz/quaternion and `joint_position` absolute seven-joint radians. Its trackers can further bound targets. It owns a libfranka connection and performs recovery/collision configuration on open; it cannot be used as a passive ROS attach driver. |
| [`FrankaROSArm`](../../rlinf/robotics/parts/arms/franka_ros.py) | ROS 1 absolute `tcp_pose` backend; connection starts impedance control, joint reset can stop/start controllers, and error clearing publishes recovery. It is not the ROS 2 transport under audit. |
| [`VectorActionAdapter`](../../rlinf/robotics/adapters.py) | Maps fixed flat slices to named paths and checks vector shape. It does not establish units, frames, finite values, freshness, ownership, or acknowledgement. |
| [`ActionKind`, `ActionPart`](../../rlinf/robotics/actions.py) | Distinguish Cartesian delta, absolute pose with rot6d, joint position/delta, and gripper/hand slots. There is no declared normalized measured-FK-offset or Cartesian-velocity kind. A new meaning must not silently reuse a misleading kind. |

The unwrapped [`FrankaEnv.step()`](../../rlinf/envs/real/franka/base.py) takes
seven values for a scalar gripper: six normalized pose deltas and one binary
gripper request. It clips to the action-space bounds `[-1,1]`, adds
`action[:3] * action_scale[0]` to the cached measured position, and constructs
orientation as:

```text
R_next = R.from_euler("xyz", action[3:6] * action_scale[1]) * R_measured
```

This is a left-multiplied orientation change, not an angular velocity. It applies
the gripper first, then clips the Cartesian target to configured xyz bounds and
a target-centred wrapped xyz-Euler window. `_move_action()` clears errors and
sends absolute xyz/quaternion under `arm.tcp_pose`; its returned part mapping is
ignored. Step pacing uses a configured frequency, default 10 Hz, and sleeping;
there is no per-action expiry, source timestamp, acknowledgement, or guaranteed
command-delivery interval. `np.clip` alone does not reject NaN.

[`RelativeFrame`](../../rlinf/envs/real/wrappers/transforms/relative_frame.py),
enabled by its default flag unless the task/config changes it, transforms the
first six channels through the current pose adjoint before the inner step.
It also transforms observations and reports intervention actions back in the
wrapper's public frame. Consequently, the exact wrapper stack is part of the
action contract; the outer six-vector cannot be assumed to be base-frame deltas.
[`GripperCloseEnv`](../../rlinf/envs/real/wrappers/transforms/gripper_close.py)
can remove the public gripper channel and append zero internally; its name alone
does not establish a separately enforced closed-state condition.

Joint-space support exists in `FrankyArm` and
[`DualFrankaJointEnv`](../../rlinf/envs/real/franka/dual_franka_joint.py): absolute
radians or scaled joint deltas become clipped absolute `joint_position` targets.
[`DualFrankaTCPEnv`](../../rlinf/envs/real/franka/dual_franka_tcp.py) uses absolute
xyz/rot6d waypoints, converts to quaternion, and sends Cartesian targets. These
are reusable task patterns, not proof that the active FR3 ROS controller accepts
either command. No active joint-trajectory or absolute-pose action server was
found on domain 70.

### Gripper, Connection, and Task Lifecycle

**Source-supported:** [`BaseGripper`](../../rlinf/robotics/parts/end_effectors/grippers/base.py)
uses one-element opening width in metres, `move(width, speed)`, `position`,
`max_width`, and `open`/`close`. `command()` checks element count; it returns an
open-state-change boolean, not a ROS goal result. `reset()` opens by default.
The enclosing `EndEffector.send_action()` accepts only `target` and echoes it.

**Source-supported:** [`RobotiqGripper`](../../rlinf/robotics/parts/end_effectors/grippers/robotiq.py)
is a direct Modbus driver. Opening its connection activates the gripper. Its
metre/register conversion is linear with a nominal 0.085 m opening, and its
`close(force=130)` handles values above one as register-scale fractions rather
than establishing calibrated Newtons. Its `is_open` is a cached command flag.
Do not reuse this driver or its conversion as proof of the ROS driver's semantics.

**Source-supported:** the single-arm `FrankaEnv` gripper channel is different
again: after `action_scale[2]`, a value at/below negative threshold requests
close, at/above positive threshold requests open, otherwise it leaves state
unchanged. The default threshold is 0.5; this path uses `open()`/`close()` and
waits 0.6 s after a change. It does not send a continuous width each step.

**Source-supported:** native `Connection` defers opening, rolls back failed
post-open setup, and supports idempotent disconnect. `FrankaRobot` composes
independent arm, end-effector, and camera lifecycles. Its standard builder resolves
an arm IP, so a ROS-only constructor/configuration needs explicit handling rather
than a dummy hardware address. Native `DeviceClaim` is a node-local advisory
file lock; it does not arbitrate ROS publishers or clients on other machines.

**Source-supported:** `FrankaEnv` construction connects hardware and interpolates
to a reset pose. Reset can reconfigure compliance, reset joints, move to a rest
pose, and clear errors; tasks may command grippers. `close()` disconnects owned
parts. None of these motion paths was instantiated or executed in this audit.

**Source-supported:** [`RealWorldEnv`](../../rlinf/envs/real/env.py) builds the Gym
task and wrapper stack, batches actions and observations, supplies intervention
metadata, and supports sequential action chunks and optional automatic reset.
Its fallback hold action is zero unless a wrapper supplies a representation-aware
hold. Episode termination/truncation itself is not a robot stop. Task loading in
[`real/__init__.py`](../../rlinf/envs/real/__init__.py) invokes
`realworld_setup()`, which kills ROS 1 `roscore`/`rosmaster`/`rosout` processes.
A future attach-only task must avoid that machine-level setup path and all
implicit reset/recovery behaviors. None was invoked here.

### What the Recorder Calls an Executed Action

**Source-supported:** [`collect_real_data.py`](../../examples/embodiment/collect_real_data.py)
starts a Ray worker, constructs `RealWorldEnv`, resets it, and repeatedly steps
zero vectors intended for teleoperation override. It substitutes
`info["intervene_action"]` when that key is present and stores that environment-space
vector in `TrajectoryStep.actions` and `forward_inputs["action"]`.
`RealWorldEnv` supplies intervention vectors and flags; the collector's selection
by key presence is not controller acceptance verification.

**Source-supported:** the
[`intervention wrapper`](../../rlinf/envs/real/wrappers/teleop/intervention.py)
reports the chosen operator action before inner scaling/Cartesian clipping;
frame wrappers can transform that metadata back to the outer action frame.
[`CollectEpisode`](../../rlinf/envs/wrappers/collect_episode.py) retains input
actions and infos; its LeRobot export can replace an action with a flagged
intervention. The
[`trajectory schema`](../../rlinf/data/schema/embodied_types.py) also supports
policy/intervention substitution for training. This records the selected
policy/operator command at the environment interface, not measured physical
execution or the exact accepted NUC target after all lower-layer transformations.

**Confirmed incompatibility — Source-supported:** no inspected path carries the
arm controller's accepted target/sequence into the recorder. A future record
must distinguish requested environment action, validated adapter action, wire
submission, controller acceptance when available, and measured outcome. An echo
from `send_action()`, a DDS publish return, or an action result named `SUCCEEDED`
cannot collapse those stages into verified physical execution.

## 3. Current FR3 Command Contract

The live endpoint identifies where requests arrive. The deployment source explains
what they mean; neither observation proves that a future command is safe.

### Runtime Snapshot

| Property | Evidence and Classification |
|---|---|
| Platform/domain | **Verified runtime:** NUC ROS 2 Humble, domain 70, Fast DDS library loaded; workstation ROS 2 Jazzy remains a separate runtime. |
| Controller | **Verified runtime:** active `joint_impedance_ik_controller`, type `franka_arm_controllers/JointImpedanceIKController`, claiming `fr3_joint1/effort` through `fr3_joint7/effort`. |
| Topic/type | **Verified runtime:** `/franka_controller/target_cartesian_velocity_percent`, `geometry_msgs/msg/Twist`; one controller subscription and zero discovered publishers during the probe. |
| QoS | **Verified runtime:** best effort / volatile. Discovery reported depth 0; **Source-supported:** code requests `SensorDataQoS().keep_last(1)`, so discovered depth 0 is not evidence of an unbounded queue. |
| Rates | **Verified runtime:** manager `update_rate=1000`, `ik_update_rate=50.0`; approximately 1000.03 JointState arrivals/s in a two-second NUC window. These are configuration and observation evidence, not a scheduling guarantee. |
| Freshness parameter | **Verified runtime:** `command_timeout=0.25`; `use_sim_time=false`. |
| Mount/gains | **Verified runtime:** `arm_mounting_orientation=[0,0,0]`; stiffness `[600,600,600,600,250,150,50]`, damping `[30,30,30,30,10,10,5]`; read only, not qualified as safe values. |
| Other commands | **Verified runtime:** recovery action exists, but no absolute-pose/joint-trajectory command action was discovered. Configured controller type names and unclaimed hardware interfaces are not active command endpoints or permission to switch controllers. |

An initial **Verified runtime** `list_hardware_components` response reported the
arm unconfigured with unavailable/unclaimed interfaces, despite the active
controller list. An isolated domain-70 repeat with longer discovery reported
state ID 3 (`active`), effort claimed/available, position/velocity unclaimed but
available; `list_hardware_interfaces` agreed. A separate short probe had discovery
or response timeouts on domain 70. DDS also emitted
`sequence size exceeds remaining buffer`. **Unverified / blocked:** the cause of
these inconsistent/absent status responses. A future readiness gate must reject
inconsistent evidence rather than selecting the reassuring response.

### Meaning of the Six Channels

**Source-supported:** `A-src/src/joint_impedance_ik_controller.cpp:331–402`
and its header define six finite, dimensionless channels, each in `[-1,1]`.
The topic's `percent` suffix does not mean values in `[-100,100]`. They are not
metres/second, radians/second, or dt-scaled integration increments.

Let `(p(q), R(q))` be measured forward kinematics of the URDF chain `base` to
`fr3_link8`, and `M` the RPY rotation from `arm_mounting_orientation`. The timer
constructs this IK candidate:

```text
u = [linear.x, linear.y, linear.z, angular.x, angular.y, angular.z]
p_candidate = p(q_measured) + 0.007 m * transpose(M) * u[0:3]
R_candidate = R(q_measured) * RPY(0.03 rad * u[3:6])
```

This equation describes code semantics, not a command recipe to execute.

| Channel | Source-Supported Meaning |
|---|---|
| `linear.x/y/z` | Dimensionless Cartesian offset axes; full scale is 0.007 m per component before inverse mounting rotation. With the observed zero mounting angles, the basis aligns algebraically with the chain's `base`. |
| `angular.x/y/z` | Dimensionless roll/pitch/yaw offset arguments; full scale is 0.03 rad per argument. The rotation is right-multiplied in the measured `fr3_link8` orientation basis; it is not a spatial angular-velocity vector. |
| All six zero | Marks intent non-moving and gates off reference advancement on subsequent RT processing; it does not turn off torque control. No neutral message was sent. |
| Any non-finite or out-of-range value | Replaces the entire intent with non-moving zero axes and updates receipt time; it does not saturate individual channels. No invalid command was sent. |

**Source-supported:** there is no explicit Cartesian combined-norm clamp, workspace
box, collision check, or proof of physical axis direction in this callback/IK
path. Translation is rotated by inverse mounting rotation; angular arguments do
not receive that transform. A single shared `frame_id` would misdescribe these
mixed frame conventions. The current URDF has a fixed 0.107 m joint from
`fr3_link7` to `fr3_link8`. **Unverified / blocked:** equivalence of that controlled
link with the measured Franka `O_T_EE`, mounted Robotiq TCP, and RLinf task TCP;
physical tool/extrinsic calibration; equivalence of `base` and `base_link`.

### IK, Reference Updates, and Watchdog

All algorithm findings here are **Source-supported**, with physical behavior
**Unverified / blocked**.

1. At activation, the code validates measured joints against URDF limits, seeds
   `q_desired=q_measured`, and starts a new steady-clock activation epoch. Connect
   in a future RLinf adapter must not invoke this lifecycle transition.
2. The RT loop copies measured joints and the latest intent into a sample when
   its mutex can be taken without waiting. The non-RT 50 Hz timer computes FK from
   that sample and bounded-joint KDL Newton–Raphson IK (100 iterations, tolerance
   `1e-6`). It rejects non-finite or out-of-URDF-range joint solutions.
3. Each candidate starts from **measured FK**, not the previous Cartesian
   candidate. A held nonzero request can repeatedly generate an offset ahead of
   the moving measured state. Multiplying 7 mm by 50 Hz does not establish a
   0.35 m/s commanded or achieved velocity.
4. The RT reference advances only if intent is moving/fresh, target valid, the
   activation epoch matches, and target state/command times are fresh. All these
   ages use the NUC's `std::chrono::steady_clock`; timeout is 0.25 s. There is no
   sender timestamp in `Twist`. The state timestamp is the RT sample-copy time,
   not a hardware acquisition timestamp; copying stale hardware values could
   still produce a recent sample-copy time.
5. The per-joint reference step is clamped to URDF velocity times
   `min(period.seconds(), 0.001)`. This bounds the code's reference increment;
   it does not establish actual velocity, acceleration, jerk, energy, or torque
   bounds. A delayed update is not allowed a larger reference jump.
6. Invalid IK explicitly invalidates the target. Neutral, invalid, expired, or
   unusable intent/target prevents further reference advancement. The code keeps
   the last `q_desired` and still computes
   `K*(q_desired-q) - D*dq_filtered + coriolis`, with
   `dq_filtered = 0.01*dq_filtered_previous + 0.99*dq_measured`.
7. Non-finite measured state, invalid update period, or non-finite computed torque
   returns an update error. The adapter must not automatically recover or restart
   the controller after such a failure.

**Source-supported:** the RT gate does not require target-command identity to equal
the newest intent's identity; it checks freshness and activation instead. A new
nonzero intent may coexist briefly with a still-fresh older IK target. Latest
callback receipt is not per-request target acceptance. No public command sequence,
authority token, or sender-specific ownership check exists in this path.

**Source-supported:** the source reads URDF position/velocity limits, not the
Panda constants in native RLinf. The current published FR3 URDF reports:

| Joint | Lower / Upper (rad) | Velocity (rad/s) |
|---|---|---|
| 1 | -2.9007 / 2.9007 | 2.62 |
| 2 | -1.8361 / 1.8361 | 2.62 |
| 3 | -2.9007 / 2.9007 | 2.62 |
| 4 | -3.0770 / -0.1169 | 2.62 |
| 5 | -2.8763 / 2.8763 | 5.26 |
| 6 | 0.4398 / 4.6216 | 4.18 |
| 7 | -3.0508 / 3.0508 | 5.26 |

**Verified runtime:** those are published description values.
**Unverified / blocked:** their suitability as an approved operating envelope,
position-dependent speed restrictions, actual hardware enforcement, joint margins,
tool/workspace limits, and safe operating speeds. They must not be adopted as an
RLinf safety profile merely because the source consumes them.

### What Is Observable About Acceptance and Stopping

**Verified runtime:** the controller publishes transition events, parameter events,
and ROS logs; no dedicated accepted-target/command-status topic was discovered.
**Source-supported:** its one-second `FR3_DIAG_V1` logs expose aggregate callback,
IK, reference-advance, and effort-write counters/peaks. These are useful diagnostics,
but carry no caller sequence and do not acknowledge hardware torque acceptance.
Ordinary desired-state broadcaster topics are not correlated acknowledgements of
this controller's private joint reference.

**Unverified / blocked:** message cessation, neutral requests, timeout timing,
physical settling, impedance hold under load, controller crash, NUC process death,
network failure, and stop distance were not physically tested. The 0.25 s gate is
not a guarantee of standstill within 0.25 s. Delayed DDS messages receive a new
local receipt timestamp and may renew intent after the original sender has
stopped; measured error can still cause motion while the reference is held.

## 4. Current Robotiq Control Contract

Select the gripper endpoint by its representation, then track its goal lifecycle
separately from mechanical outcome and force. The two action servers share one
physical command owner.

### Endpoints, Units, and Ranges

| Property | Evidence and Classification |
|---|---|
| Platform/domain | **Verified runtime:** Humble, domain 71; controller-manager update rate 500 Hz. |
| Controller/hardware | **Verified runtime:** active `robotiq_controllers/GripperCommandController`; claims only `robotiq_85_left_knuckle_joint/position`. Hardware active; max-velocity/max-effort setting interfaces available but unclaimed. Activation controller is also active; it was not invoked. |
| Raw action | **Verified runtime:** `/robotiq_gripper_controller/gripper_cmd`, `control_msgs/action/GripperCommand`. **Source-supported:** position is knuckle radians, not jaw metres. |
| Normalized action | **Verified runtime:** `/robotiq_gripper_controller/normalized_gripper_cmd`, same type. **Source-supported:** position is unitless closure, 0=open, 1=closed; wrapper relays into the raw action. |
| Configuration | **Verified runtime:** `gripper_profile=2f_85`, `gripper_closed_position=0.7929`, `max_effort=50`, `allow_stalling=true`, `use_object_status=true`, `object_status_timeout=10 s`, `goal_tolerance=0.02`, `stall_timeout=0.05 s`, action monitor 100 Hz. |
| Description | **Verified runtime:** driven joint is revolute; URDF range 0–0.8 rad; hardware profile `2f_85`, force multiplier 0.5, max-force scaling parameter 235.0, speed multiplier 1.0, max-speed scaling parameter 0.15. These are configuration values, not calibrated physical limits. |
| State samples | **Verified runtime:** raw and normalized JointState streams each approximately 500 Hz in a two-second window; raw frame `base_link`, normalized frame empty. A three-second typed-status subscription received 1501 messages; final object status 3, current 0 A, fault/severity 0. This is a snapshot, not assurance for future commands. |

**Source-supported:** `R-src/robotiq_controllers/CMakeLists.txt` selects the Humble
`GripperCommandController`, an `ObjectStatusController` over the installed
position-interface gripper controller, rather than the Jazzy parallel-gripper
implementation. The stock position adapter writes desired position only and
returns `max_allowed_effort` as its computed-command value. The Robotiq wrapper
uses that value in results. Consequently:

- Action `max_effort`, including the bridge's value 50, is not forwarded to the
  hardware's `set_gripper_max_effort` interface by this active position path.
- Result `effort` is not a measured grip-force report or proof of force limiting.
- Driver source seeds its separate force command from a 235.0 default times the
  configured multiplier, then scales it against its max-force parameter. With
  the current description, initialization arithmetic gives 117.5 and register
  127/255 after truncation. This is a source-level command-setting calculation,
  not a readback of the current internal command or a measurement of Newtons.

This is a **Source-supported safety incompatibility**. **Unverified / blocked:**
actual force calibration, physical force limiting, and honoring the requested
`BaseGripper.close(force=...)` contract. Do not claim that lowering an action's
`max_effort` makes a grasp safe. The
[official Humble position adapter](https://raw.githubusercontent.com/ros-controls/ros2_controllers/humble/gripper_controllers/include/gripper_controllers/hardware_interface_adapter.hpp)
corroborates the forwarding behavior; the installed header and live interface
claims supply the deployment-specific evidence.

**Source-supported:** `gripper_scaling.hpp` and selected SDK profile define the
conversion, with profile register endpoints 3 and 230:

```text
raw_joint_rad = 0.7929 * (register_count - 3) / 227
register_request = round(clamp(raw_joint_rad / 0.7929 * 227 + 3, 0, 255))
normalized_request -> round(clamp(normalized_request, 0, 1) * 255)
normalized_feedback = register_count / 255
```

Thus raw 0 rad requests register 3, raw 0.7929 rad requests 230; normalized 0 and
1 request registers 0 and 255. Their intermediate raw equivalents are about
-0.010479 and 0.880224 rad. These equations explain why `raw/0.7929` is not the
normalized stream, and why the raw URDF range is not a demonstrated action-server
range check. They do not imply physical travel beyond the gripper's mechanics.

**Source-supported:** the raw stock goal callback accepts without checking finite
values or configured joint bounds; downstream register conversion clamps to a
byte, while unusable commands can leave previous values in place. The normalized
callback rejects non-finite position, while finite out-of-range closure is clamped
with a warning by the SDK conversion. A future adapter must reject its own invalid
inputs before either endpoint, rather than depend on those heterogeneous behaviors.

### Goal, Feedback, Fault, and Timeout Semantics

The following findings are **Source-supported** from
`R-src/robotiq_controllers/include/robotiq_controllers/object_status_controller.hpp`,
`src/object_status_goal.cpp`, `H/gripper_action_controller_impl.hpp`, and the
installed message definition. None was exercised with a goal or cancellation.

| Event | Source-Supported Behavior / Limit |
|---|---|
| Goal accepted | Raw server accepts and can preempt an existing goal. The normalized wrapper may accept its outer goal before forwarding fails; outer acceptance alone does not prove admission by the raw server. |
| Status | Goal UUID and standard action statuses identify admission/terminal state. They do not identify command authority or guarantee mechanical success. |
| Feedback | Schema/graph have feedback endpoints, but the normalized relay installs result and goal-response callbacks without a feedback relay; inspected stock code does not continuously publish goal feedback. Do not require an unverified feedback cadence for readiness. |
| Terminal data | Result carries position, effort, `reached_goal`, and `stalled`. Raw position is radians; normalized wrapper converts result position to closure. Its effort value is not measured grip force. |
| Object status | Current `use_object_status=true` decides via gripper object detection and sets the internal velocity-stall timeout to infinity every update. The readable 0.05 s parameter therefore does not describe the active stall decision. |
| At target/contact | Object status 3 can complete with `reached_goal=true`. Opening/closing contact (1/2) can complete with `stalled=true`, `reached_goal=false`. Because `allow_stalling=true`, such a contact may produce action `SUCCEEDED`; task success still needs interpretation. |
| No status transition | Object-status verdict times out after 10 s without a decisive transition, with timing reset when movement is observed. A recognized repeat of the last decided request/status can reuse its verdict. Timeout produces neither reached nor stalled and an aborted result; it is not a motion-stop deadline. |
| Fault/lost serial state | Typed status distinguishes fault codes/severity and unknown 255 from healthy zero. Driver source warns that position may be stale during non-operational communication. Object-status logic does not substitute for a fault-specific, freshness-aware admission gate. |
| Completion/abort | Clearing the active goal does not itself replace the position command. The base update continues forwarding its retained command; an action timeout/abort is not an automatic release or guaranteed hold-at-current-position operation. |
| New goal/preemption | Goals on either endpoint share the raw action path; another client can preempt the current goal. There is no exclusive action-client lease. |
| Client loss | Ending a client or losing its network connection does not by itself revoke an admitted goal. A pending server-side action may continue. Physical behavior remains untested. |
| Cancellation | The normalized server forwards cancellation to the raw goal; stock raw cancellation requests a hold position. This is source inspection only, not a qualified stop mechanism. No goal was canceled here. |

The [official action schema](https://raw.githubusercontent.com/ros-controls/control_msgs/humble/control_msgs/action/GripperCommand.action)
labels position/effort as gap metres/Newtons. This deployment reuses that schema
with raw radians or normalized position, and the position adapter echoes effort.
Treat the endpoint-specific units above as mandatory metadata, rather than deriving
units from the action type name or its generic comments.

### Current Cross-Domain Bridge and Calibration

**Verified runtime:** `spacemouse_input_domain70` subscribes to
`/gripper_client/target_gripper_width_percent` (`std_msgs/msg/Float32`,
reliable/volatile), with zero discovered input publishers. `robotiq_output_domain71`
has raw-gripper action feedback/status subscriptions and send-goal/get-result/
cancel service clients. Those are the existing bridge's clients; this audit
created no action client.

**Source-supported from current `B`:** values `<=0.1` queue raw close 0.7929;
values `>=0.9` queue raw open 0; intermediate values are ignored. A single-element
queue replaces the previous pending command. A 0.1 s timer sends a queued raw goal
with `max_effort=50` only while not busy. It clears busy on rejection, exception,
or result. It has no command timestamp/expiry, finite/range check, result-wait
deadline, cancellation-on-input-loss, or external command-authority handshake.
A pending request can outlive the operator input and execute after the busy goal
finishes. An unanswered result can keep the bridge busy indefinitely. Graph
presence does not prove these callbacks were exercised or bound stale-command risk.

**Proposed:** a future RLinf gripper adapter must select one action representation,
retain the goal UUID and submission/acceptance/result states, and avoid the existing
bridge unless a named operator-approved owner controls it. No bridge was stopped
or modified here.

**Unverified / blocked:** metre-based `BaseGripper` mapping requires qualification
of physical model/fingertips and parallel/adaptive grasp mode; actual fully open
and closed readings; profile/register/joint consistency; a width-versus-state
map with units, direction, quantization, hysteresis and tolerance; behavior at
contact; and effort/speed behavior under the installed controller. The SDK's
nominal 0–0.085 m profile is useful source evidence, not a physical calibration
of these jaws. Neither `0.085*(1-normalized)` nor raw angle scaling is accepted
here as a calibrated conversion.

## 5. Compatibility Matrix and Proposed Typed Actions

Reuse the native ownership and observation structure, but expose a distinct action
meaning. The current controller cannot safely stand in for RLinf's absolute
`tcp_pose` backend by renaming a field.

| Component | RLinf Representation | Current ROS 2 Representation | Compatibility / Classification |
|---|---|---|---|
| Arm translation | Unwrapped normalized deltas scaled in metres and converted to an absolute target; wrapper may change frame | Three unitless measured-FK offset channels, inverse mounting rotation, 0.007 m full scale | **Source-supported mismatch:** selected basis, controlled link, sampling time, reference update, and semantics differ. |
| Arm rotation | Scaled xyz Euler delta left-multiplied into measured quaternion; absolute backend pose x,y,z,w | Unitless RPY offset right-multiplied into measured link8 FK, 0.03 rad full scale | **Source-supported mismatch:** composition frame/order and control point differ; no implicit conversion. |
| Absolute TCP pose | `arm.tcp_pose`, seven xyz/quaternion values; absolute rot6d task variant also exists | No discovered absolute-pose request endpoint | **Unverified / blocked:** no direct supported mapping. A pose-tracking controller would be a separate control-design change. |
| Joint action | Seven absolute radians, or task delta converted to those targets | Active controller owns effort; no discovered joint target endpoint | **Unverified / blocked:** native support does not authorize switching to another controller. |
| Gripper task signal | Negative close / positive open / intermediate unchanged | Bridge thresholds or raw/normalized GripperCommand goals | **Proposed adaptation:** explicit semantic conversion with owner/goal state; no streaming arbitrary goal every environment step. |
| Gripper part target/state | `BaseGripper` opening width in metres | Raw radians or normalized closure | **Unverified / blocked:** calibrated conversion required before claiming this contract. |
| Gripper force | `close(force=...)`, driver-specific realization | Goal max_effort does not drive the active position path's hardware force-setting interface | **Source-supported mismatch:** force contract cannot be promised without controller/driver qualification or correction. |
| Robot feedback | Native state object and task observation mapping | Milestone 2A measured subset with headers, frame IDs, receipt times | **Proposed reuse:** retain read-only acquisition and metadata; add typed fault/status readiness before actions. Missing velocity/Jacobian/TCP calibration remain explicit. |
| Timestamp | Flat/native action mapping has no required timestamp | Twist has no header; gripper command payload has no source-expiry field | **Proposed adaptation:** local deadline/provenance envelope; receiver-side source-age protection remains blocked by the legacy wire contract. |
| Command status | Returned mapping and environment intervention metadata | Arm aggregate diagnostics; gripper goal UUID/status/result | **Source-supported gap:** arm submission is not acceptance; gripper admission/terminal status is not physical execution. |

### Candidate Contract, Not Runtime Code

The following is **Proposed**. A future discriminated action envelope must reject
missing fields and an absent approved limit/timing profile. It must not silently
accept ordinary `tcp_pose` or velocity actions as these requests.

| Field | Candidate Type and Meaning |
|---|---|
| `kind` | Literal `fr3_normalized_pose_offset_v1` or `robotiq_normalized_closure_v1`; distinct from pose, velocity, joint, and width actions. |
| Arm payload | Exactly six finite floats, each in `[-1,1]`; dimensionless. Preserve the source algorithm's measured-FK reference, translation basis `M^T` into `base`, rotation basis `fr3_link8`, and right-multiplied RPY composition. |
| Arm semantics metadata | Domain 70, exact topic/type, controlled link `fr3_link8`, mounting rotation, 0.007 m and 0.03 rad scales, controller ELF fingerprint and validated parameter/URDF identity. TCP/frame calibration is required before relating this to policy/task geometry. |
| Gripper payload | One finite normalized closure in `[0,1]`, domain 71, exact normalized action endpoint; goal ID assigned at submission. Any effort request must be finite, nonnegative and within an approved profile, with nominal Newton request units distinguished from unverified enforcement. Reject force-dependent operations until enforcement is qualified. |
| Identity/provenance | Schema version, authority/session ID, monotonically increasing local sequence, operator/teleop/policy source ID, operator authorization reference, and conversion/calibration version. |
| Local time | Generation and expiry in a named host/session monotonic clock domain; IPC receipt and actual publish/submission times recorded separately. Cross-host monotonic values are never directly subtracted. |
| Observation dependency | The exact observed joint/pose sample sequences, ROS headers, local receipt times, frame IDs, and readiness/fault snapshot used for admission. |
| Limit/timing profile | Explicit operator-approved per-axis/norm/workspace/joint/dynamic constraints, receipt/source age and skew bounds, uncertainty budget, publish-rate range, and gripper admission/result deadlines. Values remain **Unverified / blocked** until qualified; absence prevents enable. |
| Disposition | Validated/rejected with reason, submitted, acceptance unknown for legacy arm, gripper accepted/rejected/result/unknown, and measured outcome reference. A publish return must not be labeled executed. |

**Proposed:** a native `Arm`/`ControllablePart` with its own named offset action can
reuse registry and composition without inheriting a false absolute-pose contract.
A dedicated attach-only environment/action interpretation would expose that kind
explicitly. `BaseArm` can only be reused with an honest measured subset and state
contract. `BaseGripper` can only be claimed after width and force semantics are
qualified; otherwise keep an explicitly normalized end-effector contract.
`FrankaRobot`/`PartGroup`, deferred `Connection`, placement metadata, and the
Milestone 2A process boundary remain useful.

**Unverified / blocked:** a transparent Cartesian-policy integration. There is no
arbitrary pose-error-to-Twist conversion in this audit. Choosing a different
absolute-pose controller, switching controllers, or designing a closed-loop pose
tracker needs a separate approved design and physical qualification.

## 6. Ownership and Lifecycle Proposal

Treat action authority as a separate state from an open ROS connection. The table
below is **Proposed**; referenced controller behavior remains source-supported,
not a qualified stop procedure.

| Event | Required Future Adapter Behavior |
|---|---|
| Connect | Attach observation/status and IPC only; start disarmed. Never activate, restart, recover, switch, change gains, send neutral commands, or command an opening/closing/reset as a connection side effect. |
| Verify readiness | Require consistent manager/hardware state, expected effort/position ownership, pinned controller identity, fresh measured state plus typed faults, known calibrated frames/units, selected endpoint/type/QoS, and approved timing/limit profile. Any conflict/timeout keeps the adapter disarmed. |
| Enable | Require explicit operator authorization and a named command owner. Verify a documented handover from all other arm publishers/gripper clients, including the current bridge. Zero publishers at a snapshot is insufficient; a local file lock does not exclude remote writers. |
| Send action | Validate discriminator, dimensions, finite values, bounds, authority/session/sequence, timestamp/deadline, observation dependency and safety profile at admission and again immediately before wire submission. Retain one latest unexpired arm request; bound every queue. |
| Reject invalid/stale | Reject with a recorded reason, never clamp silently or refresh the timestamp of an old action. Clear obsolete pending requests. Loss of admission conditions disarms; it does not invent a stop command or guarantee that prior actions cease physically. |
| Stop publication | Stop scheduling and drop pending arm requests. Report software cessation; account for in-flight messages and controller timeout/hold. The audit does not authorize publishing even a neutral Twist. Any later neutral/stop protocol must be separately specified and tested. |
| Reset | A task bookkeeping reset only while disarmed, with no motion/recovery/activation. Any physical repositioning or gripper release is a separately authorized operation, never inherited from `FrankaEnv.reset()` or `BaseGripper.reset()`. |
| Disconnect | Disarm, clear pending commands, stop/reap only owned bridge processes, preserve the external controllers. Do not automatically open the gripper or switch controllers. An outstanding admitted gripper goal remains an explicit unresolved/monitored resource; disconnect alone cannot promise its cancellation. |
| Controller fault | Disarm and surface fault/status and last command/goal. Require operator review and explicit re-enable after independently verified recovery; no automatic recovery or replay. |
| Communication loss | Cease new submissions, expire local commands, mark acceptance/outstanding goals unknown, preserve diagnostics and alert the operator. A later reconnect starts disarmed with a new authority epoch, never replays buffered commands. |
| Unexpected RLinf/IPC worker exit | The local publisher must stop and pending IPC must be discarded. A future publisher worker must enforce a monotonic **source-action lease** independent of parent liveness; an orphan must not keep refreshing an old request. Legacy NUC timeout remains a secondary reference gate, not a safety-rated stop. |
| NUC/controller process exit | Stop submitting and require operator intervention. Host software cannot guarantee hardware behavior, a protective stop, or end-effector release after this failure. |

**Verified runtime:** ros2_control resource claims identify the active hardware
writer. **Source-supported:** neither the Twist callback nor raw/normalized gripper
actions enforce exclusive caller authority. Multiple Twist writers can alternate
latest arrivals; a competing gripper goal can preempt the current one. **Proposed:**
maintain an enforceable ownership/handover mechanism at the command owner, monitor
unexpected graph changes, and fail closed. Discovery is an advisory check, not an
authentication, atomic lease, or distributed exclusion protocol.

### Four Different Stop Claims

Keep these meanings separate in logs, APIs, and operator procedures:

| Claim | Meaning and Audit Status |
|---|---|
| Software command cessation | **Proposed:** RLinf stops submitting new requests. This says nothing about in-flight packets, admitted goals, or physical standstill. |
| Controller reference hold | **Source-supported:** FR3 stops advancing `q_desired` on a failed freshness/motion gate but continues impedance torque. Motion can continue toward the held reference; the gripper may retain its admitted target. |
| Protective stop | **Unverified / blocked:** a configured robot safety function and its actual triggering/stopping behavior. No robot safety configuration or stop response was qualified here. |
| Hardware emergency stop | **Unverified / blocked:** the installed emergency circuit, access, wiring, stop category, affected devices, and procedure. It is independent of ROS command cessation; FR3 stopping must not be assumed to stop, release, or depower the externally controlled Robotiq. |

Official FR3 documentation distinguishes controlled/braked/monitored stopping
functions; this does not identify which functions or safety rules are installed
and qualified on this cell. See the
[FR3 hardware manual, stopping functions](https://franka.de/hubfs/Hardware%20Manual%20Franka%20Research%203_Arm_v1_R02210_1.3_EN.pdf?hsLang=en).
The [official FCI overview](https://frankarobotics.github.io/docs/doc/libfranka/docs/overview.html)
also distinguishes controller/motion-generator interfaces and warns that its
behavioral monitoring is not a safety guarantee. These references do not replace
a cell-specific risk assessment or authorize altering safeguards.

## 7. Safety and Timing Requirements

Allocate safeguards to the layer that can actually enforce them. Source checks
on the NUC and checks on an RLinf host protect different failure paths.

| Requirement | Existing NUC Evidence | Future RLinf Responsibility / Blocker |
|---|---|---|
| Arm numeric input | **Source-supported:** all six finite and within `[-1,1]`, else non-moving intent | **Proposed:** reject exact-shape/type/range failures before submission, explain reason, and enforce a qualified stricter operating envelope. |
| Arm magnitude | **Source-supported:** fixed per-axis 7 mm / 0.03 rad candidate offsets; no Cartesian combined-norm cap | **Unverified / blocked:** approved per-axis/norm limits and actual response; no arbitrary safe numeric limit selected here. |
| Workspace/tool collision | **Source-supported:** callback/IK path has no task workspace/collision admission | **Proposed:** calibrated control point/frames, current FK and workspace/joint margins, approved constraint check. **Unverified / blocked:** cell envelope, payload, collision thresholds and physical validation. |
| Joint/dynamic constraints | **Source-supported:** URDF-bounded IK and per-cycle joint-reference velocity clamp | **Unverified / blocked:** actual acceleration/jerk/torque/force/energy limits and position-dependent restrictions. Software clipping does not replace hardware safeguards. |
| Command freshness | **Source-supported:** 0.25 s NUC-local receipt/state/target gates; no sender age/sequence | **Proposed:** generation deadline and source lease checked at each local boundary. **Unverified / blocked:** end-to-end stale-message protection after DDS transit under the legacy Twist contract. |
| Observation freshness | **Verified runtime:** standard state arrives; Milestone 2A preserves source and receipt timestamps | **Proposed:** qualified receipt/source-age and cross-stream-skew limits, fresh fault status, clock-health gating. Milestone 2A's default 0.25 s is an inspection setting, not a qualified control limit. |
| Timing uncertainty | **Verified runtime:** both chrony snapshots show normal leap status/stratum 3; workstation RMS offset about 0.230 ms, NUC about 0.196 ms | **Unverified / blocked:** maximum host-to-host offset, scheduler/DDS/IPC jitter, source age and load-dependent delay. Chrony RMS is not a worst-case error bound; no acceptable maximum was invented. |
| Arm QoS | **Verified runtime:** best-effort/volatile subscription; **Source-supported:** depth 1 | **Proposed:** explicitly configured compatible best-effort/volatile keep-last-1 writer, finite delivery/lifespan policy where supported, no latched command or growing resend queue. This choice needs future qualification; no writer was created. |
| Gripper QoS/results | **Verified runtime:** standard action endpoints | **Proposed:** compatible standard action QoS, UUID tracking, bounded acceptance/result waits, explicit unknown-state handling; endpoint presence is not an acknowledgement. |
| Gripper frequency | **Source-supported:** current bridge polls at 10 Hz, manager 500 Hz, status monitor 100 Hz | **Proposed:** send on a qualified requested target change, at most one owned outstanding goal and a bounded expiring pending request. A safe goal-rate limit remains **Unverified / blocked**; no 500 Hz or per-step goal stream is justified. |
| Gripper force/faults | **Source-supported:** position path does not enforce action effort; object verdict timeout is not fault clearance | **Unverified / blocked:** force-path correction/qualification, contact interpretation, calibration, explicit typed-fault readiness, and independent mechanical-stop behavior. |
| Ownership | **Verified runtime:** no arm/input publishers at snapshot; existing gripper bridge/action client present | **Proposed:** documented exclusive handover/authority and operator authorization; never infer permission from an empty graph. |
| Failure/emergency | **Source-supported:** reference gating and error returns; gripper retains commands | **Unverified / blocked:** physical stop/hold, communication-loss behavior, emergency stop coverage and operator response. No automatic recovery, restart, switching, activation, or release. |

**Proposed timing acceptance condition:** select a qualified publish interval and
budget such that worst-case inter-arrival delay at the NUC, including IPC, network,
and scheduling variation, remains below 0.25 s by an explicitly approved margin.
Separately bound source-action age and observation age/skew; satisfying the
inter-arrival condition alone could just repeatedly refresh an obsolete action.
The numeric margin and all component bounds remain **Unverified / blocked**.
NUC steady-clock watchdogs avoid wall-clock corrections but do not timestamp the
sender. Cross-host source-age comparisons require a qualified clock-uncertainty
bound; otherwise remain disarmed.

**Source-supported:** the
[official ROS QoS reference](https://raw.githubusercontent.com/ros2/ros2_documentation/humble/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst)
distinguishes compatibility, retry, lifespan, and liveliness. A reliable action
request can arrive late, and DDS liveliness is not an operator deadman. Mixed
Humble/Jazzy state reception does not demonstrate action delivery, bounded command
age, or trustworthy command-owner discovery under failures.

## 8. Remaining Blockers and Required Physical Qualification

The audit resolves software meaning while leaving physical claims disabled. Each
item below is **Unverified / blocked** for live action execution, not unfinished
work for this read-only milestone.

| Blocker | Evidence Required Before Enabling Physical Actions |
|---|---|
| Task/controller semantic mismatch | An approved normalized-offset task contract, or a separately designed compatible absolute-pose interface; no arbitrary pose-to-Twist conversion. |
| Frames and control point | Verified relationship between chain `base`/`fr3_link8`, measured `O_T_EE`, task TCP, gripper mount, and any `base_link` frame; qualified physical axis/orientation direction. |
| Exclusive command authority | Named operator-approved owner and handover, including current bridge and external clients; enforceable prevention/detection of competing writers and preemption. |
| Source/binary assurance | Required deployment attestation level, pinned identities/parameters/URDF, and change-detection policy. Current evidence supports attribution, not a rebuilt proof. |
| Timing and stale deliveries | Bounded source/receipt age, skew, jitter, clock uncertainty, stale in-flight request handling, and source-action lease under parent/worker/network loss. |
| Readiness/fault telemetry | Stable consistent manager/hardware reports, typed fault state and freshness, command/goal status distinctions. Resolve status-query inconsistency and unreliable discovery before using them as action gates. |
| Arm operating envelope | Qualified workspace, joint margins, speed/dynamics, tool/payload and collision/safety settings; actual hold/stop response under an approved test procedure. |
| Gripper calibration | Physical width/profile endpoints, direction, contact/closure interpretation, repeatability and tolerances, and valid raw/normalized operating range. |
| Gripper effort and persistence | Establish/fix whether requested force is actually applied, qualify force/speed, terminal-goal persistence, no-result handling, fault/loss behavior, and stop/release policy. |
| Emergency procedure | Cell-specific emergency/protective-stop coverage for both arm and independently powered gripper, operator access and responsibility; no software claim substitutes for it. |
| Action provenance in datasets | Distinct requested, validated, submitted, accepted/unknown, and measured outcomes, with timestamps, units, calibration/controller identity and rejection reasons. |

**Proposed separately authorized physical qualification:** a human-approved plan
must name the operator, safety setup, bounded command/test envelope, command owner,
stop procedure, and acceptance criteria before any trial. It would need to test
axes/control point, neutral and cessation response, watchdog timing/hold, process
and communication loss, goal admission/preemption/terminal semantics, gripper
force and width, and fault handling. Even publishing an all-zero Twist, submitting
or canceling a gripper goal, or testing an emergency response is outside this
milestone's authorization. No such test was attempted.

## 9. Minimal Milestone 2B.2 Scope

Start with offline and disarmed behavior, leaving live enable dependent on the
blockers above. Everything in this section is **Proposed**; no item was implemented
in this milestone.

1. Add a versioned typed action envelope and pure validation/serialization for
   the normalized measured-FK-offset and gripper-closure representations. Include
   units, mixed frames, identity, monotonic expiry/source lease, observation
   dependency, authority, and per-stage dispositions. Reject pose/velocity/width
   inputs that lack an approved conversion contract.
2. Reuse native `Connection`/`ControllablePart` composition and Milestone 2A's
   compatible ROS process boundary. Keep the existing observation-only classes
   read-only. Any action-capable worker must have an explicit disarmed state,
   bounded IPC/latest-only storage, independent source expiry, startup rollback,
   and parent-loss cleanup; idle construction/connection must not create a motion
   writer or send a goal.
3. Add read-only readiness/fault aggregation and a fake-ROS action edge for tests.
   Cover invalid/stale/replayed actions, inconsistent status, competing ownership,
   source/worker death, late/unknown acknowledgements, goal rejection/preemption,
   stale queues, and independent arm/gripper partial failures without hardware.
4. Define an attach-only environment/action-kind path that cannot call existing
   motion reset, machine-wide ROS setup, recovery, activation, or controller
   switching. Add recording metadata distinguishing submission from execution;
   use the current trajectory storage rather than inventing a second recorder.
5. Gate the real publisher/action-client implementation and all physical enable
   behind a reviewed ownership, timing, calibration, force, and emergency plan.
   If gripper effort requires a NUC controller correction or arm semantics require
   a different server, track that as a separately approved deployment change;
   RLinf must not silently reconfigure the controller to make its API fit.

The smallest justified next deliverable is therefore a tested, disarmed adapter
contract and attach lifecycle. It is not immediate policy inference, data
collection, or a proof that physical actions are safe.

## 10. Validation and Change Boundary

Validation checks the report against the inspected contracts and confirms the
repository change boundary. It does not exercise commands or demonstrate a
physical safety property.

**Verified runtime:** passive NUC inspection used SSH filesystem/process metadata,
read-only `list_controllers`, `list_hardware_components`,
`list_hardware_interfaces`, `list_parameters`, and `get_parameters`; ROS topic,
node, and action discovery; two-second standard JointState subscriptions per
domain; a three-second gripper typed-status subscription; and chrony status on
both hosts. All probe executors/nodes/contexts were closed. The first probe needed
a Humble message-field adjustment; short discovery timeouts and an initially
inconsistent arm hardware reply were preserved as limitations rather than hidden.

**Verified runtime:** no command publisher, Twist message (including zero), action
client/goal/cancellation, serial/FCI connection, controller lifecycle change,
parameter/gain/limit write, recovery, hardware permission change, camera start,
policy inference, or real-world data collection was introduced by this audit.
Existing controllers, their bridge, and device owners were left running.

Only this standalone audit report is intended for the commit; runtime sources,
tests, and bilingual Sphinx guides remain unchanged. This requested Markdown audit
is outside the Sphinx language trees; EN/ZH parity applies to the existing guides,
which this report does not replace.

| Validation | Result |
|---|---|
| Native action/observation/lifecycle and recording trace | **Source-supported:** inspected the referenced fresh-branch definitions and representative callers without instantiating a hardware environment. |
| Deployed identity and passive ROS checks | **Verified runtime:** fingerprints, source/build evidence, endpoint types/QoS, parameter/status queries, state/status subscriptions, and repeat readiness query completed as described above. |
| `build_docs.py --venv /tmp/fr3-m2a-docs-venv` | Both EN and ZH: zero warnings with the repository harness's default autodoc-import filtering. This is a regression gate for the existing trees; the standalone Markdown report is not a Sphinx page. |
| `check_rst_markup.py` on existing EN/ZH FR3 guides | No inline-markup problems. |
| `check_doc_symbols.py` on existing EN/ZH FR3 guides | Passed with the independently verified runtime names `fr3_joint1`, `fr3_joint7`, and `fr3_link2` allowed. |
| Report structure, classifications, native-file links and Markdown rendering | Checked against this report's requested scope and the inspected evidence. |
| Whitespace and final Git diff | `git diff --check` and staged-diff review; report-only change required. |
| Runtime tests, inference, collection and physical safety tests | Not run: runtime source is unchanged and hardware-control tests are prohibited for this milestone. |

No doc-code or EN-ZH consistency issues found in checked scope. The commit uses
the configured human Git identity and `Signed-off-by` trailer, with subject
`docs(fr3): audit ROS2 action and safety contracts`. The resulting commit SHA and
push outcome are supplied in the completion message; the report does not embed
its own commit hash.
