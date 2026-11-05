# FR3 Controller Startup Failure — Diagnosis and Offline Fix

This report records the source fixes and CPU-only verification performed on the NUC on **2026-10-26 JST**. It separates the confirmed safety-abort path from the unresolved physical cause, lists the exact modified files, and defines a later operator-supervised validation gate.

**Result:** the software crash path and identified controller defects have offline fixes; **the physical startup failure is not declared resolved**. Libfranka reported an emergency/User Stop abort. Franka Desk subsequently reported a **Generic joint error** requiring self-tests that power-cycle the arm and end effector. Those self-tests, fault acknowledgement and all physical validation remain unexecuted.

## Confirmed Findings and Remaining Uncertainty

The matching NUC libfranka source is a clean `0.18.0` checkout. The inspected call path is:

```text
FrankaHardwareInterface::read()
  Robot::readOnce()                 # control mutex held
    Robot::readOnceActiveControl()  # an active-control object exists
      franka::ActiveControl::readOnce()
        Robot::Impl::readOnce()
        Robot::Impl::throwOnMotionError(state, motion_id)
          receive the robot's TCP Move response
          handleCommandResponse<Move>()
          wrap CommandException as ControlException
```

In `libfranka/src/robot_impl.h`, `Move::Status::kEmergencyAborted` maps to **“Move command aborted: User Stop pressed!”**. `kReflexAborted` and `kInputErrorAborted` have different messages. The exact text therefore identifies a robot-reported abort category; it is not an IK exception invented by the ROS controller. It does not identify which physical device or safety input caused the stop.

| Finding | Status | Evidence and limit |
| --- | --- | --- |
| Robot-reported emergency/User Stop abort | CONFIRMED | Matching libfranka status/message mapping and supplied active-control backtrace. The existing saved node log repeats this message. |
| Unhandled hardware-read exception | CONFIRMED, FIXED OFFLINE | Original `read()` directly called `robot_->readOnce()` without a boundary catch. The exception escaped the ROS control loop and the process aborted. The fix returns hardware `ERROR` and latches further I/O. |
| Source of the physical stop signal | UNRESOLVED | No safety-event export, enable-device trace or operator timing identifies the originating input. The message alone does not prove that a particular button was physically pressed. |
| Desk joint fault | CONFIRMED FROM USER REPORT; CAUSE UNRESOLVED | Exact report: “One or multiple joint errors were recognized … Generic joint error.” Desk requests self-tests with arm/end-effector power cycling. No joint IDs or detailed diagnostic codes were available. |
| Controller torque or FCI-session defect caused that stop | NOT ESTABLISHED | Effort ownership selects one torque-control session. The read failure detects its abort; a failed read is not evidence that the read call caused the stop. |
| Deadline/communication failure caused this abort | NOT ESTABLISHED | The quoted failure is not a `communication_constraints_violation` reflex. Source and build inspection found timing risks, but no measured deadline trace establishes causation. |
| Completed gripper/TCP/seed fixes | PRESERVED | `load_gripper: "false"`, the `fr3_link8` chain endpoint, measured IK seeding, neutral input initialization and non-throwing IK failure handling were retained. No Franka Hand substitution was introduced. |

Saved evidence includes `/home/aist-franka/.ros/log/ros2_control_node_681743_1793009192044.log` and its `2026-10-26-19-06-31-864272-aistfranka-NUC13ANK-B-681740/launch.log`. The latter records process exit `-6`. The node log records successful FIFO scheduling at priority 50, a memory-lock privilege warning, a 1000 Hz update rate and the User Stop message. The inspected normal controller build cache was Debug with coverage/`-O0` options. These are timing qualifications, not proof of the reported stop's cause.

One older saved log contained joint velocity/acceleration discontinuity reflex errors. It was not the supplied latest failure and is not used to explain the current Generic joint error.

The upstream RLinf open-PR/issue search for `franka ControlException` returned no matches. Implementation decisions follow the NUC sources rather than unrelated issue reports. The ROS error-return contract was also checked against [ros2_control 2.54.0 `System::read/write`](https://github.com/ros-controls/ros2_control/blob/2.54.0/hardware_interface/src/system.cpp), matching the installed version.

## Implementation Changes

Hardware error handling now preserves the libfranka reason and blocks continuation. `read()`, `write()` and command-mode transition failures catch `franka::Exception`, log the original error, latch the failure and return `hardware_interface::return_type::ERROR`. Activation checks the initial read result. No code clears this latch; the hardware component must be recreated after operator review. This is a software failure boundary, not robot fault recovery.

All five low-level control-interface initializers now propagate failed starts. Their previous catch/recover/retry sequences called `automaticErrorRecovery()` implicitly. Those automatic sequences were removed; the explicit recovery API remains available to its existing callers and was not invoked during this work. Successful mode selection, effort ownership, filtering and rate limiting remain unchanged.

The controller now separates IK from torque updates:

- KDL solvers are allocated during configuration. IK runs in a non-real-time ROS wall-timer callback, default **50 Hz**, rather than constructing solvers and running up to 100 iterations on every 1 kHz update.
- A fixed-size measured-state sample crosses to the IK callback under a mutex. The RT writer uses `try_lock()` and skips a sample instead of waiting. Callback input and solved targets cross through `realtime_tools::RealtimeBuffer`; RT readers do not perform blocking writes or allocate IK objects.
- Non-RT cleanup/reconfiguration cancels the timer and serializes solver replacement with an in-flight IK callback. Parameter-service waits happen before taking that solver lock. The RT update never takes it.
- Activation resolves joint interfaces by their complete names, seeds the held reference from measured joints, initializes the velocity filter from measured velocity and validates finite state/limits. It no longer relies on offsets `16/23/30` into loaned interfaces.
- Forward kinematics and IK use the same `base` → `fr3_link8` chain. The target no longer assumes that the robot's separately configured `O_T_EE` is the URDF flange pose. `fr3_link8` remains a flange reference, not a calibrated Robotiq TCP.
- Normalized SpaceMouse inputs must be finite and within `[-1,1]`. Neutral, invalid, pre-activation or expired input cannot advance the reference. The default **250 ms** timeout uses the NUC steady clock. Solved targets retain the originating input/state timestamps, so computation does not refresh an old command.
- An accepted target advances by no more than the corresponding URDF joint velocity times the smaller of the measured period and **1 ms**. Invalid IK results invalidate the target; the controller holds its last accepted reference. This bounds position-reference steps; it is not a certified acceleration-limited trajectory generator or stop function.
- Configuration preserves operator collision thresholds. The old automatic `set_full_collision_behavior` request was removed. No replacement thresholds or gains were introduced.

The existing `0.007` linear and `0.03` angular normalized pose-offset scales remain. Moving the solver to 50 Hz and bounding reference steps changes transient tracking behavior and requires later supervised qualification. The topic name does not make these offsets an SI velocity contract. Gain arrays and the existing torque-rate limiter were preserved byte-for-byte in their configuration/header sources.

New controller parameters are `command_timeout` (seconds, default `0.25`) and `ik_update_rate` (Hz, default `50.0`, accepted range `(0,100]`). The current configuration remains a 1000 Hz torque loop. No experiment config or launch file was changed.

## Exact Files Modified

All eleven source/test changes below were applied **on the NUC**. Existing user edits were snapshotted before modification and checked by hash before writing. Source originals are also saved under `/tmp/fr3_startup_offline_originals/` on the NUC. No remote Git commit, reset or checkout was performed.

| NUC file | Change |
| --- | --- |
| `/home/aist-franka/Desktop/aist-gamma/franka_ros2_ws/src/franka_ros2/franka_hardware/include/franka_hardware/franka_hardware_interface.hpp` | Atomic latched I/O failure state. |
| `/home/aist-franka/Desktop/aist-gamma/franka_ros2_ws/src/franka_ros2/franka_hardware/src/franka_hardware_interface.cpp` | Error boundaries for reads/writes/mode transitions; activation honors initial-read failure. |
| `/home/aist-franka/Desktop/aist-gamma/franka_ros2_ws/src/franka_ros2/franka_hardware/src/robot.cpp` | Remove implicit recovery/retry from all five interface initializers. Existing position/torque filtering and limiting preserved. |
| `/home/aist-franka/Desktop/aist-gamma/franka_ros2_ws/src/franka_ros2/franka_hardware/test/franka_hardware_interface_test.cpp` | Four injected-failure tests for latching, initial read, mode start and write errors. |
| `/home/aist-franka/Desktop/aist-gamma/franka_ros2_ws/src/franka_ros2/franka_hardware/test/franka_robot_test.cpp` | Five failed-start propagation tests; stationary position-reference fixture consistent with existing filtering. |
| `/home/aist-franka/Desktop/aist-gamma/franka_ros2_ws/src/franka_ros2/franka_hardware/test/franka_robot_test.hpp` | Explicit initialized mock robot state for read/write tests. |
| `/home/aist-franka/Desktop/aist-gamma/franka_controller_ws/src/franka_arm_controllers/include/franka_arm_controllers/joint_impedance_ik_controller.hpp` | Fixed-size exchange records, reusable solver ownership, named interface indices and lifecycle/input state. |
| `/home/aist-franka/Desktop/aist-gamma/franka_controller_ws/src/franka_arm_controllers/src/joint_impedance_ik_controller.cpp` | Non-RT IK, measured initialization, command expiry, bounded targets, finite checks and preservation of collision settings. |
| `/home/aist-franka/Desktop/aist-gamma/franka_controller_ws/src/franka_arm_controllers/test/joint_impedance_ik_controller_test.cpp` | Six tests using the real controller with an injected model and in-memory interfaces. |
| `/home/aist-franka/Desktop/aist-gamma/franka_controller_ws/src/franka_arm_controllers/CMakeLists.txt` | Add the focused test target, generate a fake-hardware URDF and isolate the test ROS domain. |
| `/home/aist-franka/Desktop/aist-gamma/franka_controller_ws/src/franka_arm_controllers/package.xml` | Declare test dependencies for GTest, description and xacro. |

The Workstation repository contains this report plus two incremental patches, [controller](franka_arm_controllers.patch) and [hardware](franka_hardware.patch), a [source hash manifest](source_manifest.json), and [offline test summaries](offline_test_results.json). Patches are relative to the respective NUC repository roots and capture only this work against the pre-existing working-tree snapshots. They are review/backup artifacts; do not apply them again to the already-modified NUC sources.

## Build and Focused Test Results

Both affected packages built successfully on the **NUC CPU, Ubuntu 22.04 / ROS Humble**, using installed dependencies and `CMAKE_BUILD_TYPE=Release`. Build/install outputs were isolated under:

```text
/tmp/fr3_startup_offline/hardware/{build,install,log}
/tmp/fr3_startup_offline/controller/{build,install,log}
```

The normal workspace `install/` binaries were not replaced. These isolated builds are **not an authorized physical deployment**. Colcon reported the expected underlay-override warning. Existing libfranka deprecation and old test strict-aliasing warnings remain; compiler diagnostics were not suppressed.

Builds selected only `franka_hardware` and `franka_arm_controllers`. A separate pre-change snapshot built only the robot-test target to investigate one regression result. The known Pinocchio dependency path was supplied as a temporary `LD_LIBRARY_PATH=/opt/openrobots/lib:...`; no loader/system configuration changed.

| Final test group | Result | Scope |
| --- | --- | --- |
| Hardware I/O/lifecycle/error regression selection | **PASSED: 16/16**, 1.748 s | Four new failure tests plus relevant existing read/write/activation/mode checks. Injected `MockRobot`; no network robot constructor. |
| Mock libfranka robot sessions | **PASSED: 19/19** | Existing mode/read/write tests and five new start-failure tests. `MockFrankaRobot`/`MockActiveControl`; no FCI session. |
| Controller fake-interface tests | **PASSED: 6/6**, 0.717 s | Measured initialization; shuffled name mapping; missing/non-finite state rejection; bounded targets and expiry; cleanup/reconfiguration; invalid input; pre-activation input rejection. |
| Generated URDF and config inspection | **PASSED** | Seven FR3 joints with effort/position/velocity command and state interfaces; `fr3_link8`; no Hand joints; Humble `fake_components/GenericSystem`; unchanged gains and `load_gripper: "false"`. The XML was parsed, not loaded as a physical plugin. |

**Total final focused tests: 41 passed, 0 failed.** Test ROS traffic used `ROS_DOMAIN_ID=213`, `ROS_LOCALHOST_ONLY=1` and a controller-test namespace `/offline_fr3_test`. The controller tests inject a constant model and scalar state/command handles; their test-local input publisher cannot reach the domain-70 physical controller.

The generic fake plugin does not provide the opaque real Franka model/state pointers required by this controller. Accordingly, no full fake launch is claimed: the actual controller is tested with explicit in-memory semantic interfaces, and the generic fake URDF is inspected separately.

The relevant build/test commands, after sourcing Humble and the existing workspace underlay, were:

```bash
# Hardware package; use the corresponding workspace as the working directory.
colcon --log-base /tmp/fr3_startup_offline/hardware/log build \
  --packages-select franka_hardware \
  --build-base /tmp/fr3_startup_offline/hardware/build \
  --install-base /tmp/fr3_startup_offline/hardware/install \
  --parallel-workers 2 \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON

# Controller package; source the isolated hardware overlay for its matching headers/libraries.
colcon --log-base /tmp/fr3_startup_offline/controller/log build \
  --packages-select franka_arm_controllers \
  --build-base /tmp/fr3_startup_offline/controller/build \
  --install-base /tmp/fr3_startup_offline/controller/install \
  --parallel-workers 2 \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
```

These commands compile libraries and tests; they do not launch a controller manager or initialize robot hardware. Final test invocations and case names are captured in the linked JSON summaries; NUC XML/log files are `/tmp/fr3_startup_offline/{hardware_io_final,robot_sessions_final,controller_fake_interfaces_final}.{xml,log}`. The controller test also passed through its targeted CTest entry before the added pre-activation case.

Development failures were retained in the diagnosis:

- The initial controller test executable failed to link because the temporary Pinocchio library path was omitted. Adding the existing library path to the build process resolved it.
- The expanded robot-session suite initially passed 18/19. Its unchanged position test also failed **against the original source snapshot**: it expected an abrupt raw target despite the user's already-enabled position filtering/limiting, and supplied no explicit mock state. The test now supplies initialized state and checks a stationary target; production position limiting was not changed. The final suite passed 19/19.
- The first URDF inspection expected the newer `mock_components` namespace. Humble generated `fake_components/GenericSystem`; correcting that inspection expectation resolved the assertion without changing the generated model.

No unrelated package, GPU/training suite, hardware experiment, robot restart or gripper test was executed.

Final source hashes matched all eleven files in the manifest. Both incremental patches passed `git apply --check` against the saved pre-change snapshots. Remote `git diff --check`, report links and JSON validation also passed. The normal installed binaries were left untouched; only the isolated temporary controller install was refreshed after the final build.

## Remaining Risks

The **Generic joint error and requested power-cycling self-tests remain the primary blocker**. Software fixes do not acknowledge, clear or explain that fault. The physical source of the emergency/User Stop response remains unknown. An authorized AIST operator must review the exact Desk diagnostic history and use the approved Franka procedure/support process before any physical validation.

Offline tests establish software behavior, not FCI timing, torque tracking, physical joint health or a collision-safe trajectory. The observed memory-lock warning and actual 1 kHz cycle performance still need qualification. Reference velocity bounds and the unchanged torque-rate limiter do not prove acceleration/jerk compliance or certify an emergency stop. Expired input holds the last reference while impedance control remains active; that is not a deadman or safety-rated stop.

The input message is unstamped. Its timeout measures local callback receipt, not the age of the original HID event. Payload/TCP/inertia calibration for the mounted Robotiq remains unqualified. Current names/frame checks target the supplied FR3 configuration; other prefixes or namespaces need their own configuration tests. Physical validation must use one reviewed, consistent build/overlay and one FCI owner; no mixture of isolated test libraries and normal installed binaries is approved here.

## Operator-Supervised Validation Plan — Not Executed

1. Keep the system out of active testing until an authorized operator reviews the Generic joint error, requested self-tests/power cycling, retained objects and exact Desk diagnostics with the responsible AIST safety lead or Franka support. Obtain explicit authorization for subsequent hardware validation. This report does not authorize self-tests or fault acknowledgement.
2. Review the patches, Robotiq payload/TCP and current safety settings, then select/build a consistent approved runtime. Preserve the stop/enable protections, collision thresholds and one existing FCI owner. Keep SpaceMouse/policy command sources disconnected for the first check.
3. After fault review and explicit authorization, perform **one supervised stationary activation** under the approved procedure. Record Desk events, robot mode/last-motion errors, control-command success rate and control-loop timing. Stop under the approved procedure on any unexpected event; do not run repeated automatic relaunches or recovery loops.
4. Review that evidence before authorizing any separate command, expiry/deadman or motion test. No such test, physical launch, active-control connection, reset, recovery or safety-setting change was performed in this work.

Prepared with an AI assistant. The NUC source changes and isolated offline builds are available for review; physical startup remains unvalidated.
