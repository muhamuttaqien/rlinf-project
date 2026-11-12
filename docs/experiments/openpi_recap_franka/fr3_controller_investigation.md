# FR3 Controller Investigation

Use this report to reproduce the FR3 dependency failure and prepare a supervised
command-to-motion test. The 2026-11-12 investigation uses the NUC's actual source,
installed binaries, and saved launch log. It covers runtime loading first, then
command execution. Physical controller startup and movement remain untested in
this investigation. An AI assistant performed the investigation and changes.

## Original Problem

The supplied physical result was eleven publications of `[0.03, 0, 0, 0, 0, 0]`
over 0.5 seconds, with TCP displacement `[0.030, -0.003, -0.031]` mm and maximum
travel 0.045 mm. Positive X movement was not verified. The subsequent diagnostic
launch failed while loading `libfranka_hardware.so`, before controller acceptance
could be measured. The existing `FR3_DIAG_V1` instrumentation compiled separately.

The objective is to qualify the controller before deploying the 3,000-step π₀.₅
policy and collecting 20 real outcomes. Existing training, inference, dataset,
and RECAP implementations are preserved.

## Startup Root Cause And Evidence

The missing dependency is installed; its directory is absent from a clean launch's
library search path. Both ROS workspace overlays alone reproduce the failure.

| Check on NUC | Observed result |
| --- | --- |
| Saved `~/fr3_diag_launch.log` | `pluginlib::LibraryLoadException`: `libpinocchio_parsers.so.4.1.0` not found while loading the hardware plugin; exit -6. |
| `dpkg-query -W robotpkg-pinocchio` | Version `4.1.0`; package owns `/opt/openrobots/lib/libpinocchio_parsers.so.4.1.0`. |
| Exact parser/default libraries | Both present under `/opt/openrobots/lib`; parser SONAME is `libpinocchio_parsers.so.4.1.0`. |
| ROS Pinocchio | Separate 4.0.0 libraries under `/opt/ros/humble/lib/x86_64-linux-gnu`; these do not satisfy the required 4.1.0 SONAMEs. |
| Hardware ELF metadata | `NEEDED libfranka.so.0.18`; `RUNPATH` includes ROS, workspace messages and libfranka directories. |
| Libfranka ELF metadata | Requires parser/default 4.1.0; no RPATH/RUNPATH supplying `/opt/openrobots/lib`. |
| Loader cache | `ldconfig -p` has no Pinocchio entries. |
| Clean Humble + hardware/controller overlays | `ldd -r` reports both 4.1.0 libraries missing and unresolved Pinocchio symbols. |
| Same environment plus `/opt/openrobots/lib` | `ldd -r` resolves both exact libraries, with no missing libraries or unresolved symbols. |
| `ctypes.CDLL(..., mode=os.RTLD_NOW)` with fixed environment | Hardware plugin and isolated diagnostic controller library both load successfully; no controller/hardware instances created. |

This establishes a runtime path defect, rather than evidence of an ABI mismatch
or missing installation. The hardware plugin's RUNPATH does not supply its
libfranka dependency's transitive dependencies. No dependency installation,
library symlink, ELF edit, or system loader configuration change was needed.

The existing NUC process inspected during this session is a **Robotiq gripper**
`ros2_control_node` on domain 71, with `/opt/openrobots/lib` in `LD_LIBRARY_PATH`.
It is not a running FR3 arm controller. The original successful arm process has
exited, so its environment cannot be recovered from `/proc`; the gripper's
environment and earlier offline build report show that the required path was
used elsewhere, but do not prove the old arm launch's complete environment.

## Reproduce And Apply The Shell Fix

Reproduce without starting ROS nodes or contacting the robot. On the NUC, open a
clean shell, then source the standard overlays and inspect the trusted local ELF:

```bash
env -i HOME="$HOME" PATH=/usr/bin:/bin bash --noprofile --norc
source /opt/ros/humble/setup.bash
source ~/Desktop/aist-gamma/franka_ros2_ws/install/setup.bash
source ~/Desktop/aist-gamma/franka_controller_ws/install/setup.bash
hardware="$HOME/Desktop/aist-gamma/franka_ros2_ws/install/franka_hardware/lib/libfranka_hardware.so"
ldd -r "$hardware"
```

Expect the two missing 4.1.0 dependencies. The minimal correction in that shell is:

```bash
export LD_LIBRARY_PATH="/opt/openrobots/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd -r "$hardware"
```

Expect no `not found` or `undefined symbol` entries. For repeatable future use,
source the committed helper from a NUC copy of this repository, in a fresh Bash
shell:

```bash
source toolkits/realworld_check/fr3_controller_env.sh \
  "$HOME/fr3_controller_diag_build/install"
```

The helper checks the exact installed SONAME files, sources Humble and both
workspaces, optionally selects the existing diagnostic overlay, prepends the
verified dependency directory, and sets domain 70. It starts no processes.
Omit its argument to select the ordinary controller overlay. The diagnostic
overlay must be selected explicitly; fixing hardware loading alone does not
select the instrumented controller.

After sourcing, verify immediate symbol resolution without constructing a hardware
object:

```bash
python3 - <<'PY'
import ctypes
import os
from pathlib import Path

for relative in (
    "Desktop/aist-gamma/franka_ros2_ws/install/franka_hardware/lib/libfranka_hardware.so",
    "fr3_controller_diag_build/install/franka_arm_controllers/lib/libfranka_arm_controllers.so",
):
    path = Path.home() / relative
    ctypes.CDLL(str(path), mode=os.RTLD_NOW)
    print(f"Loaded {path.name}; no hardware instance created")
PY
```

Both loads must succeed. This checks the selected dependency set without launching
ROS control; it does not qualify robot communication or physical startup.

## Command Path Findings

The NUC source and offline tests establish the controller's request semantics.
They do **not** establish which stage limited the earlier physical pulse. The
ordinary controller source SHA-256 is
`8d6ebf4ce420586dca3fe39a0befbd0d7bd61a4230153a2c7a5637cdd8bbdc47`, matching the
existing [source manifest](fr3_startup_fix/source_manifest.json). Applying the
existing diagnostic generator to that source yields
`0bc5e940c5e15a6b3b251908806de04c30ffaa6cd476be4a7688c75fe4f65c68`, exactly matching
the isolated diagnostic source. No second diagnostic system was introduced.

| Stage | Confirmed implementation | Physical evidence still needed |
| --- | --- | --- |
| ROS delivery | Relative topic `franka_controller/target_cartesian_velocity_percent`, `Twist`, best-effort SensorDataQoS, depth 1; empty arm namespace resolves to the specified absolute topic. Reliable Python publishers can offer to this subscriber. | Domain 70 endpoints, exclusive publisher and increasing `rx` during the pulse. Publication counts alone do not establish delivery; depth 1 can drop intermediate samples. |
| Validation | All six values must be finite and within `[-1,1]`. Any nonzero accepted axis means moving; no controller deadband. Invalid requests become neutral. The pulse tool sends 0.03 directly. | `moving_rx`, `invalid_rx`, `last_x`; cross-version DDS delivery remains unverified by local NUC tests. |
| Timeout/state exchange | RT loop copies measured joints plus latest intent into a fixed sample using `try_lock`. Steady-clock input and state timestamps must be at/after activation and at most 0.25 s old by default. | Actual input gaps, solver delay and hardware-state progress. The sample timestamp marks a local copy, rather than proving a new robot firmware sample. |
| Cartesian target | FK of **measured** joints in the URDF `base` → `fr3_link8` chain, plus `0.007 * R_mountᵀ * linear_request` metres. Rotation is postmultiplied by RPY `0.03 * angular_request` radians. Current mounting RPY is `[0,0,0]`. | Correct physical axis direction and calibrated flange/tool relation. Linear offsets use base axes at zero mounting; rotations are composed in flange axes. |
| IK | Non-RT wall timer defaults to 50 Hz; KDL pseudoinverse velocity solver and joint-limited Newton-Raphson position solver, 100 iterations, epsilon `1e-6`. Starts from sampled measured joints. FK failure skips the attempt counter; negative IK return, nonfinite or out-of-limit joints invalidate the target. | `ik_attempt`, `ik_valid`, `ik_failed` at the actual robot pose. Counters do not identify a numerical failure reason. |
| Reference | Desired joints initialize to measured joints. A valid, fresh target in the same activation advances each joint by at most `URDF velocity_limit * min(period, 0.001)`. Nominal RT rate is 1000 Hz. | `gate_ticks`, `ref_advanced_ticks`; configured frequency is not a measured RT deadline trace. |
| Hold/overwrite | Every completed IK solve replaces the target. Neutral/invalid/expired intent blocks advancement but **holds the last joint reference**, rather than rebasing it to current joints. A changed moving request can briefly use an older still-fresh target until the next solve; timestamps are freshness checks, not exact request IDs. | Joint tracking and residual movement after neutral. Neutral/timeout is not a hardware stop. |
| Torque | `K*(q_desired-q) - D*dq_filtered + coriolis`; `dq_filtered = 0.01*old + 0.99*dq`. Finite torque is written to named joint effort interfaces. K is `[600,600,600,600,250,150,50]`, D is `[30,30,30,30,10,10,5]` in current config. | `effort_write_ticks` counts controller interface writes, including neutral operation; it proves neither torque magnitude nor hardware acceptance. |
| Hardware | Hardware read supplies libfranka `q`, `dq`, `tau_J`, `O_T_EE`; effort mode calls `Robot::writeOnceJointEfforts`, existing torque-rate limiting, then active torque-control `writeOnce`. Existing exception handling latches I/O errors. | Robot desired/measured effort, joint changes, status/faults and TCP response. No physical acceptance is inferred from a controller counter. |

Targets are recomputed as an offset from each sampled measured pose, rather than
integrated from the last Cartesian target. For 0.03, IK asks for a **0.21 mm**
linear offset. Repeated messages refresh intent while a timer recomputes targets;
eleven messages do not imply eleven completed 0.21 mm moves. If the arm follows,
the next measured pose moves the target onward. If it remains stationary, the
same equilibrium is repeatedly requested. Thus physical response determines the
effective speed; the topic name does not define SI velocity.

The new [test-only patch](fr3_startup_fix/controller_motion_tests.patch) extends
the existing controller test file. It runs real ROS subscription, KDL and torque
code against in-memory hardware interfaces, a constant external model, and the
FR3 URDF. At the fixture pose `[0.1,-0.3,0.1,-1.6,0.1,1.2,0.5]` radians:

- Both +0.03 and -0.03 produce torque equilibria whose FK offsets are ±0.21 mm
  along X, within 2 micrometres; Y/Z offsets remain within that tolerance.
- Repeating +0.03 with stationary measured joints leaves output torque unchanged.
- Supplying an ideal measured response at the first equilibrium advances the
  next equilibrium to approximately +0.42 mm from the starting pose.

The fixture derives equilibrium joints from its known K=10, D=0 and constant
Coriolis=0.25 torque law. These are **test fixture** values, not gain changes or a
physical dynamics simulation. The result rules out a universal small-request
deadband at this fixture pose; it cannot rule out IK failure at the real pose.

The existing diagnostics logged the following during the stationary-state test:

```text
FR3_DIAG_V1 rx=195 moving_rx=195 invalid_rx=0 last_x=0.030000
ik_attempt=50 ik_valid=50 ik_failed=0 gate_ticks=191
ref_advanced_ticks=2 effort_write_ticks=195 last_reference_step_rad=0.000000000
```

Reference advancement can cease after reaching an unchanged target while the gate,
IK and effort-write counters continue. A zero latest reference step alone is
therefore not evidence of failure. These counters were recorded on CPU fake
interfaces in localhost domain 213, not during a physical FR3 pulse.

## Policy Request Interpretation

The training labels are time-window-averaged normalized Twist requests, as recorded
by `fr3_request_lerobot_export.py`, not measured motion or accepted targets. The
NUC SpaceMouse publisher already maps HID axes to Twist as
`[-y, x, z, -roll, -pitch, -yaw]`, with optional front/back sign changes, at 100 Hz.
Replay of recorded Twist labels must not apply that HID transform a second time.

`fr3_pi05_request_adapter.py` passes those six request channels through. However,
`fr3_pi05_ros_executor.py` defaults to `legacy_limited`: an extra **0.15** multiplier
at a 10 Hz action interval. For an example decoded 0.03 prediction, that mode
sends 0.0045, asking IK for only 0.0315 mm. `identity_verified` sends the original
request, but remains gated on separate physical verification. This existing
attenuation is confirmed and preserved. It does not explain the direct pulse,
which sends 0.03 without it. Temporal averaging and 10 Hz playback versus 100 Hz
demonstration input also need qualification; matching axis names alone is
insufficient to certify physical policy execution.

## Physical Cause Still Unresolved

The small offset and measured-pose target regeneration explain why this command
cannot be treated as a requested travel distance. They do not establish why the
real FR3 moved only 0.045 mm. No matching physical `FR3_DIAG_V1` trace was found
in the saved arm-node logs inspected during this session; the supplied diagnostic
launch failed before loading the controller. Receipt, valid IK, reference
advancement and hardware execution during the earlier pulse remain unconfirmed.

Possible limiting stages include competing neutral publishers or lost input,
freshness rejection, IK failure at the actual seed pose, insufficient physical
tracking for this small offset, or hardware/session faults. Debug/coverage builds
and solver/executor scheduling may affect timing. Those are hypotheses, not
reasons to change gains, scales, limits, timeouts or safety settings. Frame and
tool calibration also remain necessary: the controller targets URDF flange FK,
whereas the pose broadcaster reports robot `O_T_EE`; neither alone certifies the
Robotiq peg TCP.

## Validation And Remaining Work

The tested shell helper selects the diagnostic package prefix and domain 70,
resolves the exact dependencies, and passes both immediate-symbol-resolution
library loads. No physical hardware instance was created, no physical ROS motion
command was sent, and no active process was restarted. The startup **dependency**
failure is resolved in the tested process environment; a complete physical launch
still requires operator approval and may expose other failures.

| Check | Result |
| --- | --- |
| Clean overlays before/after dependency path fix, NUC Ubuntu/Humble CPU | Reproduced failure, then successful `ldd -r` and hardware/diagnostic `RTLD_NOW` loads. |
| Committed shell helper staged at `/tmp/fr3_controller_env_investigation.sh` | Selected `~/fr3_controller_diag_build/install/franka_arm_controllers`; domain 70; both loads passed. |
| `bash -n toolkits/realworld_check/fr3_controller_env.sh` | Passed. Direct execution and a Jazzy-contaminated shell are rejected. |
| Focused workstation pytest, Python 3.11 CPU | 27 passed: diagnostic staging/reference patch, supervised pulse guards, request adapter and ROS executor validation. |
| Baseline NUC CTest, domain 213/localhost | Existing six C++ controller cases passed before test extension. |
| Extended isolated C++ target + CTest | Eight cases passed, including both new small-request cases; compiler reports an existing deprecated libfranka Coriolis overload. |
| Fresh CMake Debug build under `/tmp/fr3_motion_investigation_build` + CTest, NUC CPU | Controller and test sources compiled from scratch; all eight cases passed in localhost domain 213. No installation performed. |
| Ruff on existing diagnostic generator, `git diff --check` | Passed; Ruff reports existing inactive preview-rule configuration notices. |
| Docs symbol checker | Failed with 24 findings in unchanged EN/ZH Sphinx pages, outside this report. |
| RST markup checker | Could not run: system Python lacks `docutils`. |
| Sphinx EN/ZH and hardware e2e | Skipped: this experiment Markdown is outside the Sphinx trees; physical operation is awaiting human approval. |

The focused Python command was:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/unit_tests/test_fr3_controller_diagnostic_stage.py \
  tests/unit_tests/test_fr3_controller_diagnostic_repository_patch.py \
  tests/unit_tests/test_fr3_supervised_arm_pulse.py \
  tests/unit_tests/test_fr3_pi05_request_adapter.py \
  tests/unit_tests/test_fr3_pi05_ros_executor.py
```

These checks cover the existing guards and adapters; they do not contact ROS or
exercise the physical controller.

Apply the committed C++ test patch only to a disposable copy of the existing
diagnostic source package. On the NUC from the repository root, these commands
check the patch, preserve a numbered test-source backup, and exercise the existing
isolated build without installation:

```bash
package="$HOME/fr3_controller_diag_build/src/franka_arm_controllers"
patch -d "$package" -p1 --dry-run < \
  docs/experiments/openpi_recap_franka/fr3_startup_fix/controller_motion_tests.patch
patch -d "$package" --backup --version-control=numbered -p1 < \
  docs/experiments/openpi_recap_franka/fr3_startup_fix/controller_motion_tests.patch
source toolkits/realworld_check/fr3_controller_env.sh \
  "$HOME/fr3_controller_diag_build/install"
export ROS_DOMAIN_ID=213 ROS_LOCALHOST_ONLY=1
cmake --build "$HOME/fr3_controller_diag_build/build/franka_arm_controllers" \
  --target joint_impedance_ik_controller_test -j2
ctest --test-dir "$HOME/fr3_controller_diag_build/build/franka_arm_controllers" \
  -R '^joint_impedance_ik_controller_test$' --output-on-failure
```

The inspected NUC isolated source already has this test patch applied. A repeated
dry run should reject duplicate application. These tests need no robot and use
only localhost domain 213; exit their shell before any domain-70 validation.

For the clean compilation, after the same source setup/domain isolation, the
additional build commands were:

```bash
cmake -S "$package" -B /tmp/fr3_motion_investigation_build \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build /tmp/fr3_motion_investigation_build \
  --target joint_impedance_ik_controller_test -j2
ctest --test-dir /tmp/fr3_motion_investigation_build \
  -R '^joint_impedance_ik_controller_test$' --output-on-failure
```

Debug/coverage was retained for offline verification; this is not a real-time
performance qualification. The controller library under the normal workspace
and the installed diagnostic library were not overwritten by this build.

The upstream `RLinf/RLinf` search found related SpaceMouse/setup issues
[#1012](https://github.com/RLinf/RLinf/issues/1012) and
[#1412](https://github.com/RLinf/RLinf/issues/1412), without an established matching
fix. This report is an experiment record outside the EN/ZH Sphinx trees.

## One Supervised Physical Validation

Use one repeat of the existing low-amplitude pulse to locate the limiting stage,
only after explicit human approval. This procedure was **not executed** by the
assistant. It is a controller qualification, not a policy rollout.

1. Have the trained operator inspect the workcell, keep the approved hardware
   stop available, and select an obstacle-free starting pose with no peg contact.
   Stop competing SpaceMouse/policy publishers through the operator's normal
   procedure. Do not reset faults, bypass safety inputs, or change gains/scales.
2. In a fresh NUC shell, source the helper with the diagnostic prefix. Check
   `ros2 pkg prefix franka_arm_controllers` and both library-load results before
   using the operator-approved launch/activation procedure. The existing
   `aist_fr3_config.yaml` specifies `load_gripper: "false"`, zero mounting RPY and
   real hardware; the **example** config specifies `load_gripper: "true"` and is
   unsuitable for this Robotiq setup. Check the actual selected config. Do not
   treat the environment fix as permission to launch or activate hardware.
3. Verify the selected controller is active, seven effort interfaces are claimed,
   domain 70 has exactly one intended arm subscriber, and no other arm publisher.
   Observe at least two baseline `FR3_DIAG_V1` lines plus fresh measured pose and
   joint telemetry. Missing diagnostic output means stop and inspect the selected
   library/activation; do not pulse to discover whether it is instrumented.
4. On the Jazzy workstation from this repository, run the existing read-only
   preflight, then obtain the tool's operator confirmation for **one** pulse:

   ```bash
   export ROS_DOMAIN_ID=70
   source /opt/ros/jazzy/setup.bash
   /usr/bin/python3 -m toolkits.realworld_check.fr3_supervised_arm_pulse
   # Execute only with explicit approval and the physically present operator:
   /usr/bin/python3 -m toolkits.realworld_check.fr3_supervised_arm_pulse \
     --execute --request 0.03 --duration 0.5
   ```

   Save the compact JSON result and diagnostic lines spanning baseline, pulse,
   and at least two seconds afterward. Observe the existing broadcaster's
   `current_pose`, `measured_joint_states`, `desired_joint_states`, and hardware
   status/faults. Desired joint telemetry is robot-side state, not automatically
   the controller's private `q_desired`. Avoid attributing it as such.

5. Expect positive X **request** receipt (`last_x=0.03`), positive deltas in
   `moving_rx`, `ik_attempt`, `ik_valid`, `gate_ticks`, and some reference
   advancement, with no additional invalid input. Effort-write ticks should run
   before/during/after. Exact counter equality to publish count is unnecessary.
   The nominal offset is 0.21 mm per measured-pose solve; total physical travel
   has no guaranteed value. Report directional motion relative to baseline noise;
   the tool's 0.2 mm detection threshold is not a predicted displacement.
6. End after this single pulse. The tool requests neutral and observes for 0.5 s;
   neutral holds the last reference. Stop immediately through the approved
   hardware procedure on unexpected direction, contact, oscillation, fault,
   stale telemetry or loss of exclusive publication. The existing tool aborts
   for pose age over 0.2 s or travel over 10 mm; these software guards are not
   hardware stop guarantees. The operator should stop sooner for any unexpected
   movement. Do not raise request magnitude, duration or gains after a weak result.

Interpret **counter deltas** across that pulse: no receipt localizes the problem
to delivery/domain/exclusivity; receipt without attempts points to sampling or
freshness/activation; attempts without valid results point to IK/bounds; valid
results without gate ticks point to target freshness; gate ticks without any
reference advancement require checking whether a target was already reached.
Reference and effort advancement without measured response requires joint/torque
and hardware evidence before a controller change. A successful single X test
still does not qualify six-axis policy execution or provide any of the required
20 policy outcomes.

## Rollback

Exit the shell that sourced the helper to discard its environment changes. No
system files or loaded binaries changed. Only the isolated diagnostic **test**
source was patched, with a numbered backup; reverse that test patch in the same
isolated package to withdraw it. Production controller source and gains are
unchanged. Keep the ordinary NUC workspace and existing controller backups intact.
Select its overlay in a fresh shell to withdraw diagnostic selection at the next
operator-approved maintenance window. No controller restart is part of rollback
without human approval. Revert only the specific investigation commits if
withdrawing repository changes; do not reset the branch or replace unrelated
NUC source.
