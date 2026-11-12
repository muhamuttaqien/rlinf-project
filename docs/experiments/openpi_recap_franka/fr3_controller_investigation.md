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

## Command Path Findings

Source inspection confirms that the topic carries normalized operator requests.
The controller adds `0.007 * linear_request` metres to the measured flange pose
used for each IK solve. A request of 0.03 therefore asks IK for a 0.21 mm offset.
This is neither 0.03 m/s nor a guaranteed 0.21 mm measured displacement per
message. Full pipeline evidence and offline command-path tests are pending.

## Validation And Remaining Work

The dependency reproduction and both immediate-symbol-resolution library loads
passed on the Ubuntu/Humble NUC CPU. No hardware object was constructed, no ROS
motion command was sent, and no active process was restarted. The startup
**dependency** failure is resolved in the tested process environment; a complete
physical launch still requires operator approval and may expose other failures.

The upstream `RLinf/RLinf` search found related SpaceMouse/setup issues
[#1012](https://github.com/RLinf/RLinf/issues/1012) and
[#1412](https://github.com/RLinf/RLinf/issues/1412), without an established matching
fix. This report is an experiment record outside the EN/ZH Sphinx trees.

## Rollback

Exit the shell that sourced the helper to discard its environment changes. No
system files or loaded binaries changed. Keep the ordinary NUC workspace and
existing controller backups intact. Revert only the specific investigation
commits if withdrawing these repository changes; do not reset the branch or
replace unrelated NUC source.
