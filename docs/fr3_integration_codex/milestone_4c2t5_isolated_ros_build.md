# 4C.2T.5 — Isolated NUC ROS controller build

**Status: COMPLETE for isolated NUC build on November 7, 2026.** Operator reported `Finished <<< franka_arm_controllers [0.54s]` and `Summary: 1 package finished [0.70s]` with `-DBUILD_TESTING=OFF`. No controller launched, and no deployment or live telemetry collection was performed. The script clones the real NUC franka_arm_controllers package into ~/fr3_telemetry_isolated_4c2t5, inserts the bounded telemetry queue and a post-update controller-reference hook into the COPY, and builds only that isolated copy. It never changes, restarts or loads the running controller.

Workstation action using existing SSH alias nuc-franka:

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
tar -czf - toolkits/realworld_check/stage_fr3_offline_telemetry.py toolkits/realworld_check/fr3_rt_telemetry_queue.hpp toolkits/realworld_check/fr3_rt_telemetry_capture.hpp | ssh nuc-franka 'mkdir -p ~/fr3_offline_transfer && tar -xzf - -C ~/fr3_offline_transfer && python3 ~/fr3_offline_transfer/toolkits/realworld_check/stage_fr3_offline_telemetry.py --build'
```

The build output goes only to ~/fr3_telemetry_isolated_4c2t5/{build,install,log}. Do not source this isolated install into the active controller environment or start its plugin. If the source is unexpected, the script refuses to patch it. It also refuses to overwrite an existing staged directory.

The code queues internal IK and post-limit q_desired_ snapshots, but a non-RT consumer/recorder is not yet installed, so even a successful build does not produce external telemetry or training labels. Some timestamps unavailable in the deployed controller remain zero instead of being invented. Separate safety and real-time review is required before deployment.

Local synthetic staging checks: 2 passed (isolation and fail-closed source detection). **Isolated NUC ROS 2 controller build: passed**, after excluding the optional test target.
## ROS environment fix and existing-stage build retry

The first NUC attempt **staged successfully**, but sourcing `/opt/ros/humble/setup.bash` failed on `AMENT_TRACE_SETUP_FILES: unbound variable` before any compilation. The staging script now uses `set -eo pipefail` (without `-u`) for ROS shell setup.

**Do not stage again:** the isolated directory already exists and staging intentionally refuses overwrite. Retry its build from the workstation:

```bash
ssh nuc-franka 'bash -c '\''set -eo pipefail; source /opt/ros/humble/setup.bash; source "$HOME/Desktop/aist-gamma/franka_controller_ws/install/setup.bash"; D="$HOME/fr3_telemetry_isolated_4c2t5"; colcon --log-base "$D/log" build --base-paths "$D/src" --packages-select franka_arm_controllers --build-base "$D/build" --install-base "$D/install" --event-handlers console_direct+'\'''
```

This recompiles only the isolated copy. No installed live binary is overwritten, no ROS node is launched, and no robot motion is commanded.

## November 7 NUC build finding

The isolated controller shared library compiled and linked (77%). The build subsequently failed while linking `joint_impedance_ik_controller_test`: `libpinocchio_parsers.so.4.1.0` and `libpinocchio_default.so.4.1.0` were unresolved through the existing libfranka dependency. This is a **test executable linkage problem**, not a controller source compile failure. The build-only staging script now passes `-DBUILD_TESTING=OFF` to avoid building optional controller tests. It does not resolve or alter system Pinocchio dependencies. Reuse the existing isolated directory and re-run colcon with that flag; no restaging, installation into live ROS workspace, or controller launch.

## Next gate

The instrumented plugin was successfully built **only in the isolated workspace**, but it currently enqueues snapshots without a non-RT consumer, so there are **no observable controller-target recordings yet**. Implement and test an isolated non-RT drain/serialization path before any deployment; physical validation, controller safety review and timestamps must remain separately gated.
