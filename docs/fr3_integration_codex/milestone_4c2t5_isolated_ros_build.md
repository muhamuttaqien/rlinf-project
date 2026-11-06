# 4C.2T.5 — Isolated NUC ROS controller build

Status: GitHub staging script prepared; actual NUC compile pending. The script clones the real NUC franka_arm_controllers package into ~/fr3_telemetry_isolated_4c2t5, inserts the bounded telemetry queue and a post-update controller-reference hook into the COPY, and builds only that isolated copy. It never changes, restarts or loads the running controller.

Workstation action using existing SSH alias nuc-franka:

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
tar -czf - toolkits/realworld_check/stage_fr3_offline_telemetry.py toolkits/realworld_check/fr3_rt_telemetry_queue.hpp toolkits/realworld_check/fr3_rt_telemetry_capture.hpp | ssh nuc-franka 'mkdir -p ~/fr3_offline_transfer && tar -xzf - -C ~/fr3_offline_transfer && python3 ~/fr3_offline_transfer/toolkits/realworld_check/stage_fr3_offline_telemetry.py --build'
```

The build output goes only to ~/fr3_telemetry_isolated_4c2t5/{build,install,log}. Do not source this isolated install into the active controller environment or start its plugin. If the source is unexpected, the script refuses to patch it. It also refuses to overwrite an existing staged directory.

The code queues internal IK and post-limit q_desired_ snapshots, but a non-RT consumer/recorder is not yet installed, so even a successful build does not produce external telemetry or training labels. Some timestamps unavailable in the deployed controller remain zero instead of being invented. Separate safety and real-time review is required before deployment.

Local synthetic staging checks: 2 passed (isolation and fail-closed source detection). NUC build result not yet known.
## ROS environment fix and existing-stage build retry

The first NUC attempt **staged successfully**, but sourcing `/opt/ros/humble/setup.bash` failed on `AMENT_TRACE_SETUP_FILES: unbound variable` before any compilation. The staging script now uses `set -eo pipefail` (without `-u`) for ROS shell setup.

**Do not stage again:** the isolated directory already exists and staging intentionally refuses overwrite. Retry its build from the workstation:

```bash
ssh nuc-franka 'bash -c '\''set -eo pipefail; source /opt/ros/humble/setup.bash; source "$HOME/Desktop/aist-gamma/franka_controller_ws/install/setup.bash"; D="$HOME/fr3_telemetry_isolated_4c2t5"; colcon --log-base "$D/log" build --base-paths "$D/src" --packages-select franka_arm_controllers --build-base "$D/build" --install-base "$D/install" --event-handlers console_direct+'\'''
```

This recompiles only the isolated copy. No installed live binary is overwritten, no ROS node is launched, and no robot motion is commanded.
