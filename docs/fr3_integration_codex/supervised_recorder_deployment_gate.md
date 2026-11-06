# FR3 stationary telemetry — supervised deployment gate

**State (November 7, 2026): not approved for live controller replacement.**

## Verified
- Live controller `joint_impedance_ik_controller` is active in `NUC-Franka:0.0`, launched with `robot_config_file:=aist_fr3_config.yaml`.
- Original controller library is loaded from `~/Desktop/aist-gamma/franka_controller_ws/build/franka_arm_controllers/libfranka_arm_controllers.so`.
- Backup directory `~/fr3_controller_backup_20261107_055031/` contains the original library, `controllers.yaml`, `aist_fr3_config.yaml`, and `joint_impedance_ik_controller.launch.py`. Original and backed-up library SHA-256 matched: `ea4ac304e079eb5d8ce2f15917bcabe3823ea060c86a8be82702a560937f30c6`. The operator reported successful configuration backup.
- Instrumented controller with default-disabled non-RT recorder compiled in `~/fr3_telemetry_isolated_4c2t7/` and passed a mock-controller gtest. It has **not** been loaded on physical hardware.

## Required physical authorization before live swap
A responsible lab operator must approve a maintenance window, physically secure the FR3 and workspace, verify a usable emergency stop, ensure no teleoperation/policy commands are being published, and approve explicit stop/restart and rollback procedures. Record baseline controller state, software version and recorder-output path before any live change.

**Do not perform an unattended hot-swap or overwrite the original workspace.** Plan an approved, controlled restart with a separately configured recorder-disabled fallback and a rollback to the verified original library/configuration. No physical moving-arm test is authorized by this document. Conduct stationary/neutral-input qualification first and review telemetry drop/timing/RT impact before any motion.

The immediate operator action is to obtain supervisor/lab operator approval for the supervised maintenance window. **No further offline synthetic test is required simply to obtain this approval.**

## After approval (not yet executed)
Prepare a separate approved launch/runtime overlay pointing to the isolated recorder build, a bounded writable JSONL output path, and specific commands to stop, start, validate neutral input, and roll back in the supervised session. Validate that data are truly controller-originated before converting them to any training action labels.

## Runtime overlay preflight (no controller changes)

From the workstation, read the **isolated** ROS package resolution and the active launch/config structure before constructing the approved launch command. Do not stop or swap the active controller at this step.

```bash
ssh nuc-franka 'bash -s' <<'SH'
set -eo pipefail
source /opt/ros/humble/setup.bash
source "$HOME/Desktop/aist-gamma/franka_controller_ws/install/setup.bash"
source "$HOME/fr3_telemetry_isolated_4c2t7/install/local_setup.bash"
export LD_LIBRARY_PATH="/opt/openrobots/lib:${LD_LIBRARY_PATH:-}"

P="$(ros2 pkg prefix franka_arm_controllers)"
EXPECTED="$HOME/fr3_telemetry_isolated_4c2t7/install/franka_arm_controllers"
printf 'RESOLVED_PACKAGE=%s\n' "$P"
test "$P" = "$EXPECTED" || { echo "FAIL: original controller overlay selected"; exit 1; }
echo 'PASS: isolated recorder overlay selected'

LAUNCH="$P/share/franka_arm_controllers/launch/joint_impedance_ik_controller.launch.py"
YAML="$P/share/franka_arm_controllers/config/controllers.yaml"
echo '--- launch wiring ---'
grep -nE 'robot_config_file|controllers.yaml|spawner|controller_manager|package' "$LAUNCH" | head -n 40 || true
echo '--- controller YAML wiring ---'
grep -nE 'joint_impedance_ik_controller|ros__parameters|telemetry_jsonl_path|controller_manager' "$YAML" | head -n 50 || true
SH
```

The `telemetry_jsonl_path` parameter defaults to empty and must be correctly configured in a **separate** approved launch/config path before any stationary recording is possible. Library loader resolution and mock-controller tests do not demonstrate real hardware safety. A supervised maintenance window, physical workspace isolation, E-stop verification, neutral-input qualification, and a rollback procedure remain prerequisites to restarting or replacing the controller.
