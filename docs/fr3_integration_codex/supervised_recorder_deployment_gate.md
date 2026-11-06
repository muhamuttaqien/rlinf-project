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
