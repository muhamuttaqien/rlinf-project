# 4C.2T.7 — Recorder linked to isolated FR3 ROS 2 controller

**Status: isolated NUC build verified November 7, 2026 (no hardware deployment).** The NUC synthetic recorder demo from 4C.2T.6 reported `saved=2 dropped=1 path=synthetic.jsonl`.

This script clones the existing isolated 4C.2T.5 controller, adds a **separate non-real-time consumer thread** that drains the SPSC queue into JSONL, and compiles a separate `~/fr3_telemetry_isolated_4c2t7` package. The recorder is *disabled by default* through `telemetry_jsonl_path=""`. It starts only when that parameter is explicitly configured and the plugin is activated. The thread stops and joins at deactivation, cleanup and destruction.

**Workstation action:**

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
tar -czf - toolkits/realworld_check/stage_fr3_recorder_controller.py toolkits/realworld_check/fr3_rt_telemetry_recorder.hpp | ssh nuc-franka 'mkdir -p ~/fr3_recorder_stage_transfer && tar -xzf - -C ~/fr3_recorder_stage_transfer && python3 ~/fr3_recorder_stage_transfer/toolkits/realworld_check/stage_fr3_recorder_controller.py --build'
```

Expected (if NUC compilation succeeds): `1 package finished` and `ISOLATED RECORDER CONTROLLER BUILD PASSED; live FR3 untouched.`

**Build only.** Do not launch or load the isolated controller, set the recording parameter, send robot commands, or source the isolated install into the live ROS workspace. The original controller and the earlier isolated build are unchanged. No live telemetry is being written at this stage.

This is raw telemetry, not validated training actions. Safety, actual RT overhead, recorder lifecycle, timing synchronization and the physical motion gate must be reviewed before any deployment.

## NUC build verification (reported)

`Finished <<< franka_arm_controllers [11.2s]` and `Summary: 1 package finished [11.4s]` followed by `ISOLATED RECORDER CONTROLLER BUILD PASSED; live FR3 untouched.` The only reported stderr was an external libfranka/Franka hardware Pinocchio deprecated-method warning, not a compilation failure. This result verifies compilation of the isolated recorder-enabled controller; it does **not** prove recorder runtime behavior, controller lifecycle, real-time safety, or action-label validity.

**Next read-only check:** confirm isolated shared-library presence and that the recorder is explicitly disabled by default (no launch needed).

## Post-build verification (November 7, 2026)

Operator reported: `PASS: Library exists, recorder connected, recording disabled by default.` The read-only check confirmed a nonempty isolated shared library and grepped the isolated source for the JSONL drain and default-empty `telemetry_jsonl_path`. This does **not** validate dynamic linker dependencies, controller lifecycle activation, or capture from hardware. The next action is a read-only dynamic dependency check (`ldd -r`) on the isolated library under the correct ROS overlays.

## Runtime dependencies resolved (November 7, 2026)

The operator located `libpinocchio_default.so.4.1.0` and `libpinocchio_parsers.so.4.1.0` under `/opt/openrobots/lib/`. The **read-only** `ldd -r` check for the isolated `libfranka_arm_controllers.so` passed after prefixing `LD_LIBRARY_PATH=/opt/openrobots/lib:${LD_LIBRARY_PATH:-}` for that shell. No global linker configuration, system libraries, or live controller were modified. **Remaining offline validation**: plugin library loadability and recorder lifecycle behavior; these have not been tested, and no hardware deployment is authorized.

## Isolated runtime loader verification — November 7, 2026

The operator ran Python `ctypes.CDLL` against the **isolated** `libfranka_arm_controllers.so`, with `LD_LIBRARY_PATH=/opt/openrobots/lib:...`, and reported `PASS: Isolated FR3 recorder library loaded successfully.` This verifies loadability only; it did not instantiate or activate a ROS controller or test the recorder thread.

## Next action: offline mock-interface lifecycle test

The existing package contains `joint_impedance_ik_controller_test`, a ROS 2 gtest using a synthetic URDF, a mock `franka_hardware::Model`, and fake state/command interfaces; it does not connect to real FR3 hardware. Re-enable **only in isolated directory**:

```bash
ssh nuc-franka 'bash -s' <<'SH'
set -eo pipefail
source /opt/ros/humble/setup.bash
source "$HOME/Desktop/aist-gamma/franka_controller_ws/install/setup.bash"
export LD_LIBRARY_PATH="/opt/openrobots/lib:${LD_LIBRARY_PATH:-}"
export ROS_DOMAIN_ID=213
export ROS_LOCALHOST_ONLY=1
D="$HOME/fr3_telemetry_isolated_4c2t7"
colcon --log-base "$D/log" build \
  --base-paths "$D/src" --packages-select franka_arm_controllers \
  --build-base "$D/build" --install-base "$D/install" \
  --cmake-args -DBUILD_TESTING=ON \
  -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/openrobots/lib \
  --event-handlers console_direct+
ctest --test-dir "$D/build/franka_arm_controllers" \
  -R '^joint_impedance_ik_controller_test$' --output-on-failure
SH
```

The linker path flag addresses the **previous optional-test-only** `libpinocchio` link failure without any global install changes. The mock test may still expose issues; do not claim it passed until its output is observed. Running the test **does not load the controller into the robot's controller manager**. It does not test real-time overhead or live hardware.
