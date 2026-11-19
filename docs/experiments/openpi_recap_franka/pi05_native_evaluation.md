# Native OpenPI π₀.₅ FR3 Request Evaluation

Run one operator-supervised evaluation of the existing 3,000-step request-SFT checkpoint through RLinf's `EmbodiedEvalRunner`, `EnvWorker`, and `MultiStepRolloutWorker`. Use `FR3RequestEnv-v1` and `FrankaRobot` to attach to the active NUC controller and Robotiq bridge.

## Training Contract

| Input or output | Native evaluation contract |
| --- | --- |
| State | 19 measured float32 values: `current_pose` position + quaternion xyzw (7), named `fr3_joint1..7` positions (7), base-frame force xyz (3), physical `robotiq_85_left_knuckle_joint` radians (1), wrench torque norm (1) |
| RGB | Base serial `337122075473`, 640×480 at 30 Hz; wrist `218622277827`, 848×480 at 30 Hz. Camera BGR converts once to RGB. |
| Model | `pi05_fr3_peg_request_offline`, `Pi0Eval`, continuous state input, padded model dimension 32, ten-step horizon, seven decoded outputs |
| Statistics | `/data1/Muttaqien/fr3_peg_request_norm_v1/norm_stats.json`; SHA-256 `9bc91e36bd59e20e57f073a578a5c8e23977598a58f6ccd0f793c410522f9cba`, checked before workers start |
| Prompt | `Insert the peg into the hole.` |
| Arm output | Six normalized SpaceMouse requests in unchanged Twist order/signs, scaling **1.0**. The NUC controller owns its IK conversion and hardware limits. |
| Gripper output | Normalized request, distinct from the measured knuckle angle. Three consecutive close ≤0.3 or open ≥0.7 predictions trigger the existing bridge; minimum goal interval 0.8 s. Predictions within [-0.06, 1.06] saturate only the gripper to [0, 1]. |

The configuration loads the existing base architecture/assets, then strictly loads `runner.ckpt_path` from `global_step_3000/actor/model_state_dict/full_weights.pt`. It creates evaluation workers only. No training or optimizer runs. The request path uses the same OpenPI quantile normalization and output unnormalization as training, once each.

The state preserves the recorded pose convention. The nominal flange/EE transform is not evidence of a calibrated peg tip. This evaluation measures request transmission and observed robot response; it does not certify TCP calibration or task success.

## Existing Runtime Interfaces

The ROS process uses system Python 3.12, `/opt/ros/jazzy/setup.bash`, and the already-built `/tmp/fr3_calibration/jazzy_msgs/install/local_setup.bash` overlay. That overlay supplies the NUC's Humble `franka_msgs` and `controller_manager_msgs` schemas compiled for Jazzy. Keep it available; stock Jazzy's controller-manager schema differs. The ML worker keeps Python 3.11. ROS setup runs inside the ROS child and preserves its required package paths.

The adapter reads arm state and controller-manager services on domain 70, and filters physical gripper measurements on domain 71. It reserves `/franka_controller/target_cartesian_velocity_percent` and `/gripper_client/target_gripper_width_percent` after finding zero other writers, exactly one subscriber on each command topic, and the active `joint_impedance_ik_controller` claiming precisely seven effort interfaces. It never launches a controller or opens a libfranka/serial session. `RLINF_ROS_SETUP=preserve` prevents the generic environment setup from stopping existing ROS processes.

Every accepted arm sample expires after 100 ms. Policy samples execute at 10 Hz within each ten-step horizon; the ROS child refreshes the active Twist at 100 Hz. Between synchronous inference chunks, expired samples become neutral. The three-second wall-clock window starts at the first accepted sample and includes inference gaps; it can contain fewer than 30 executed policy samples. No delayed sample extends that deadline.

State receipt age is bounded to 250 ms, source timestamps must advance, and the observation used for inference must be at most one second old when its chunk begins. Controller ownership is checked repeatedly. An operator stop, driver/parent loss, stale state, controller error/mode change, competing writer, command refresh gap over 50 ms, or measured translation over 8 cm latches neutral. Reset only observes, and a second reset refuses rearming. The active NUC retains all controller gains and hardware limits.

**Neutral holds the NUC's last joint reference. Use the physical emergency stop for an immediate robot stop.**

## Operator Procedure

1. Keep the existing NUC arm controller and Robotiq bridge running. Ensure the two cameras are available and that SpaceMouse/other command publishers have released the two request topics. Leave the working SpaceMouse configuration intact.
2. Put the robot and peg in a clear, supervised starting configuration using your established manual procedure. Keep an operator at the physical emergency stop. Check the workspace and gripper contents before permitting open/close requests.
3. From the repository root, activate the existing ML environment and run the command below. The NVIDIA 590 libraries are selected only for this process because this workstation also contains 610 user-space libraries. RLinf selects physical GPU 1 through the YAML placement fields; check that GPU remains available.

```bash
source .venv/bin/activate
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.590.48.01:/usr/lib/x86_64-linux-gnu/libcuda.so.590.48.01 \
RAY_ADDRESS=local \
bash evaluations/run_eval.sh realworld realworld_fr3_pi05_request_eval
```

This starts an isolated local Ray instance, loads the checkpoint, and attaches the native task. Before displaying the operator prompt, the rollout worker performs two discarded synthetic predictions and restores its random-generator state. After confirmation, reset acquires the actual rollout observation; the one-second freshness limit remains active. Initialization sends no robot commands. It prints a unique `/tmp/rlinf-fr3-native-...` session directory, then asks for consent after worker initialization. Decline or press Ctrl-C to end without authorizing a request window.

4. Once the robot, gripper, and emergency-stop operator are ready, type **`RUN FR3 3S`** at that prompt. This grants one three-second request window. Supervise continuously. Use Ctrl-C or `touch /tmp/rlinf-fr3-native-<printed-id>/STOP` for software neutral; use the physical emergency stop when needed.
5. After the run, inspect `env_timing.jsonl` and `policy_timing.jsonl` for acquisition, wrapping/transport, inference, and action-admission times. Inspect the session's `ros_evidence.jsonl` and `ros.stderr`, alongside the normal RLinf evaluation log. Compare `policy_request.value[:6]` with `twist_refresh.value`, controller diagnostic counters, and `observed_state.state[:3]`. Every non-neutral Twist must retain gain 1.0, and all refreshes after the deadline must be neutral. Repeat only with a new explicit operator confirmation.

No physical calibration record or 17-check approval file participates in this native path. Technical configuration checks and runtime supervision remain mandatory. The software does not report a successful peg insertion from zero reward or from the absence of a fault.

## Offline Verification

Run each component test in its own process, preserving the existing Python path:

```bash
PYTHONPATH="$PWD:$PWD/tests${PYTHONPATH:+:$PYTHONPATH}" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q tests/unit_tests/test_fr3_native_eval.py
```

These tests use fake ROS and camera SDKs while exercising the real native vector/chunk environment, `FrankaRobot`, and ROS subprocess. They cover state ordering, RGB conversion, scaling 1.0, expiry, deadline, neutral tails, controller ownership, stale inference, physical gripper filtering, and operator consent. Run `test_robotics.py` and `test_real_env.py` separately with `-m 'not placement'` for CPU regressions. Placement tests need a dedicated scheduler test run. On a restricted host, give local socket tests access to loopback and set `NUMBA_CACHE_DIR` to a writable temporary directory for the installed robosuite package.

Actual-checkpoint inference on recorded episode 08 passed strict loading and produced finite `(1, 10, 7)` outputs on an RTX PRO 6000. A live read-only native ROS check received the complete 19D state and verified controller/bridge ownership with zero command publishers created and zero commands sent. Physical gain-1.0 performance remains untested until an operator authorizes the command above.

Offline latency verification reproduced the cold first-chunk refusal at 1.182 s: wrapping/outbound transport 0.284 s, inference/output copy 0.619 s, return/admission 0.264 s, and source age at acquisition 0.015 s. After warm-up, a complete native evaluation with the same checkpoint and fake ROS/cameras passed with chunk-admission ages of 0.750, 0.393, and 0.370 s. These are offline measurements, not physical execution results. The evidence is in `pi05_native_latency_evidence.json`.
