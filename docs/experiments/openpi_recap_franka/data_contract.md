# Data contract: verified shapes vs unresolved semantics

The following describes repository implementations, **not** a finalized deployment schema. Every stage must share an explicitly chosen schema before real-robot data are recorded.

## Upstream representation A — real-world OpenPI

`rlinf/models/embodiment/openpi/dataconfig/realworld_dataconfig.py` repacks LeRobot input keys:

| LeRobot field | OpenPI transform field | Expected type/shape in `RealworldInputs` |
| --- | --- | --- |
| `image` | `observation/image` | **base camera**, HWC 3-channel image in candidate transform |
| `extra_view_image` | `observation/extra_view_image` | **wrist camera**, HWC 3-channel image; the experiment requires it even though upstream substitutes blank when absent |
| `state` | `observation/state` | **19 values** required |
| `actions` | `actions` | **(horizon, 7)** float actions at transform input |
| `prompt` | `prompt` | task text |

`RealworldInputs` maps `observation/image` to `base_0_rgb` and `observation/extra_view_image` to `left_wrist_0_rgb` (a model slot name, not a physical left arm), padding state/actions to the policy dimension; `RealworldOutputs` returns 7D actions. This is the **candidate** 0B sample contract, not proof that raw LeRobot exports have this form. A recorded single frame may carry a 7D action; the loader constructs temporal action windows.

**Experiment enforcement:** both images are mandatory. The existing `RealworldInputs` silently inserts a blank second image if missing, so the experiment-only validator now rejects this case *before* the transform. Image camera serials and color channel semantics are not yet verified.

The planned SpaceMouse input is **6D normalized/cartesian twist intent** plus **Robotiq** open/close or width request, NOT the 7D SFT policy target. Neither upstream `franka_spacemouse` `Twist` nor RLinf `SpaceMouse` output establishes a metric translation/rotation action label. Record source commands and measured EE pose separately; define a reproducible action conversion only after the NUC control interface and time alignment are verified.

## Upstream representation B — Franka environment / robot

`rlinf/envs/real/franka/base.py` uses a nested observation with `state.tcp_pose` (7), `tcp_vel` (6), `tcp_force` (3), `tcp_torque` (3), and `gripper_position` (1) for a parallel-jaw gripper; it also contains `frames` keyed by camera name. These total **20 scalar state values** when flattened, **not 19**. The code does *not* establish which field to drop or how to form a 19D policy state. Do **not** silently truncate.

The actual mounted tool is a **Robotiq gripper**. RLinf's `RobotiqGripper` (for 2F-85/2F-140) controls width in metres over Modbus RTU and derives width feedback from raw register position; the `is_open` flag tracks the requested open/close state, not independent grasp success. The precise Robotiq model, calibration, full opening width, physical mounting, TCP/tool offset and feedback validity are pending. Do not reuse Franka Hand width limits or treat a command as proof of gripping.

`FrankaRobotState` stores TCP position + quaternion (SciPy `as_quat` in the ROS 1 backend, i.e. xyzw there), plus joint and force measurements. Before ROS 2 integration verify target/reference frame, quaternion convention, TCP/flange offsets, gripper width vs normalized command, units, RGB/BGR conversion, camera role mapping and timing. `FrankaEnv._frames_from` currently stores policy frames in BGR order: verify the complete model path before deciding whether conversion is required.

## Upstream representation C — RECAP

`rlinf/data/datasets/recap/value_dataset.py` supports `robot_type=franka` and `franka_co_train` but maps both through **`FrankaEEInputs`**, not `RealworldInputs`. Its repack keys also differ from the two-camera real-world adapter. Default example YAML uses `robot_type=libero`. Consequently, **a 19D two-camera SFT dataset is NOT yet established to work with all four RECAP stages**. Preserve this as a blocking integration gap until an end-to-end dataset-transform audit resolves it; do not rename robot_type merely to silence errors.

RECAP return computation reads `episode_index` and `frame_index`; rollout data additionally require `is_success` to distinguish success/failure. The method generates sidecar returns and later advantages. All stages must use matching datasets and tags.

## Dataset status

**Two real stationary raw pilots have been recorded (November 6, 2026)** with lossless base/wrist frames, measured FR3/Robotiq state and ROS operator events. These are **not human task demonstrations or policy rollouts**. No success-labeled episodes, training action targets, normalization statistics, train/validation splits, RECAP returns or critic checkpoints exist for real Franka data. Use the offline [Milestone 4C preflight](../../fr3_integration_codex/milestone_4c_lerobot_preflight.md) before developing any schema-approved LeRobot export; raw validation does not make the data training-ready.

## Candidate 4C.2 representation (not approved)

An explicit experimental 19D projection has been implemented as an
offline **preview only**, with 7 TCP pose values, 7 measured joint
positions, 3 measured external force components, 1 Robotiq knuckle
angle and 1 external torque magnitude. These selections intentionally
lose torque direction and omit joint velocity/effort; the frame and
calibration assumptions remain unverified. Proposed 7D future targets
are Cartesian delta translation (m), delta rotation vector (rad), and
Robotiq knuckle-joint position target (candidate radians, units/joint mapping to verify). **No metric action labels are generated.**
See [Milestone 4C.2 contract](../../fr3_integration_codex/milestone_4c2_policy_contract.md).
Do not treat this candidate as an approved SFT / RECAP schema. The November 7 NUC binary bridge uses `control_msgs/action/GripperCommand`, target `position=0.0` for open and `position=0.7929` for close (with `max_effort=50.0`), not a metre-valued gripper-width goal. See the revised **candidate v2** contract; controller joint semantics and action timestamps remain to be verified.

## Proposed non-ROS transport-neutral capture record (for later design)

- Source-timestamped camera frames, camera IDs, and capture-time metadata from Workstation.
- Timestamped measured arm/gripper state from NUC; also record robot/controller health.
- Timestamped *commanded* robot and gripper actions with frame/units and mode; keep distinct from measured state.
- Episode/frame IDs, task/prompt, terminal reason, success/failure labels, and precise sampling-alignment policy.
- Convert **only after confirming** the chosen pi0.5 and RECAP transforms; never create guessed 19D state layouts or actuation values.

## Milestone 0B offline validation

`toolkits/realworld_check/franka_pi05_contract.py` requires both base and wrist images in a transform-stage sample and checks static shape/dtype and numeric finiteness only. It cannot validate ROS frame transforms, physical action meanings, dataset timing, safety, or full RECAP compatibility. When actual data arrive, verify LeRobot metadata, action windows, normalization stats, and value dataset transforms against that data.
