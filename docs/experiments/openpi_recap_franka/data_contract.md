# Data contract: verified shapes vs unresolved semantics

The following describes repository implementations, **not** a finalized deployment schema. Every stage must share an explicitly chosen schema before real-robot data are recorded.

## Upstream representation A — real-world OpenPI

`rlinf/models/embodiment/openpi/dataconfig/realworld_dataconfig.py` repacks LeRobot input keys:

| LeRobot field | OpenPI transform field | Expected type/shape in `RealworldInputs` |
| --- | --- | --- |
| `image` | `observation/image` | HWC RGB-like image; code parses ndarray/channel layout |
| `extra_view_image` | `observation/extra_view_image` | second camera image; code can substitute blank if absent |
| `state` | `observation/state` | **19 values** required |
| `actions` | `actions` | **(horizon, 7)** float actions at transform input |
| `prompt` | `prompt` | task text |

`RealworldInputs` uses two active image slots (base and left wrist), padding state/actions to the policy dimension; `RealworldOutputs` returns 7D actions. This is the **candidate** 0B sample contract, not proof that raw LeRobot exports have this form. A recorded single frame may carry a 7D action; the loader constructs temporal action windows.

## Upstream representation B — Franka environment / robot

`rlinf/envs/real/franka/base.py` uses a nested observation with `state.tcp_pose` (7), `tcp_vel` (6), `tcp_force` (3), `tcp_torque` (3), and `gripper_position` (1) for a parallel-jaw gripper; it also contains `frames` keyed by camera name. These total **20 scalar state values** when flattened, **not 19**. The code does *not* establish which field to drop or how to form a 19D policy state. Do **not** silently truncate.

`FrankaRobotState` stores TCP position + quaternion (SciPy `as_quat` in the ROS 1 backend, i.e. xyzw there), plus joint and force measurements. Before ROS 2 integration verify target/reference frame, quaternion convention, TCP/flange offsets, gripper width vs normalized command, units, RGB/BGR conversion, camera role mapping and timing. `FrankaEnv._frames_from` currently stores policy frames in BGR order: verify the complete model path before deciding whether conversion is required.

## Upstream representation C — RECAP

`rlinf/data/datasets/recap/value_dataset.py` supports `robot_type=franka` and `franka_co_train` but maps both through **`FrankaEEInputs`**, not `RealworldInputs`. Its repack keys also differ from the two-camera real-world adapter. Default example YAML uses `robot_type=libero`. Consequently, **a 19D two-camera SFT dataset is NOT yet established to work with all four RECAP stages**. Preserve this as a blocking integration gap until an end-to-end dataset-transform audit resolves it; do not rename robot_type merely to silence errors.

RECAP return computation reads `episode_index` and `frame_index`; rollout data additionally require `is_success` to distinguish success/failure. The method generates sidecar returns and later advantages. All stages must use matching datasets and tags.

## Proposed non-ROS transport-neutral capture record (for later design)

- Source-timestamped camera frames, camera IDs, and capture-time metadata from Workstation.
- Timestamped measured arm/gripper state from NUC; also record robot/controller health.
- Timestamped *commanded* robot and gripper actions with frame/units and mode; keep distinct from measured state.
- Episode/frame IDs, task/prompt, terminal reason, success/failure labels, and precise sampling-alignment policy.
- Convert **only after confirming** the chosen pi0.5 and RECAP transforms; never create guessed 19D state layouts or actuation values.

## Milestone 0B offline validation

`toolkits/realworld_check/franka_pi05_contract.py` checks a transform-stage sample's static shape/dtype and numeric finiteness only. It cannot validate ROS frame transforms, physical action meanings, dataset timing, safety, or full RECAP compatibility. When actual data arrive, verify LeRobot metadata, action windows, normalization stats, and value dataset transforms against that data.
