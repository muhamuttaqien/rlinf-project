# Milestone 6A — FR3 OpenPI and RECAP Contract Qualification

This report fixes the contract for an offline FR3 dataset adapter before implementing numerical preprocessing. It establishes a lossless 42D structural state and preserves the recorded seven controller-request channels. The generic OpenPI observation container can represent those values, but no FR3 checkpoint, learned action target or physical camera-role mapping is qualified.

**Physical execution remains `NO_GO / NOT_GRANTED`.** Training, policy rollout, RECAP experiments, controller operation and physical acquisition are outside this milestone. The 6B implementation may validate structural samples; it must reject checkpoint execution and predicted-action conversion.

## Baseline and Source Identity

The worktree and remote fresh branch both started at Milestone 5 commit `f4d61ad4f0f99fe75a19297cfd5c5e8a546e2873`, with no local changes. Only this branch and installed dependencies were inspected. The completed [2B.1](milestone_2b1_action_safety_contract.md), [2B.2](milestone_2b2_ros2_action_adapter.md), [recovery revalidation](milestone_2d_recovery_revalidation.md), [3](milestone_3_rlinf_configuration.md), [4](milestone_4_data_collection_integration.md), [5](milestone_5_dataset_validation.md) and [quality evidence](evidence/milestone_5_dataset_quality.json) remain authoritative for their scopes. An AI assistant prepared this work for the human submitter.

Upstream PR/issue searches for `FR3 OpenPI` found no competing implementation. [Issue #1412](https://github.com/RLinf/RLinf/issues/1412) concerns GELLO device/configuration setup and whether FR3 is supported; it does not implement this dataset adapter. The passive recovery capture remains session-specific evidence, with prior abort diagnosis, calibration, timing and physical authorization unresolved. No live revalidation was performed here.

The machine-readable [contract](../../rlinf/data/datasets/openpi/fr3_contract_v1.json) is `fr3_openpi_structural_v1`. Its canonical sorted compact JSON SHA-256 is `e731df69aee38bed82e832f8108f43787a53670e6d621031fbc5ec3565de614c`. It pins the installation by package version and all 54 Python source-file SHA-256 values, independently of the official reference revision.

| Source | Verified Identity and Limit |
|---|---|
| Installed distribution owning `openpi` | **`rlinf-openpi==0.1.1`**; `importlib.metadata.packages_distributions()` identifies ownership. Asking for distribution `openpi` alone returns no metadata. |
| Installed source tree | Canonical file-map hash `2cb921836f35f511ab7fb59c868c56bdfe3f44049c0ada86be0516e69965b647`; all owned Python files match the wheel's RECORD. |
| Installed Git revision | Not embedded in the wheel; `direct_url.json` is absent. No claim that the wheel equals the official reference commit. The version plus exact file hashes pins the inspected implementation. |
| Official OpenPI reference | [`215abfb217dbac7d5f1273282331b9b1866c0479`](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479); model config, training config, transforms and loader inspected at that immutable revision. Their Git blob identities are in the contract. |
| Native RLinf | Fresh-branch baseline above; current model config, data factories, environment wrappers, collector/serialization and RECAP sources inspected. |
| Dependencies relevant to data | LeRobot 0.3.3 / v2.1, NumPy 1.26.4, JAX/JAXlib 0.5.3. CPU-only preprocessing is the validation target. |

The install script selects `rlinf-openpi==0.1.1` for Franka. It was inspected, not run or modified. No package installation, large weight download, tokenizer download or model construction was required for qualification.

## General Model Contract and Checkpoint Gate

OpenPI architecture dimensions describe tensors; they do not assign robot units or coordinate frames. The installed `models/model.py::Observation` accepts arbitrary low-dimensional state arrays and named image dictionaries. `Pi0Config` defaults to action dimension 32 and horizon 50. Its π₀.₅ mode uses a 200-token limit and defaults to discrete state input, with configurable continuous-state behavior. None of those facts mandates a 19D robot state.

| Configuration Inspected | Exact Source-Level Values | Interpretation |
|---|---|---|
| `pi0_realworld` | PI0, dimension 32, horizon 10, 48 tokens, continuous state, `LeRobotRealworldDataConfig` | Contains the embodiment-specific 19D assertion; not a π₀.₅ requirement. |
| `pi05_franka_state` | PI05, dimension 32, horizon 20, 200 tokens, discrete state; Gemma 2B + 300M expert; `LeRobotFrankaEEDataConfig(pad_state=False)` | Useful reference configuration, not a selected FR3 checkpoint. Base factory selects quantile normalization. |
| FR3 collection configuration | Contains environment/recorder configuration; no actor/model/checkpoint binding | No target checkpoint identity is configured. |

The generic examples contain templates such as `checkpoints/torch/pi05_base` and `/path/to/model`; their presence does not select a checkpoint for FR3. `get_openpi_config` can replace checkpoint/asset paths, and the official SFT loader checks model dimension/horizon consistency. No FR3-specific checkpoint/configuration or normalization assets have been verified.

The gate is **`CHECKPOINT_UNVERIFIED`**, with checkpoint identity, selected model configuration, model-facing state dimension, action target and model horizon all null. Padding does not turn 42 values into 32: installed `pad_to_dim` leaves oversized arrays unchanged. In PI05 discrete-state mode, `TokenizePrompt` consumes the raw state before padding, so a 32D action expert alone does not rule out 42D state text. Tokenizer limits, normalization and checkpoint semantics still require qualification; no arbitrary truncation or fitting is permitted. A future gate closure needs the exact checkpoint digest/revision and compatible architecture, original state/action training schema, camera preprocessing and roles, task/tokenizer assets, normalization statistics and action-window/loss-mask policy.

## Why the Legacy Adapter Uses 19 Values

The 19D layout comes from the current generic Franka environment plus wrappers. `FrankaEnv._get_observation` supplies gripper position, force, quaternion pose, torque and TCP velocity. `Quat2EulerWrapper` changes the seven-value pose to xyz + xyz Euler angles; `RealWorldEnv.wrap_obs` concatenates sorted state keys. `RelativeFrame` can additionally express pose relative to reset and twist through an adjoint, depending on configuration. `RealworldInputs` only asserts dimension 19 and pads; it does not define these semantics itself.

| Legacy Sorted Field | Span / Size | Current Source Units and Frame | FR3 Relationship |
|---|---|---|---|
| `gripper_position` | `[0:1]`, 1 | `BaseGripper.get_state`: opening width in metres | FR3 raw knuckle radians and normalized closure do not establish width. |
| `tcp_force` | `[1:4]`, 3 | N, Franky `K_F_ext_hat_K` stiffness-frame estimate | FR3 has stiffness wrench, but calibrated frame equivalence is unqualified. |
| `tcp_pose` | `[4:10]`, 6 | xyz metres + SciPy xyz Euler radians; reset-relative if enabled | FR3 preserves seven-value xyz/xyzw in `synthetic/base`. No calibrated reset/TCP conversion is qualified. |
| `tcp_torque` | `[10:13]`, 3 | N m, same stiffness-frame estimate | Same units do not establish frame equivalence. |
| `tcp_vel` | `[13:19]`, 6 | Cartesian twist, m/s and rad/s; current driver computes Jacobian × joint velocity, optional adjoint wrapper | FR3 explicitly lacks TCP velocity and Jacobian; joint velocities alone cannot replace it. |

This comparison describes the current environment path, not a recovered training schema for an unknown checkpoint. It does not justify manufacturing 19 values from FR3 data.

## Verified FR3 State Representation

The Milestone 3 environment, Milestone 4 serializer and Milestone 5 validator agree on the following 42D layout. The structural state is an exact float32 copy in this order. No field is omitted, rotated, rescaled, converted or added.

| Field | Span / Size | Units / Preserved Reference |
|---|---|---|
| `arm_joint_effort` | `[0:7]`, 7 | N m, `fr3_joint1` … `fr3_joint7` |
| `arm_joint_position` | `[7:14]`, 7 | rad, same joint order |
| `arm_joint_velocity` | `[14:21]`, 7 | rad/s, same joint order |
| `base_force` | `[21:24]`, 3 | N, `synthetic/base` |
| `base_torque` | `[24:27]`, 3 | N m, `synthetic/base` |
| `knuckle_position_rad` | `[27:28]`, 1 | rad, `robotiq_85_left_knuckle_joint` |
| `normalized_closure` | `[28:29]`, 1 | dimensionless `[0,1]`, `robotiq_normalized_joint` |
| `tcp_force` | `[29:32]`, 3 | N, `synthetic/fr3_link2` |
| `tcp_pose` | `[32:39]`, 7 | xyz metres + unit xyzw quaternion, `synthetic/base` |
| `tcp_torque` | `[39:42]`, 3 | N m, `synthetic/fr3_link2` |

Require the exact feature list, exact seven joint names/order, finite values, dimension 42, closure range and unit quaternion within absolute norm tolerance 1e-5. TCP velocity/Jacobian, calibrated jaw width and grip force stay unavailable. Frame labels retain their native meaning and do not establish physical TCP or mounting calibration.

The versioned transformation is **`lossless_float32_identity_v1`**, structurally 42 → 42. The checkpoint-facing state dimension remains **unqualified**. This proposal preserves evidence while supporting OpenPI's generic observation structure; it is not a new claim that a 32D checkpoint accepts 42D state.

## Seven Recorded Channels and Model Targets

The stored seven values are validated/mock-submitted controller requests, not verified execution or known OpenPI targets. `fr3_normalized_pose_offset_v1` uses six dimensionless channels in `[-1,1]`: translation relative to measured FK with inverse mounting rotation and full-scale 0.007 m; right-multiplied RPY rotation in measured `fr3_link8`, full-scale 0.03 rad. `robotiq_normalized_closure_v1` is one absolute normalized closure request in `[0,1]`, not opening width or grip force. These are not SI velocities or dt-integrated increments.

`FrankaEEOutputs` documents absolute xyz/RPY/gripper output but numerically only slices model channels. Current `FrankaEnv.step` applies scaled deltas with left-multiplied rotation; wrappers can add adjoint transformations and binary gripper behavior. These source paths do not prove a selected checkpoint's training semantics and do not match the FR3 controller simply because they have seven channels.

6B must keep operator request, recorded normalized controller request, model training target, predicted action and future controller request distinct. Only the recorded request representation is verified. The model target stays null; prediction-to-controller conversion must always reject in this contract. No `DeltaActions`, quaternion/Euler conversion, gripper width conversion, scale multiplication or pose reconstruction is approved.

## Camera, Language and Temporal Contracts

D435i `image` and D405 `extra_view_image` are verified export storage slots. Both preserve RGB uint8 HWC pixels and native source/frame provenance. Physical viewpoint, intrinsics and extrinsics remain unknown. OpenPI expects `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`; existing adapters use different camera selections and masks.

The structural probe routes D435i/D405 to the first/second named model slots and marks an empty third slot false. This is an explicitly synthetic positional routing test, **not** a base/wrist calibration claim. Preserve the raw images separately, apply the pinned native `ResizeImages(224,224)` aspect-preserving letterbox only for the probe, and use `Observation.from_dict`'s uint8 → float32 `[-1,1]` conversion. A missing required recorded camera must fail; a declared unused model slot may contain zeros with a false mask. No physical role may be inferred from its name.

`ModelTransformFactory` also constructs the tokenizer and `PadStatesAndActions`. Tokenizer/checkpoint assets are unqualified, so 6B must not download them or invent tokens. The derived Milestone 5 synthetic task label is not an authentic instruction. Missing task identity/instruction remains explicit, with no default prompt substitution. Future original instructions require task ID, exact text, version, origin, annotation provenance and episode binding; synthetic test instructions must retain their synthetic origin.

The native LeRobot 0.3.3 loader constructs action queries from `[t / fps for t in range(H)]`, clamps indices within the current episode, repeats boundary values and emits `actions_is_pad`. The 6B structural contract permits this exact behavior for horizons 1–1000, preserves the mask and source frame indices, and also supports rejecting incomplete windows. No query may cross episodes. These are structural test horizons; the selected model horizon is still null. Four-step 10 Hz fixtures can test horizons 1, 4, 20 and 50 without becoming evidence for model performance. Padding is not a future measurement and is never passed as an executable command.

## Normalization and RECAP Metadata

Controller normalization describes request channels. Dataset normalization and output denormalization require checkpoint-bound statistics and feature semantics. Installed OpenPI uses `(x-mean)/(std+1e-6)` or quantile `(x-q01)/(q99-q01+1e-6)*2-1`, with corresponding inverses. Its loader requires statistics unless explicitly skipped; the local inference transform helper can skip missing defaults. Neither behavior qualifies missing assets for FR3.

6B may exercise the pinned `Normalize`/`Unnormalize` with explicit **`synthetic_math_only`** fixtures, exact 42D/7D statistics and method/provenance identities. Do not estimate physical statistics from eight transitions, apply them to model targets, or mark normalization qualified. Actual stats remain null. Reject missing assets, wrong dimensions, nonfinite/invalid scales, wrong contract identity and claims that test fixtures are checkpoint assets.

`fr3_recap_metadata_v1` will bind annotations to dataset ID, source manifest hash, native episode ID and provenance path. Its typed fields cover task ID/instruction/version/origin; episode-level split; source type; policy/checkpoint and rollout identity/iteration; intervention intervals and operator provenance; frame-indexed reward, return and advantage labels; outcome/success and annotation origin; and execution/acceptance provenance. Source categories are synthetic demonstration, human demonstration, policy rollout, correction and offline replay. Splits are unassigned, train, validation and test.

Missing labels must remain null or absent annotations. Unscored fixture zeros are not RECAP rewards/returns, mock completion is not success, and false native intervention flags are not human annotation evidence. Duplicate episode identities across splits fail. Synthetic datasets cannot be reclassified as real demonstrations or policy rollouts. A skeleton with unassigned splits and missing labels can be structurally valid while remaining experimentally unready.

The existing RECAP `utils.py` loads task/return sidecars by episode/frame. `value_dataset.py` requires returns and existing robot transforms; `cfg_model.py` requires matching advantages. 6B must validate metadata without fabricating those sidecars, computing advantages, training a value model or registering a trainable FR3 checkpoint.

## Qualification Checks and 6B Decision

The installed-source audit resolved both reference configs without constructing models. Package ownership/version, all wheel RECORD checks, the complete source hash map, source-backed state spans and immutable official reference files were verified. Original Milestone 5 quality/evidence gates remain unchanged. The contract fixes the permitted structural transformation and the unqualified gates before 6B code is added.

Reproduce the essential contract/source checks from the repository root:

```bash
JAX_PLATFORMS=cpu PYTHONPATH="$PWD" python - <<'PY'
import hashlib, importlib.metadata, importlib.util, json
from pathlib import Path
c = json.loads(Path('rlinf/data/datasets/openpi/fr3_contract_v1.json').read_text())
assert hashlib.sha256(json.dumps(c, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == 'e731df69aee38bed82e832f8108f43787a53670e6d621031fbc5ec3565de614c'
assert importlib.metadata.version('rlinf-openpi') == c['openpi']['version']
root = Path(importlib.util.find_spec('openpi').origin).parent
for name, digest in c['openpi']['python_files_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
assert sum(f['stop'] - f['start'] for f in c['state']['fields']) == 42
assert c['checkpoint']['status'] == 'CHECKPOINT_UNVERIFIED'
assert c['checkpoint']['execution_allowed'] is False
print('6A contract and installed source pins verified')
PY
UV_CACHE_DIR=/tmp/fr3-milestone6-uv-cache python .agents/skills/docs-check/build_docs.py --venv /tmp/fr3-m2a-docs-venv
git diff --check
```

Both EN/ZH Sphinx trees are checked with the harness's default autodoc filtering. This standalone milestone report remains outside those trees and receives separate Markdown-link/source checks. The permitted 6B scope is the structural adapter, native loader/windows, explicit synthetic image preprocessing and normalization mathematics, typed RECAP metadata and compatibility reporting. Semantic model state/action, physical camera roles, authentic language, normalization assets, checkpoint execution and physical authorization must retain their unqualified gates.
