# Milestone 6 — RECAP Rollout Collection and Dataset Validation

**Status: PASSED for the bounded pilot and dataset-validation objective. The
collected data is not yet fully RECAP training-ready.** The collector completed
ten closed-loop policy episodes, wrote all 1,500 transitions to a local
LeRobot dataset, preserved camera video and raw traces, passed integrity
checks, and was accepted by RLinf's official RECAP return processor. The
advantage/value-model stage does not currently support this StackCube schema.

## Repository and protocol

- Branch: `branch-openpi-recap-codex`
- Starting revision: `417f38277baaa7a6a1b27f11cadebc6d1226db26`
- Starting working tree: clean.
- Environment: ManiSkill `StackCube-v1`, Franka `panda_wristcam`,
  `pd_ee_delta_pose`, `physx_cpu`, base and hand RGB cameras, qpos+qvel (18D).
- Policy: Milestone 4 SFT checkpoint at
  `outputs/stackcube_recap/full_sft/results/stackcube_panda_pi05_full_sft/checkpoints/global_step_1084/actor/model_state_dict/full_weights.pt`.
- Checkpoint SHA-256:
  `1db71780bcad741053517000652f01bb8f8383e43f1b2dacb4da5ab9f02c09c3`.
- Normalization file:
  `outputs/stackcube_recap/openpi_norm/stackcube_panda/norm_stats.json`.
- Normalization SHA-256:
  `9740881e6e1ff87773a0247397826ace1d2935a79e668105ec3eee45170456c7`.
- Horizon: 150 simulator transitions; K=5, meaning the unchanged ten-action
  OpenPI chunk is generated and its first five actions execute before a fresh
  observation and new inference. The collector stops on native success,
  termination, truncation, or the horizon.
- Seeds: 3000–3009. They are separate from recorded prior policy-evaluation
  ranges (42–61, 200–202, and 1000–1029). Expert collection seed coverage was
  not available to verify, so non-overlap with expert seeds is unconfirmed.
- GPU: physical GPU 1, one CUDA-visible device, NVIDIA RTX PRO 6000 Blackwell
  Max-Q. The process-scoped library path was
  `/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib`.

The sandbox GPU preflight could not access the driver. Host-context preflight
then found GPU 1 at 0% utilization with 97,235 MiB free and no compute process;
CUDA initialization succeeded. Host memory showed 472 GiB available and the
filesystem had 112 GiB free. No other processes were stopped and no system
configuration was changed.

## RECAP implementation and data requirements

The repository's `examples/offline_rl/config/recap_compute_returns.yaml` and
`examples/offline_rl/advantage_labeling/recap/process/compute_returns.py`
establish the verified return-labeling interface. For a non-SFT rollout
dataset, each episode needs an `is_success` column; the processor reads the
terminal row's value. It constructs its own reward sequence (`-1` per step,
terminal `0` on success or `failure_reward` on failure), then writes return,
reward, and prompt into a sidecar parquet. The pilot also stores native
ManiSkill rewards for research and audit, but this return processor does not
consume those native rewards. It does not require rollout log probabilities.

The later advantage stage in
`examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py`
requires an observation mapping, return/reward sidecars, a compatible value
model, and checkpoint. `rlinf/data/datasets/recap/value_dataset.py` supports
`libero`, `libero_v2`, `franka`, and `franka_co_train` mappings; its
`FrankaEEInputs` transformation is not semantically valid for this dataset's
18D joint qpos+qvel state and `pd_ee_delta_pose` actions. The advantage mapping
also has no StackCube/Panda joint-state plus base/hand-camera entry. The
available value-model configs are not StackCube configs. No false `robot_type`
alias or incompatible transform was introduced.

Thus the verified requirements are: complete LeRobot episodes, frame/episode
indices, task prompt, observations, actions, and terminal `is_success` for
return labeling; plus a StackCube-compatible value observation mapping and
value-model checkpoint for advantage labels. The pilot has the former, not the
latter. No pairwise success/failure comparison or action-probability field is
required by the inspected offline RECAP return/advantage implementation.

## Collector and saved schema

Added `scripts/collect_stackcube_rollouts.py`, which reuses the Milestone 5.5
RLinf model factory, observation preparation, action-chunk selection, scene
signature, native success evaluator, and stop checks. It passes the predicted
7D action directly to the existing environment controller. Each LeRobot row
stores the pre-action observation and the resulting transition's action,
native reward, success, termination/truncation, and diagnostic stage flags.
Two 128×128 RGB streams are stored as MP4. Extra per-step JSONL traces preserve
pre-action state, post-transition robot/cube/target pose, predicted and sent
actions, gripper readings/controller target, reward and flags, simulator step,
and initial-state signature. Episode summaries contain seeds, initial state,
outcome, runtime, inference calls and latency, and stage markers.

Artifacts are under `outputs/stackcube_recap/milestone6_rollouts/`:

- `pilot/dataset/`: LeRobot dataset with 10 episodes, 1,500 frames, 20 camera
  videos, 18D state and 7D action.
- `pilot/traces/seed_3000.jsonl` through `seed_3009.jsonl`: 1,500 raw transition
  records.
- `pilot/episode_summaries.json`, `pilot/collection_config.json`.
- `pilot/integrity_validation.json`.
- `returns_processed/`: a copy processed by the official RECAP returns stage;
  the original pilot data was not changed.
- `logs/collection.log`, `logs/recap_returns_smoke.log`,
  `logs/lerobot_decoder_probe.log`, and `logs/gpu_postrun.log`. The host GPU
  preflight is preserved at
  `outputs/stackcube_recap/milestone3/logs/milestone6_gpu_preflight_host.log`;
  the sandbox device-access failure is in
  `outputs/stackcube_recap/milestone3/logs/milestone6_gpu_preflight.log`.

The output directory is ignored by Git. No dataset, video, checkpoint, or
generated binary is staged.

## Pilot results

| Metric | Result |
|---|---:|
| Episodes / transitions | 10 / 1,500 |
| Native StackCube successes | 0 / 10 (0%) |
| Grasp heuristic | 2 / 10 |
| Lift by 2.5 cm heuristic | 1 / 10 |
| Transport by 3 cm heuristic | 1 / 10 |
| Native cube-on-cube placement flag | 0 / 10 |
| Terminated / time-limit truncated | 0 / 10; 10 / 10 |
| Episode lengths | 150 each |
| Inference calls | 30 per episode, 300 total |
| Mean inference latency | 180.8 ms per call |
| Mean per-episode runtime | 8.78 s |
| Mean / min / max summed native episode reward | 19.856 / 3.296 / 53.622 |
| Dataset size | 6.05 MB |
| Entire pilot output size | 9.11 MB |
| Peak allocated / reserved GPU memory | 6,808.2 / 6,968.0 MiB |

The stage markers are diagnostics, not task-success labels. For example,
transport is a 3 cm reduction in horizontal cube-to-target distance; it does
not establish a stable grasp or successful transport. All episodes timed out
at 150 steps, while all terminal `is_success` values correctly remained false.

The RECAP returns smoke test used `process_dataset(source, copy,
dataset_type="rollout", gamma=1.0, failure_reward=-300.0, num_workers=1,
tag="pilot")`. It completed on the separate copy and generated 1,500 rows.
The resulting synthetic RECAP return statistics were mean `-374.5`, standard
deviation `43.3003`, min `-449`, max `-300`; the generated reward statistics
were mean `-2.9933`, min `-300`, max `-1`. These are processor labels, not the
native ManiSkill reward distribution.

## Validation and limitations

`scripts/validate_stackcube_rollouts.py` passed: it checked all 10 episode
parquet files; verified all 1,500 states, actions and native rewards against
the raw traces; checked 18D/7D shapes, finite numeric data, contiguous simulator
step indices, no transitions after terminal flags, terminal outcome
consistency, and decoded 60 sampled frames (first/middle/last from both
cameras) with PyAV. Video metadata reports 128×128, 20 fps, and 150 frames per
episode. `py_compile` passed for both scripts and `git diff --check` passed.

LeRobot's own `__getitem__` video decoder could not be used: its selected
torchvision backend raised `AttributeError: module 'torchvision.io' has no
attribute 'VideoReader'`. Direct PyAV decoding passed; the cause appears to be
the local LeRobot/torchvision API combination, not corrupt video. The validator
therefore verifies parquet directly and decodes MP4s via PyAV.

The collection ran with the checkpoint's existing normalization and action
semantics. It records the policy output sent to `env.step`; there is no
separately exposed post-controller clipped action in this API, so the report
does not claim to record controller-internal actuator commands. Partial-stage
counts are low and no successful trajectory was collected. The pilot validates
collection mechanics and return labeling, not policy performance or full
RECAP readiness.

## Scaled collection proposal

At the observed mean, 100 episodes would require about 15 minutes of episode
runtime and roughly 61 MB of LeRobot dataset storage (about 91 MB including
pilot-like traces/metadata). The estimate excludes startup and finalization;
the 10-episode pilot itself took about 90 seconds of episode runtime. A
reproducible first scaled batch could be 100 episodes, seeds 4000–4099, same
checkpoint, K=5, 150-step horizon, one GPU, fixed task/controller/cameras, and
retention of every outcome. This is a proposed budget only and was not started.

Given zero pilot successes and prior zero successes on 30 unseen seeds at both
K=5 and K=10, 100 episodes may still provide mostly failure labels. Any plan to
balance successes/failures would require an authorized strategy change (for
example additional expert or scripted resets/interventions); it must not be
silently achieved by altering native rewards or filtering failed trajectories.
The collected dataset could feed the current return processor after this
schema validation, but cannot yet feed value/advantage training without a
StackCube-specific observation mapping, correct input transforms, compatible
value-model config/checkpoint, and a focused end-to-end validation.

## Changes and next milestone

Source changes: `scripts/collect_stackcube_rollouts.py` and
`scripts/validate_stackcube_rollouts.py`. Documentation changes:
`docs/milestones/milestone_6_report.md` and
`docs/CODEX_EXPERIMENT_PROGRESS.md`. The local commit and final Git state are
recorded after validation; no push was performed.

Recommended next milestone: implement and test a StackCube-specific RECAP value
dataset/observation mapping and value-model configuration, including input
transform validation against the existing 18D state, 7D action, and two camera
streams. After that, request approval for a fixed scaled collection budget.
Do not start RECAP optimization based on this pilot alone.
