# Milestone 5.1 Report — StackCube Policy Failure Diagnosis

## Status

**PASSED for the requested audit and diagnosis.** The primary verified mismatch
is that the evaluator uses ManiSkill's registered 50-step episode limit, while
all ten expert trajectories run for 94–120 actions (mean 108.4) and the replay
tool continues past the Gym truncation boundary. Representative expert replay
still succeeded from the recorded initial environment states. No code defect
was demonstrated, so no evaluator rule, model, or checkpoint was changed and
no new closed-loop policy evaluation was run.

The horizon mismatch means the policy is evaluated with less than half the
demonstration's average action budget. It does not by itself explain why all
policy episodes had no grasp by step 50. The SFT checkpoint closely reproduced
expert action chunks when queried at expert states, which points to a
closed-loop state-distribution or temporal execution gap as an additional
likely cause. That explanation remains a hypothesis, not a verified sole
cause.

## Starting State and Preserved Artifacts

- Repository: `/home/muttaqien/Desktop/AIST-Tokyo/gamma/rlinf-project`.
- Branch: `branch-openpi-recap`, HEAD `eeb5f39d`, seven commits ahead of
  `origin/branch-openpi-recap`.
- Starting worktree already contained the uncommitted Milestone 5 progress-log
  update and untracked `docs/milestones/milestone_5_report.md`; both were
  preserved.
- Dataset, original M4 final checkpoint, pretrained checkpoint, normalization
  statistics, M5 JSON metrics, and all 20 M5 video files were preserved.
- No model training, rollout collection, RECAP run, or remote Git operation
  was started.

## Audit Findings

### Environment, termination, and success reporting

The existing M5 evaluator created ManiSkill `StackCube-v1` with
`robot_uids="panda_wristcam"`, `obs_mode="rgb"`,
`control_mode="pd_ee_delta_pose"`, one environment, CPU PhysX, and RGB array
rendering. These match the raw expert replay metadata. ManiSkill registers
`StackCube-v1` with `max_episode_steps=50`; the evaluator's 50-step cap matches
that native registration.

All 20 M5 episodes used 50 transitions, returned `truncated=true` and
`terminated=false`, and reported `success=false`. The evaluator recorded
ManiSkill's `success` and grasp flags at every transition and retained an
episode-level success if it occurred at any point. Its `no_grasp` category is
consistent with all per-step grasp signals being false. There was no evidence
of success being mislabeled as truncation or of a termination/reporting bug.

The evaluator asked the policy for a `(1,10,7)` action chunk, executed actions
sequentially, stopped on termination/truncation or after the configured
transition count, and queried again after ten actions. That is consistent with
the model's 10-step horizon and RLinf's 10-step environment chunk contract.
No chunk-shape or action-order defect was found.

### Demonstration horizon and expert replay

The source metadata and HDF5 actions agree on these successful demonstration
lengths:

| Episode | Actions | Episode | Actions |
|---|---:|---|---:|
| 0 | 107 | 5 | 120 |
| 1 | 94 | 6 | 119 |
| 2 | 101 | 7 | 101 |
| 3 | 109 | 8 | 110 |
| 4 | 109 | 9 | 114 |

All ten terminal success flags are true. The mean is 108.4 actions; even the
shortest episode is 1.88 times the registered/evaluation horizon. The
demonstration metadata also says `max_episode_steps: 50`, so the metadata
limit did not prevent the motion-planning/replay pipeline from recording later
expert actions.

The host-context ManiSkill replay command used the first recorded environment
state and replayed the first three expert episodes with their recorded actions:

```bash
MPLCONFIGDIR=/tmp/rlinf-mpl MPLBACKEND=Agg PYTHONPATH="$PWD:${PYTHONPATH:-}" \
  .venv/bin/python -m mani_skill.trajectory.replay_trajectory \
  --traj-path outputs/stackcube_recap/raw/StackCube-v1/motionplanning/stackcube_expert.rgb.pd_ee_delta_pose.physx_cpu.h5 \
  --use-first-env-state --count 3
```

Result: `Replayed 3 episodes, 3/3=100.00% demos saved`. The replay executes
all recorded actions even after Gym's 50-step `truncated` flag; thus this
result verifies action replay under the current CPU PhysX/controller setup,
but does not show success within the registered 50-step limit. The failed
sandbox attempt and successful host replay output are retained at
`outputs/stackcube_recap/milestone5_eval/expert_replay_first3.log` and
`outputs/stackcube_recap/milestone5_eval/expert_replay_first3_host.log`.

### Videos and episode metrics

All 20 M5 rollout videos were inspected using contact sheets across the initial
frame and steps 10, 20, 30, 40, and 50, for both base and wrist cameras. The
images show arm movement toward or around the cubes in multiple episodes, but
no grasp or lift. This agrees with the recorded per-episode metrics: 0/10
success for each checkpoint, 0/10 grasp, lift, transport, or placement events,
and `no_grasp` for every episode. No video evidence contradicts the metrics.

## Data, Observation, and Action Checks

- `scripts/validate_stackcube_replay.py` passed for all ten raw episodes:
  7D actions, two RGB streams, observation/action alignment, and terminal
  success flags.
- `scripts/validate_stackcube_lerobot.py` passed using its default TorchCodec
  decoder: 10 episodes, 1,084 frames, 20 decoded videos, expected 128×128
  camera tensors, finite values, action correspondence, and terminal success.
- At five sampled boundaries across episodes 0, 1, and 9, LeRobot's
  `observation.state` exactly matched concatenated HDF5 `qpos` then `qvel`
  (maximum absolute difference 0), and its action exactly matched the raw
  HDF5 action at each sampled index. Image keys decoded to channel-first
  `(3,128,128)` tensors, corresponding to the named base and hand camera
  streams. The M5 evaluator converts live HWC RGB camera frames and the same
  qpos-then-qvel ordering to the policy's batched input shapes.
- The existing StackCube adapter maps base RGB to `base_0_rgb`, hand RGB to
  `left_wrist_0_rgb`, and uses the validated 18D state. Training and inference
  both load the same StackCube statistics and use the OpenPI input transform,
  which resizes both views to the model's 224×224 input. The transform pipeline
  normalizes before model input and unnormalizes predicted actions before the
  7D action adapter returns them. No extra delta conversion is applied to
  `pd_ee_delta_pose` actions.
- Expert gripper targets include 529 `-1` and 555 `+1` commands. Panda's
  normalized gripper controller maps the negative command toward its closed
  lower bound and the positive command toward its open upper bound. The data
  converter copies this dimension unchanged. No gripper sign inversion was
  found.
- A six-state teacher-forced inference comparison used two expert episodes at
  frames 0, 20, and 40. It loaded both the official pretrained model and the
  original M4 SFT checkpoint through RLinf's `get_model`, with the same norm
  stats and transforms. Every output was finite and shaped `(1,10,7)`.
  Mean absolute error across the sampled 10-step chunks was 0.034 for SFT and
  0.274 for pretrained. SFT matched the open gripper command at frame 0 and
  the close transition near frames 40–50 on the sampled expert states. These
  are teacher-forced action comparisons, not closed-loop results. The exact
  predictions and per-dimension errors are in
  `outputs/stackcube_recap/milestone5_eval/expert_action_comparison.json`.

## Root Causes and Remaining Hypotheses

### Verified

1. **Expert/evaluation horizon mismatch.** The native task/evaluator limit is
   50 transitions; successful demonstrations need 94–120. Policy evaluations
   therefore ended before the time span covered by the expert behavior. This
   is a verified limitation of interpreting the 0/10 evaluation against the
   demonstrations.
2. **No policy grasp occurred during the measured horizon.** Per-step task
   signals and videos consistently support the M5 `no_grasp` classifications.
3. **Action and observation data paths are coherent on expert states.**
   Sampled LeRobot state/action rows match source HDF5 data; video features
   decode; SFT outputs closely match expert chunks at the selected states.
4. **Chunk shape/cadence and controller mapping are internally consistent.**
   The evaluator runs the configured 10-action chunk and uses the same 7D
   delta-pose interface as demonstrations. Gripper polarity is consistent.

### Not established

- Ten demonstrations may not cover the evaluation reset distribution well;
  this is plausible but not isolated by the current evidence.
- Ten-step open-loop execution may accumulate prediction error before the
  next observation; this is plausible, but the M5 rollout artifacts do not
  record action traces needed to measure its exact contribution.
- The 50-step limit is a definite confound for comparison to the 94–120-step
  experts, but does not prove that a longer policy rollout would grasp or
  succeed. The policy already had no grasp at step 50.
- Visual contact-sheet review cannot identify the exact first contact/grasp
  opportunity at simulator precision. M5 stored video and per-episode metrics
  but not the predicted action trace, and its reset seeds differ from expert
  starts. The exact first closed-loop divergence therefore cannot be located
  by directly aligning those rollouts. The six expert-state predictions are
  comparisons at recorded matching states, not evidence about where the live
  policy first leaves the expert trajectory.

## Changes, Tests, and Controlled Evaluation

No tracked code, model, configuration, normalization statistic, or task rule
was changed. No correction was justified by the audit. In particular, the
50-step limit was left unchanged because it is ManiSkill's registered task
limit; extending it solely to raise the success count would change the
evaluation rule. No new closed-loop policy episode was run, and the original
paired M5 metrics remain untouched.

Executed checks:

- `scripts/validate_stackcube_replay.py` — passed, 10/10 valid expert
  trajectories.
- `scripts/validate_stackcube_lerobot.py` with default TorchCodec — passed,
  10 episodes / 1,084 frames.
- Five sampled HDF5-versus-LeRobot state/action checks — passed with zero
  maximum absolute state/action difference.
- Expert replay from recorded initial state — passed, 3/3 episodes.
- Pretrained/SFT expert-state inference comparison — completed; all 12 action
  chunks finite with shape `(1,10,7)`.
- The validator's explicit `--video-backend pyav` attempt failed because the
  installed `torchvision.io` does not expose `VideoReader`; the default
  TorchCodec path then passed. No dependency was changed.
- A sandboxed expert replay attempt failed at SAPIEN renderer initialization
  because the sandbox lacks Vulkan/GPU device access. The narrowly scoped
  host-context replay succeeded. This did not indicate a simulator controller
  failure.
- Validation stdout is saved in `m51_expert_dataset_validation.log` and
  `m51_lerobot_validation.log`; the alternate-decoder traceback is saved in
  `m51_pyav_backend_failure.log`. Replay, inference, and GPU inventories are
  saved in the correspondingly named `expert_replay_*`,
  `expert_action_comparison.*`, and `m51_gpu_*.log` files in the same output
  directory.

For bounded inference, the preflight found physical GPU 1 idle with 97,235 MiB
free of 97,887 MiB, 0% utilization, and no listed compute processes. System
memory had 473 GiB available; disk had 114 GiB free. `CUDA_VISIBLE_DEVICES=1`
and the process-scoped NVML path
`/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib` were used. The inference
diagnostic did not capture peak allocated memory. Post-run, GPU 1 again had
97,235 MiB free, 0% utilization, and no compute process. Preflight and postrun
inventories are preserved in `m51_gpu_preflight.log` and
`m51_gpu_postrun.log` under the M5 output directory.

## Files, Git, and Artifacts

- New tracked deliverables: this report and the appended section in
  `docs/CODEX_EXPERIMENT_PROGRESS.md`.
- Untracked M5 report and existing M5 progress changes from the prior
  milestone remain intact.
- Diagnostic helper, JSON action comparisons, replay logs, and resource
  inventories are under `outputs/stackcube_recap/milestone5_eval/` and are
  ignored by Git.
- Starting Git commit: `eeb5f39d`. No Milestone 5.1 commit was created; no
  remote branch was modified.
- `git diff --check` is run after the documentation update.

## Limitations and Recommended Next Milestone

Milestone 5.1 identifies a verified training/evaluation horizon mismatch but
does not show that it alone caused failure. The 0/10 no-grasp result remains
valid under ManiSkill's registered 50-step horizon. The data mapping and SFT
action decoder show no demonstrated implementation fault. The small dataset
and chunked closed-loop distribution shift remain plausible contributors.

The next research decision should define an evaluation protocol aligned with
the task and demonstrations before collecting any new data: either retain the
official 50-step benchmark and produce demonstrations executable within that
limit, or explicitly approve and document a longer-horizon diagnostic as a
separate metric. Any further rollout collection, policy retraining, or RECAP
training remains outside this milestone and requires approval.
