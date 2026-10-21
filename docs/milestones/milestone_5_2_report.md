# Milestone 5.2 Report — Extended-Horizon StackCube Evaluation

## Status

**PASSED for completing the requested evaluation and behavioral diagnosis.**
The 150-step pilot and all 20 paired seeds ran to completion with the native
ManiSkill success evaluator active. Neither policy achieved task success:
pretrained π₀.₅ scored 0/20 and the Milestone 4 SFT checkpoint scored 0/20.
This is an evaluation result, not evidence of successful robot learning.

## Starting State and Preserved Artifacts

- Repository: `/home/muttaqien/Desktop/AIST-Tokyo/gamma/rlinf-project`.
- Branch: `branch-openpi-recap`; starting/current HEAD: `eeb5f39d`.
- The starting worktree already had an M5 progress-log modification and
  untracked M5/M5.1 reports. Those changes were preserved.
- Dataset: `outputs/stackcube_recap/lerobot_stackcube_v2`.
- Norm statistics: `outputs/stackcube_recap/openpi_norm/stackcube_panda/norm_stats.json`.
- Pretrained checkpoint:
  `checkpoints/torch/pi05_stackcube_panda_base`.
- SFT checkpoint:
  `outputs/stackcube_recap/full_sft/results/stackcube_panda_pi05_full_sft/checkpoints/global_step_1084/actor/model_state_dict`.
- The original M5 evaluation configuration, metrics, and videos were not
  modified. No dataset or model checkpoint was changed.

## Evaluation Protocol

Created the ignored diagnostic helper
`outputs/stackcube_recap/milestone5_2_eval/evaluate_extended.py` and separate
pilot/paired configurations under the same output directory. The evaluator
uses the existing RLinf OpenPI model factory, the M4 resolved model
configuration, the same StackCube normalization file, and both verified model
checkpoints.

The environment remained `StackCube-v1`, `panda_wristcam`, RGB observations,
`pd_ee_delta_pose`, one environment, and CPU PhysX. Action representation is
7D; model output chunks are `(1,10,7)`. The live ManiSkill wrapper reported
`_max_episode_steps=150` after the supported `max_episode_steps=150` override.
The native `StackCubeEnv.evaluate()` and environment termination were kept
active. Every episode ended at step 150 with `truncated=true` and
`terminated=false` because no task success occurred.

The base and hand RGB observations and the 18D qpos-then-qvel state were
refreshed from the simulator every step. A new policy prediction was requested
after every ten executed actions. Actions were submitted unchanged to the
existing controller. Logs also capture actual arm/gripper controller target
joint positions after command processing.

Two setup-only validation attempts stopped before model loading or simulator
transitions: the first expected `env.spec.max_episode_steps` to be populated,
but ManiSkill exposes the effective limit through its own wrapper; the second
looked for Gymnasium's generic `TimeLimit` class, while the live environment
uses ManiSkill's `TimeLimitWrapper`. The helper's guard was corrected to walk
the wrapper chain and verify `_max_episode_steps == 150`. No policy, task, or
action behavior was changed. The successful pilot log records the resolved
wrapper value.

### Commands

Pilot (three SFT episodes, seeds 200–202):

```bash
LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib \
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD:${PYTHONPATH:-}" \
timeout 1200s .venv/bin/python \
  outputs/stackcube_recap/milestone5_2_eval/evaluate_extended.py \
  --phase pilot --episodes 3 --seed 200 --policies sft --save-video
```

Paired evaluation (20 matched seeds, pretrained then SFT):

```bash
LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib \
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD:${PYTHONPATH:-}" \
timeout 3600s .venv/bin/python \
  outputs/stackcube_recap/milestone5_2_eval/evaluate_extended.py \
  --phase paired --episodes 20 --seed 42 \
  --policies pretrained sft --save-video
```

Both runs also used `HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets`,
`MPLCONFIGDIR=/tmp/rlinf-mpl`, `MPLBACKEND=Agg`, and unbuffered Python output.
The M3.1 process-scoped NVIDIA library workaround was used; no system CUDA,
driver, kernel, or library configuration was changed.

## Pilot Results

| Policy | Seed | Steps | Native success | Grasp | Lift | Transport | Failure stage |
|---|---:|---:|---|---|---|---|---|
| SFT | 200 | 150 | No | No | No | No | `no_grasp` |
| SFT | 201 | 150 | No | Yes | No | No | `grasped_not_lifted` |
| SFT | 202 | 150 | No | No | No | No | `no_grasp` |

Pilot success rate was 0/3. The pilot validated that the 150-step wrapper,
chunk refresh, step-level instrumentation, and native success/termination
reporting behave as configured. It is separate from the final paired results.

## Paired Results

Seeds 42–61 were reset independently for each model. All 20 initial scene
signatures (robot qpos/qvel and both cube poses) were identical across policies
for each paired seed.

| Policy | Episodes | Successes | Success rate | Failure stages |
|---|---:|---:|---:|---|
| Official pretrained π₀.₅ | 20 | 0 | 0% | 20 no grasp |
| Milestone 4 SFT | 20 | 0 | 0% | 13 no grasp; 3 grasped/not lifted; 2 lifted/not transported; 2 transported/not stacked |

Per-seed failure classifications:

| Seed | Pretrained | SFT |
|---:|---|---|
| 42 | no grasp | no grasp |
| 43 | no grasp | no grasp |
| 44 | no grasp | no grasp |
| 45 | no grasp | no grasp |
| 46 | no grasp | no grasp |
| 47 | no grasp | no grasp |
| 48 | no grasp | grasped, not lifted |
| 49 | no grasp | no grasp |
| 50 | no grasp | grasped, not lifted |
| 51 | no grasp | no grasp |
| 52 | no grasp | no grasp |
| 53 | no grasp | lifted, not transported |
| 54 | no grasp | grasped, not lifted |
| 55 | no grasp | no grasp |
| 56 | no grasp | no grasp |
| 57 | no grasp | no grasp |
| 58 | no grasp | transported, not stacked |
| 59 | no grasp | lifted, not transported |
| 60 | no grasp | transported, not stacked |
| 61 | no grasp | no grasp |

The SFT policy showed more task-directed behavior than the pretrained policy
under these conditions: native grasp events occurred on 7/20 SFT episodes and
none of the pretrained episodes. Four SFT episodes crossed the 2.5 cm lift
threshold; two crossed the 3 cm transport-stage threshold. The SFT cube came
within 3.3 cm and 3.9 cm XY distance of the target in seeds 58 and 60, but no
episode met ManiSkill's task-success condition. These descriptive counts do
not establish a general performance improvement; paired task success was tied
at 0/20.

## Trace and Behavioral Diagnosis

The diagnostic saved 150 JSONL rows for each policy/seed, totaling 6,000
paired transitions. A post-run audit confirmed:

- Each trace contained steps 1–150 in order, one `truncated=true` final step,
  and no termination or task-success flags.
- The action chunk refresh flag was true at every 10-action boundary (300
  refreshes per policy) and false on the other actions; simulator observation
  refresh was recorded on all transitions.
- Predicted and submitted actions were equal on every transition and all
  predicted actions were finite. The action shape was checked as `(1,10,7)`
  before execution.
- Per-step rows include seed, step, end-effector pose, red cube pose, green
  target pose, TCP-to-cube distance, normalized gripper command, observed
  gripper joint positions, predicted and executed actions, controller target
  joint positions, native success, grasp, cube-on-cube, lift/transport
  indicators, and termination/truncation flags.

Across the paired traces, pretrained TCP-to-red-cube distance reached a
minimum of 11.8 cm and the red cube stayed at its initial 2 cm height. Its
gripper command covered both open and closed values, but no simulator grasp
occurred. SFT reached 0.5 cm TCP-to-cube distance; its red cube reached a
maximum of 13.3 cm height. SFT gripper qpos varied from 0 to 0.04 m, and both
open and close commands occurred. This indicates the SFT failure is not
uniformly “never approaches” or “never closes”: some SFT episodes grasped and
lifted. The remaining failures occurred during lift, transport, or stacking.

For action magnitude and direction, the 3D translation command is interpreted
in the Panda controller's root-translation frame. Across all paired
transitions, pretrained translation-command norm had mean/median 0.142/0.121;
SFT had 0.084/0.062 (normalized controller input units). The cosine between
that command vector and the current TCP-to-red-cube vector averaged -0.242 for
pretrained and +0.178 for SFT; it pointed toward the cube (positive cosine)
on 33% and 61% of transitions, respectively. This is descriptive across
changing states and does not account for the full IK/control response. It is
consistent with the smaller SFT TCP distances and partial grasps, while neither
policy completed the task.

Representative SFT seed 58 reaches the cube, grasps, lifts, and reduces
red/green cube XY separation to 3.3 cm, but does not place a stable stack.
SFT seed 53 lifts the cube to 13.3 cm but does not transport it to the target.
In seed 60, the policy grasps, lifts to 10.5 cm, and approaches to 3.9 cm XY
separation without native success. Pretrained seed 42 never grasps and remains
far from the cube. Representative MP4s are provided below. Each is a
side-by-side base-camera/hand-camera recording at 20 fps; the source NPZ files
retain both full-resolution streams.

| Recording | File |
|---|---|
| Pretrained seed 42 | `outputs/stackcube_recap/milestone5_2_eval/videos/paired_pretrained_seed_42_side_by_side.mp4` |
| SFT seed 53 | `outputs/stackcube_recap/milestone5_2_eval/videos/paired_sft_seed_53_side_by_side.mp4` |
| SFT seed 58 | `outputs/stackcube_recap/milestone5_2_eval/videos/paired_sft_seed_58_side_by_side.mp4` |
| SFT seed 60 | `outputs/stackcube_recap/milestone5_2_eval/videos/paired_sft_seed_60_side_by_side.mp4` |

Expert trajectories are 94–120 actions and the new horizon is 150, so the
50-step duration mismatch diagnosed in M5.1 is removed for this diagnostic.
Expert-state replays in M5.1 succeeded, while this closed-loop policy still
fails at varying stages. The evaluation starts are different from the expert
recorded initial states, so episode-level trajectories are not directly
time-aligned. M5.1's HDF5/LeRobot/action-space checks found no representation
or gripper-sign fault; the current traces likewise show no action mutation or
chunk-cadence defect. Plausible contributors include closed-loop distribution
shift, limited demonstration coverage, and placement precision. This run does
not isolate one as the sole cause. No action semantics were changed to
increase success.

## Resources, Validation, and Artifacts

- Physical GPU 1 was selected after preflight: 97,235 MiB free of 97,887 MiB,
  0% utilization, no compute processes. Workload was limited with
  `CUDA_VISIBLE_DEVICES=1`; simulator physics ran on CPU.
- Peak CUDA allocated/reserved memory: pretrained 6,808.2/6,968.0 MiB; SFT
  6,809.4/6,958.0 MiB. Paired configuration-to-summary elapsed time was
  4m00s; mean episode runtime was 4.42s for each policy. The three pilot
  episodes averaged 4.59s. The full paired run used approximately four minutes
  of wall time after setup.
- Postflight: GPU 1 had 97,235 MiB free and 0% utilization with no compute
  process; system RAM had 473 GiB available; disk had 114 GiB free. The M5.2
  output directory occupied 434 MiB at the time of inspection.
- `python3 -m py_compile outputs/stackcube_recap/milestone5_2_eval/evaluate_extended.py` — passed.
- `--help` — one initial attempt without `PYTHONPATH` failed with
  `ModuleNotFoundError: rlinf`; rerunning with repository `PYTHONPATH` passed.
- Episode/trace/video audit — passed for all 43 episodes (three pilot plus 40
  paired), all 6,450 recorded transitions, 20 identical paired resets, and 40
  paired NPZ recordings. Four representative MP4s were encoded from the saved
  source frames. A contact sheet from pretrained seed 42 and SFT seed 58 was
  visually reviewed; the seed-58 MP4 passed `ffprobe` at 7.55 seconds.
- A sandboxed `nvidia-smi` postflight attempt could not communicate with the
  host NVIDIA driver because the sandbox lacks NVIDIA device access. The
  authorized host-context postflight immediately afterward succeeded and is
  saved at `outputs/stackcube_recap/milestone5_2_eval/postflight.log`.

Persistent outputs:

- `pilot_config.json`, `paired_config.json`, `pilot_summary.json`,
  `paired_summary.json`.
- `pilot_sft_episodes.json`, `paired_pretrained_episodes.json`,
  `paired_sft_episodes.json`.
- 43 source NPZ recordings in `videos/`, four derived representative MP4s,
  and `representative_contact_sheet.jpg`.
- 43 per-step JSONL traces in `traces/`, per-reset initial state JSON files in
  `initial_states/`, and command output logs `pilot.log`, `paired.log`,
  `preflight.log`, and `postflight.log`.

No checkpoint was written or modified. The helper and evaluation data are
under ignored `outputs/`; no repository source code or evaluation semantics
were modified. `docs/CODEX_EXPERIMENT_PROGRESS.md` records the milestone.
No Git commit was created; HEAD remains `eeb5f39d` and no push was made.

## Limitations and Recommended Next Milestone

The paired result is 0/20 success for both policies. SFT has evidence of
partial behaviors, but no stable placement. A 20-seed result gives limited
precision, and only simulator outcomes on this exact reset distribution are
measured. Distances and stage flags are diagnostic; only ManiSkill's native
success signal is counted as task success.

Recommended next step: review M5.2 traces and videos, then define an approved,
narrowly scoped policy/data improvement study that can separate grasp
precision, temporal chunking, and placement failure. Do not begin additional
training, autonomous rollout collection, or RECAP until explicitly authorized.
