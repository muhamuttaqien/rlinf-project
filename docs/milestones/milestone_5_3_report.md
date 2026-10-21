# Milestone 5.3 — OpenPI π₀.₅ Behavioral Failure Analysis

**Status: PARTIAL.** The saved trajectories identify where the SFT policy fails and verify several concrete failure behaviors. They do not isolate a single causal mechanism such as grasp geometry, chunk length, or dataset coverage. No code, model, dataset, or evaluation rule was changed.

## Scope and artifacts

The analysis covers the 20 paired SFT runs from Milestone 5.2 (seeds 42–61), each with 150 simulator transitions, and all 10 successful expert demonstrations. The paired pretrained policy had 0/20 task successes; the SFT policy also had 0/20. SFT stage counts from native per-step traces are 7 episodes with at least one ManiSkill grasp flag, 4 reaching the 2.5 cm lift marker, and 2 reaching the 3 cm targetward transport marker. All 20 episodes truncated at the 150-step limit; native task success and `cubeA_on_cubeB` remained false.

The analysis script and derived files are under `outputs/stackcube_recap/milestone5_3_diagnostics/`:

- `analyze_m53.py` reconstructs per-episode events from saved traces and expert HDF5.
- `sft_episode_event_reconstruction.{json,csv}` and `expert_episode_event_reconstruction.{json,csv}` contain the event metrics.
- `analysis_summary.json`, `sft_height_target_all20.png`, and `event_stage_durations.png` summarize counts and trajectories.
- `sft_contact_sheet_seed42_51.jpg`, `sft_contact_sheet_seed52_61.jpg`, and `expert_contact_sheet_traj0_2.jpg` contain representative dual-camera frames. The complete original recordings remain under `outputs/stackcube_recap/milestone5_2_eval/videos/`.

The per-step SFT traces contain poses, TCP-to-cube distance, gripper command and finger joint positions, predicted and submitted actions, controller target positions, chunk index and observation-refresh flags, task success, native grasp, cube-on-cube, lift/transport markers, termination, and truncation. They do not contain finger contact forces/angles, actual arm joint positions, actor velocities/static flags, or per-step policy logits. Cube motion below is computed from consecutive recorded poses, not from logged simulator velocity.

## Per-episode reconstruction

`Grasp` means the native ManiSkill Panda grasp predicate became true at least once. `Lift` means red-cube height rose at least 2.5 cm above its initial height. `Transport` means red/green XY separation decreased by at least 3 cm from its initial value. These stage markers are diagnostics, not successful subtasks or substitutes for StackCube success.

| Seed | Stage reached | Native grasp steps | Lift first step | Transport first step | Min TCP–cube (m) | Max cube z (m) | Min cube–target XY (m) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | None | 0 | — | — | 0.179 | 0.020 | 0.143 |
| 43 | None | 0 | — | — | 0.044 | 0.020 | 0.166 |
| 44 | None | 0 | — | — | 0.083 | 0.020 | 0.241 |
| 45 | None | 0 | — | — | 0.143 | 0.020 | 0.187 |
| 46 | None | 0 | — | — | 0.088 | 0.020 | 0.276 |
| 47 | None | 0 | — | — | 0.126 | 0.020 | 0.271 |
| 48 | Grasp, no lift | 6 | — | — | 0.025 | 0.022 | 0.207 |
| 49 | None | 0 | — | — | 0.196 | 0.020 | 0.086 |
| 50 | Grasp, no lift | 7 | — | — | 0.017 | 0.024 | 0.179 |
| 51 | None | 0 | — | — | 0.034 | 0.020 | 0.088 |
| 52 | None | 0 | — | — | 0.040 | 0.022 | 0.115 |
| 53 | Lift, no transport | 32 | 123 | — | 0.005 | 0.133 | 0.187 |
| 54 | Grasp, no lift | 3 | — | — | 0.027 | 0.022 | 0.151 |
| 55 | None | 0 | — | — | 0.082 | 0.020 | 0.155 |
| 56 | None | 0 | — | — | 0.097 | 0.020 | 0.079 |
| 57 | None | 0 | — | — | 0.085 | 0.020 | 0.199 |
| 58 | Transport, no stack | 98 | 65 | 90 | 0.019 | 0.079 | 0.033 |
| 59 | Lift, no transport | 105 | 69 | — | 0.017 | 0.103 | 0.202 |
| 60 | Transport, no stack | 50 | 66 | 96 | 0.011 | 0.105 | 0.039 |
| 61 | None | 0 | — | — | 0.177 | 0.020 | 0.061 |

The table shows recorded extrema over a 150-step episode; a small TCP-to-cube-center distance alone does not establish finger alignment or contact. Of the 13 episodes without the native grasp flag, only seeds 43, 51, and 52 came within 5 cm TCP-to-cube-center distance. The other ten never came within 5 cm. Neither closeness nor the contact-sheet appearance is used to label a grasp.

## Grasp to lift

The native grasp signal is the ManiSkill Panda contact-based predicate, requiring sufficient contact force at both finger links and acceptable contact direction (as documented in the earlier evaluator audit). It is stronger evidence than visual proximity, but the trace does not retain the underlying force and angle values.

Three of seven grasp episodes did not cross the 2.5 cm lift marker:

- Seed 48 had two brief grasp runs (1 and 5 steps; six grasped steps total). The cube’s recorded height increased only about 2.4 mm during the longer run; its height peaked at 22.4 mm versus 20 mm initially.
- Seed 50 had one 7-step grasp run. The cube moved about 4.5 mm in x and 16.7 mm in y across that run, while its maximum height was 23.7 mm (3.7 mm above initial), then the native grasp flag was lost.
- Seed 54 had grasp runs of 1 and 2 steps, totaling 3 steps. The finger positions reached near-closed values, but the cube rose only about 2.1 mm above its initial height.

These observations verify transient contact without a meaningful lift. They do not prove whether the cause is lateral alignment, closure timing, contact geometry, or action selection: the necessary contact-force geometry and full arm joint states were not saved. “The gripper never closes” is contradicted by seed 54’s near-closed joint readings; “the grasp is stable” is contradicted by the short contact runs and absent lift. The three episodes support a grasp acquisition/retention weakness, with the precise mechanism unresolved.

## Lift to transport

Seeds 53 and 59 crossed the lift marker but not the transport marker.

- Seed 53 first crossed the lift threshold at step 123 and remained natively grasped through timeout (32 consecutive grasped steps). It raised the cube to 133.3 mm, but minimum target XY separation was 186.8 mm compared with an initial 194.0 mm: only 7.2 mm of targetward change, below the 30 mm transport marker. It ended holding the cube high rather than dropping it.
- Seed 59 remained natively grasped for 105 consecutive steps and reached 102.6 mm height. Minimum target XY separation was 202.3 mm versus 220.7 mm initially: 18.4 mm of targetward change, still below threshold. It did not drop the cube before timeout.

Thus these two transport-stage failures are not explained by a detected early drop. The traces show prolonged holding with little net targetward displacement. They support incomplete transport-directed policy behavior; whether this reflects learned action selection, a state-distribution issue, or another policy limitation cannot be isolated from these episodes.

## Transport to placement

ManiSkill’s StackCube positional condition requires the cube centers to be within about 33.3 mm in XY and the vertical center offset to be within 40 ± 5 mm; the task also requires the cube to be static and not grasped. The task’s native success evaluator remained active. Neither transport episode ever set `cubeA_on_cubeB`, and neither episode ever simultaneously met the recorded XY and vertical position tolerances. No task success occurred.

- **Seed 58 is closest by XY.** It stayed grasped for 98 consecutive steps, crossed transport at step 90, and ended with XY error 32.92 mm, just inside the XY tolerance. At that same final state its vertical center error relative to the required 40 mm offset was +10.61 mm, beyond the ±5 mm tolerance. It remained grasped through step 150, so it never completed release/settling. The closest XY state was also the final state; no valid stack-position state was reached.
- **Seed 60 opened and dropped early.** At step 101, the closest XY error was 38.65 mm and vertical error was +12.99 mm, both outside tolerance. At step 109, the gripper command changed from closing to opening (`+0.689`), native grasp became false, XY error was 41.8 mm, and vertical error was +14.7 mm. At step 110 the cube was at 48.5 mm center height; by step 113 it returned to the table height of 20 mm. This is direct evidence of release before placement alignment, followed by a fall. The red/green XY distance grew after release.

These two episodes fail differently: one retains the cube without lowering/releasing; the other releases before reaching the placement tolerances. The saved states identify these as the final obstacles for these episodes, but do not establish why the policy produced those actions.

## Expert reference comparison

All ten HDF5 expert trajectories contain terminal success; lengths range from 94 to 120 actions. Earlier Milestone 5.1 replay validated three successful expert replays in the current simulator (3/3). The expert action/pose reconstruction finds these event durations using comparable diagnostic thresholds:

| Event interval | Expert median (range, n=10) | SFT observed |
|---|---:|---:|
| First close command → 2.5 cm lift | 16 (15–16) | Not directly comparable: native contact → lift for 4 lifted episodes was 4, 7, 12, and 23 steps |
| Lift → 3 cm transport marker | 22 (19–24) | 23 steps in seed 58; 30 in seed 60; seeds 53 and 59 did not reach marker |
| Transport marker → first success transition | 16 (7–24) | No successes |

Expert close command is only an action-based proxy, not a logged native contact event. Expert and SFT episodes have different initial states, so these are event-aligned descriptive durations, not paired causal comparisons. Expert trajectories reach a median cube height near 0.118 m; SFT seeds 53 and 59 also lift to 0.133 m and 0.103 m but do not transport. In successful experts, the first positive/open command after closing occurs at or one action before the first logged success transition, consistent with release near a completed placement. SFT seed 58 has no release through timeout; seed 60 releases well before positional tolerance. The expert contact sheet for trajectories 0–2 corroborates approach, lift, transport, and stacking visually. Teacher-forced action agreement at recorded states, previously reported in Milestone 5.1, is not evidence of closed-loop competence.

## Action chunks, controller, and evaluator

The model returns `(1, 10, 7)` chunks. The evaluator executes ten actions with a fresh policy observation/prediction at each chunk boundary; it logs current simulator state each step. The trace audit found the expected refresh cadence in all 3,000 SFT transitions (300 refreshed policy observations), and predicted values equal values submitted to `env.step` on every transition. The action stream has 7 components and preserved the validated action representation.

The saved `executed_action` is the input sent to ManiSkill, before its controller’s designed preprocessing/clipping; equality with `predicted_action` does not mean the controller applied every raw value without clipping. The `pd_ee_delta_pose` controller clips normalized translation/rotation and gripper inputs to its configured range. Across SFT traces, no rotational vector norm exceeded 1; 1,968/3,000 gripper commands had magnitude greater than 1 and therefore saturated at the controller boundary. This is a frequent learned command pattern, but the range clipping is expected controller behavior rather than evidence of a controller defect. The controller target joint positions were recorded, but actual arm joint positions were not, so tracking error cannot be measured.

The release in seed 60 occurred at action index 8 within the chunk predicted from the step-101 observation (step 109 in one-based trace numbering); the next policy observation arrived at step 111. This makes open-loop execution a plausible contributor to late correction, not a verified defect. No matched 1-step versus 10-step execution experiment was run. The 150-step horizon exceeds all expert lengths, and the native task success/truncation checks were retained, so a too-short horizon or disabled success evaluator does not explain this result. No controller, action denormalization, or observation-mapping defect was found in the existing audit.

## Root-cause assessment and recommendations

| Priority | Finding | Evidence and confidence | Likely category | Follow-up |
|---:|---|---|---|---|
| 1 | SFT behavior does not reliably convert contact into lift, transport, and placement. | Verified episode outcomes: 7 grasp, 4 lift, 2 transport, 0 stack/success. High confidence for behavior; underlying cause unresolved. | Policy learning / limited demonstrations | If approved, improve trajectory coverage of stable grasps, sustained lift, targetward transport, lowering, and release; then retrain and evaluate separately. |
| 2 | Placement behavior is inconsistent: one episode never releases; another releases before alignment and drops. | Direct pose, gripper command, grasp, and cube-height trace for seeds 58 and 60. High confidence. | Policy action selection | Inspect placement-stage data and closed-loop state/action errors; retraining or additional demonstrations likely required. |
| 3 | Two long-held lifts show insufficient targetward movement. | Seeds 53/59: 7.2/18.4 mm net targetward change, below the 30 mm marker. High confidence descriptive. | Policy action selection / distribution shift | Study transport actions relative to target and expert, with comparable starting states. |
| 4 | Three native grasp episodes have brief contact and no meaningful lift. | Seeds 48/50/54: total contact 3–7 steps and peak rise 2.1–3.7 mm. High confidence for measured events; geometry/timing explanation is unverified. | Grasp policy / possibly data coverage | Capture finger contacts and actual joint trajectories in any future targeted diagnostic; do not infer a geometry fix from current logs. |
| 5 | Ten demonstrations may not cover enough closed-loop variation. | Dataset has ten successful trajectories; plausible but not isolated by this evaluation. Low-to-medium confidence. | Dataset/generalization | If approved, assess coverage before deciding on targeted expert data collection. |
| 6 | Ten-action open-loop chunks may delay correction around contact/release. | Seed 60 release at action 8/10, without an intervening policy observation; no schedule comparison. Hypothesis only. | Inference/control schedule | A small paired replay with shorter refresh intervals could test this, keeping checkpoint and success rules fixed. |

The action clipping observed is expected, and the evaluation uses the verified 150-step limit with native success checks. There is no evidence in these artifacts for a simulator, task-definition, or success-reporting defect. Do not claim the SFT policy is successful or that it improved task success over pretrained: both are 0/20, although the SFT policy shows more task-directed partial behavior.

## Execution, code, and limitations

This milestone used only saved traces, expert HDF5 states/actions, and recordings. No GPU, additional simulator rollout, policy inference, retraining, or model-weight modification was performed. The analysis script ran successfully and generated the tables/plots/contact sheets above. No project source code or tests were changed. Git remained on `branch-openpi-recap` at `eeb5f39d`, ahead 7; pre-existing dirty documentation files were preserved. No commit or push was made.

Remaining limitations: only 20 SFT seeds and 10 demonstrations are available; paired policies share reset states, but expert trajectories do not; underlying contact force/angle, actual arm joint positions, velocities/static-state components, and per-step policy observation tensors are absent from the saved traces. Therefore exact geometric contact error, arm tracking lag, dynamic settling, and the earliest internal policy divergence cannot be established. The trace analysis is reproducible but does not isolate a causal treatment effect.

**Recommended next milestone:** with approval, run a narrowly scoped diagnostic that adds per-finger contact, actual robot joint state, and cube velocity/static measurements to a small number of representative replays, then compare fixed-checkpoint 1-step and 10-step observation refresh on the same initial states. Do not collect autonomous rollouts, retrain, or start RECAP without approval.
