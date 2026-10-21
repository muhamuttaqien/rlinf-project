# Milestone 5.5 — Policy Success Reproducibility and Generalization

**Status: PASSED for the evaluation protocol.** Five repeated K=5 runs exactly reproduced the saved seed-60 success trajectory, and the paired unseen-seed experiment completed on 30 matched initial states for K=5 and K=10. Neither schedule succeeded on the new seeds: each achieved 0/30 native StackCube successes. This evaluates reproducibility and generalization; it does not establish policy superiority.

## Objective and Scope

This milestone checked whether the first K=5 success could be reproduced and whether K=5 generalized better than the original K=10 schedule. The same Milestone 4 SFT checkpoint, normalization, model output chunk of ten actions, ManiSkill task, robot, controller, 150-step horizon, and native success evaluation were used throughout. No weights, dataset, controller semantics, or success criteria changed.

The new paired seed set was 1000–1029. A scan of existing JSON artifacts under `outputs/stackcube_recap/` found no seed collisions; the set also excludes the known earlier evaluation seeds 42–61, pilot seeds 200–202, and replay seed 60. Existing demonstration metadata does not record simulator reset seeds, so overlap with demonstration collection cannot be ruled out from the artifacts available. The manifest was saved before evaluation at `outputs/stackcube_recap/milestone5_5_eval/generalization_seed_manifest.json`.

## Integrity and Evaluation Configuration

The repository was on `branch-openpi-recap` at `eeb5f39dd27334303bd1b061dba65fb545f5fce9` (seven commits ahead of origin). Pre-existing changes to the progress log and untracked prior milestone reports were preserved.

The final SFT checkpoint was `outputs/stackcube_recap/full_sft/results/stackcube_panda_pi05_full_sft/checkpoints/global_step_1084/actor/model_state_dict/full_weights.pt`, SHA-256 `1db71780bcad741053517000652f01bb8f8383e43f1b2dacb4da5ab9f02c09c3`. Normalization statistics were `outputs/stackcube_recap/openpi_norm/stackcube_panda/norm_stats.json`, SHA-256 `9740881e6e1ff87773a0247397826ace1d2935a79e668105ec3eee45170456c7`.

The copied evaluator is `outputs/stackcube_recap/milestone5_5_eval/evaluate_m55.py`. It uses ManiSkill 3.0.0b22, `StackCube-v1`, `panda_wristcam`, `pd_ee_delta_pose`, `physx_cpu`, base and wrist RGB, 18D qpos+qvel, and the native `StackCubeEnv.evaluate` success path. The Gym `TimeLimit` remained 150 steps. OpenPI generated `(1,10,7)` chunks; K selected the prefix executed before refreshing the observation and calling the model again. There was no persistent action queue. The runner checked success, termination, truncation, and horizon after every simulator transition.

The runs used Python 3.11.15, PyTorch 2.11.0+cu130, CUDA 13.0, ManiSkill 3.0.0b22, Gymnasium 0.29.1, NumPy 1.26.4, and SciPy 1.17.1. Physical GPU 1 was mapped to logical CUDA device 0 with `CUDA_VISIBLE_DEVICES=1`. The flow-matching generator seed was `episode_seed * 1000 + inference_call_index`. Repeats at a given schedule therefore replayed the same per-call sampling sequence. CUDA deterministic-algorithm mode was not enabled, so exact reproducibility is an observed property of these runs and environment, not a guarantee across other software or hardware. K=5 and K=10 have different inference-call schedules and therefore do not share identical sampled-noise streams at matching simulator steps.

The evaluations ran the copied `evaluate_m55.py` from the repository root. For example, the five replay commands used `--phase replay --episodes 1 --seed 60 --actions-per-inference 5`, with separate output directories `replays/replay_01` through `replays/replay_05`. The paired commands used `--phase generalization --episodes 30 --seed 1000 --actions-per-inference 5` and `--actions-per-inference 10`, writing to `generalization/k5` and `generalization/k10`. Each process used `PYTHONPATH=$PWD`, `LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib`, `CUDA_VISIBLE_DEVICES=1`, and a 1,800-second timeout for each 30-episode sweep. Full command logs are preserved under `outputs/stackcube_recap/milestone5_5_eval/logs/`.

## Seed-60 Reproducibility

The corrected Milestone 5.4 reference reports native success and termination at step 129, with 129 executed actions and 26 inference calls. All five new seed-60 replays also succeeded at step 129 with 26 calls. Each reset-state signature exactly matched the reference. Every recorded action, TCP pose, cube pose, gripper value, and task/termination flag matched the reference trace at every step; inference latency fields were excluded from this equality check.

| Run | Native success | Success step | Inference calls | Exact control-trace match |
|---|---:|---:|---:|---:|
| M5.4 corrected reference | Yes | 129 | 26 | Reference |
| Replay 1 | Yes | 129 | 26 | Yes |
| Replay 2 | Yes | 129 | 26 | Yes |
| Replay 3 | Yes | 129 | 26 | Yes |
| Replay 4 | Yes | 129 | 26 | Yes |
| Replay 5 | Yes | 129 | 26 | Yes |

The seed-60 initial cube positions were red cube `[-0.02366, -0.03821, 0.02000] m` and green target `[-0.04833, 0.04130, 0.02000] m`. Event markers in the reference and all matching traces were: native grasp at step 55; cube height at least 2.5 cm above its initial height at step 67; targetward XY-distance reduction of at least 3 cm at step 85; native cube-on-cube flag at step 120; first positive/open gripper command at step 125; native task success and termination at step 129. The lift and transport markers are geometric diagnostics, not native subtask completion conditions.

Replay runtime averaged 6.51 seconds per episode, excluding process/model startup, and peak CUDA memory was 6,808.2 MiB allocated and 6,968.0 MiB reserved. Traces, reset signatures, videos, summaries, and logs are under `outputs/stackcube_recap/milestone5_5_eval/replays/`.

## Unseen-Seed Results

All 30 initial-state signatures were identical between K=5 and K=10. Both schedules completed 30 episodes at the full 150-step horizon. Success means the native ManiSkill task success signal, not a stage marker.

| Schedule | Native successes | Success rate (exact 95% CI) | Contact / grasp flag | Lift-height marker | Transport-distance marker | Cube-on-cube flag | Mean inference calls | Mean episode runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K=5 | 0/30 | 0.0% (0.0–11.57%) | 4/30 | 2/30 | 3/30 | 0/30 | 30.0 | 6.99 s |
| K=10 | 0/30 | 0.0% (0.0–11.57%) | 6/30 | 3/30 | 5/30 | 1/30 | 15.0 | 4.32 s |

The exact two-sided McNemar test had zero discordant success pairs (K=5 only: 0; K=10 only: 0), giving p=1.0. With no successes under either schedule, this test carries no evidence of equivalence or superiority. The confidence intervals are also broad. K=5 used twice as many inference calls on average and took 2.67 seconds longer per episode in this run (about 62% longer than K=10). Per-call mean latency was 174.9 ms at K=5 and 172.6 ms at K=10. Average runtime includes simulator stepping but excludes model loading.

The stage counters require careful interpretation. Grasp uses the native contact indicator. Lift is a 2.5 cm cube-height threshold, transport is a 3 cm reduction in cube-to-target XY distance, and placement is the native cube-on-cube positional flag. These are not a sequence of validated manipulation subgoals: for example, a transport-distance marker can precede a grasp because contact or incidental motion changes the cube position. They must not be read as proof of a stable grasp or deliberate transport.

K=10 seed 1000 was the closest observed placement attempt: it reached the cube-on-cube flag at step 131, had a minimum horizontal cube-to-target distance of 31.5 mm, and first issued a positive/open command at step 130. Native success never became true, and the cube was back near table height at the final step. This is direct evidence of transient placement without a stable successful terminal state. Under K=5 on the same initial state, the cube reached the transport-distance marker at step 115, but minimum target distance was 51.8 mm and no cube-on-cube flag occurred. This single paired episode does not establish a schedule effect.

In the new evaluation, the dominant classification remained no grasp: 26/30 at K=5 and 24/30 at K=10. K=10 had more episodes with grasp, lift-height, transport-distance, and placement indicators, but no schedule achieved native success. The per-seed table records three schedule-dependent failure-stage differences (seeds 1000, 1006, and 1020); the remaining seeds shared the same coarse failure classification. These outcomes suggest isolated differences in partial behavior, not reliable generalization or a consistent K=5 advantage.

## Resource Use, Validation, and Artifacts

Physical GPU 1 was selected. Before evaluation it had 97,235 MiB free, 0% utilization, and no compute processes. Host-context PyTorch reported 2.11.0+cu130, CUDA 13.0, one visible RTX PRO 6000 Blackwell Max-Q device. The sandbox could not access CUDA/NVML, so the documented process-scoped library path and host-context execution were used. Between the K=5 and K=10 sweeps GPU 1 returned to 0% utilization with no compute processes and 97,235 MiB free. After both sweeps it showed 16 MiB used, 97,235 MiB free, and 0% utilization. Each run reported 6,808.2 MiB peak allocated and 6,968.0 MiB peak reserved.

The K=5 generalization process ran from 18:35:32 to 18:39:56 JST (about 4m23s); K=10 ran from 18:47:30 to 18:50:33 JST (about 3m03s). Per-episode runtime totals were 209.6s for K=5 and 129.6s for K=10. The five replay processes took approximately 5m07s wall time in total. The output tree occupies about 630 MiB; 113 GiB remained available on the filesystem after the sweeps.

Focused validation passed: evaluator and analysis scripts compiled with `py_compile`; the analysis verified five reference-identical replay traces, 30 matched paired seeds, and no reset-signature mismatches; `git diff --check` passed. The first replay invocation exited before model initialization because `PYTHONPATH` was missing; retrying with `PYTHONPATH=$PWD` succeeded. No simulator, model, or GPU failure remained. The sandbox-only post-run `nvidia-smi` probe could not access the host driver; a host-context check confirmed the device was released.

Artifacts are under `outputs/stackcube_recap/milestone5_5_eval/`:

- `replays/replay_01/` through `replay_05/`: per-step traces, initial state, metrics, and videos.
- `generalization/k5/` and `generalization/k10/`: all 60 per-step traces, videos, initial states, metrics, and run summaries.
- `analysis_summary.json`, `paired_per_seed.csv`, and `behavior_summary.json`: replay comparison, paired outcomes, confidence intervals, stage classifications, and selected trajectory measurements.
- `logs/`: preflight and full execution logs, including the initial failed invocation.
- `evaluate_m55.py`: isolated evaluator copy; `analyze_m55.py`: reproducible offline analysis.

Each generalization schedule has 30 episode records and 4,500 transition rows (150 per episode). The paired analysis checked the full reset-state signature—robot qpos/qvel and both cube poses—not just the integer seeds.

The only tracked-file edit for this milestone is the progress log; this report is new. The evaluator and analysis helpers are inside ignored output artifacts. Pre-existing untracked prior milestone reports remain untouched. No commit or push was made; Git remains at `eeb5f39d`.

## Limitations and Next Step

This follow-up is exploratory because K=5 was selected after the Milestone 5.4 results. The 30 paired seeds provide limited precision, both native success rates are zero, and different schedules consume different flow-matching noise streams. The stage metrics are heuristic and sometimes occur out of the intuitive grasp→lift→transport order. The successful seed-60 replay proves repeatability under this configuration, not broad policy competence.

Recommended next research step: inspect the K=10 seed-1000 transient placement and the recorded K-dependent grasp/transport traces before proposing a narrowly scoped data or policy intervention. Do not start additional training, autonomous rollout collection, or RECAP without approval.
