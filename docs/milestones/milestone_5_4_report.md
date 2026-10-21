# Milestone 5.4 — Controlled Action Chunking Experiment

## Status

**PASSED for the controlled evaluation and its limited conclusion.** The 80-episode matched comparison completed, and one K=5 episode reached native StackCube success. K=10, K=2, and K=1 each had 0/20 successes. The single K=5 success is promising but does not establish a general improvement or an optimal refresh interval.

The initial full-run evaluator had one stop-check placement defect: K=5 seed 60 terminated successfully at step 129, but the loop executed one additional queued action. The issue was fixed and the same seed was replayed. The corrected replay reproduced the same observations, actions, and states through step 129, reported native success and termination there, and executed no step 130. The corrected replay trace is used for the final K=5 episode length and trace analysis. The original full-run artifacts remain preserved.

## Design and protocol

The hypothesis was that shorter action execution intervals allow more frequent state feedback and improve closed-loop behavior. Each inference continued to generate the existing `(1,10,7)` action chunk. K selected how many leading actions were executed before observing the updated environment and requesting another prediction. No model, controller, action representation, normalization, camera transform, or task-success rule changed.

| Controlled element | Value |
|---|---|
| Policy | Milestone 4 final OpenPI π₀.₅ SFT checkpoint, `global_step_1084` |
| Environment / robot | ManiSkill `StackCube-v1`, `panda_wristcam` |
| Controller / actions | `pd_ee_delta_pose`, validated 7D action representation |
| Observations | Base and hand RGB cameras; 18D qpos+qvel state |
| Simulator / horizon | CPU PhysX, 150 steps, native task evaluation retained |
| Full paired seeds | 42–61, identical initial robot and cube state signatures across K |
| Pilot seeds | 200–202, separate from final results |
| K values | 10, 5, 2, 1 actions per inference |

K=10’s full results were reused from Milestone 5.2 because the checkpoint, norm stats, reset seeds, simulator, horizon, observation pipeline, and original ten-action execution semantics matched. The M5.4 K=10 pilot reproduced the prior pilot outcomes, and all pilot and full-run initial-state signatures matched their Milestone 5.2 counterparts. The original K=10 full trace recorded 15 inference calls per 150-step episode.

The OpenPI inference path has no cross-call action queue or temporal aggregation. It produces a new ten-action chunk at each call. The M5.4 runner takes only the first K rows, obtains the updated observation, and calls the same RLInf/OpenPI model path again. Simulator state is recorded every transition. The policy’s generator seed is `episode_seed * 1000 + inference_call_index`; trajectories are reproducible for each K, although different K schedules consume different sampled-noise streams.

## Implementation and checks

The isolated evaluator, regression tests, analysis, and validation scripts are stored under `outputs/stackcube_recap/milestone5_4_eval/`. The original M5.2 evaluator and all M5.2 outputs were left unchanged. The evaluator defaults to K=10, validates `1 <= K <= actual output chunk length`, records inference calls and latency, and stops the current chunk immediately on native success, termination, truncation, or horizon.

Four focused tests passed:

- K=1, 2, 5, and 10 select exactly the first K actions and preserve their order and values.
- K=10 returns the complete original ten-action chunk.
- Invalid counts and action shapes are rejected.
- Success, termination, truncation, and horizon conditions signal immediate stop.

Python compilation passed for the evaluator and analysis scripts. The post-run trace validator passed action-dimension/finiteness checks, action equality, step sequence, inference-call counts, refresh cadence, and reset-signature checks for all canonical episodes. The analysis run confirmed 20 episodes per K and native task flags against the episode summaries. `git diff --check` passed.

The validator found the stop-check defect in the original K=5 seed-60 trace: native task success and termination were true at step 129, followed by one action at step 130. No other full-sweep episode had success or early termination before its final row. After moving the break into the per-action loop, a same-seed K=5 replay stopped at step 129 (`task_success=true`, `terminated=true`, `truncated=false`). Its first 129 actions and TCP/cube states exactly matched the original run’s prefix. The corrected trace is at `termination_fix_k5_seed60/traces/replay_sft_seed_60.jsonl`; the original full trace remains in `sweep_k5/`.

## Results

| K | Episodes | Native successes | Success rate | Grasp | Lift | Transport | Cube-on-cube observed | Mean steps | Mean inference calls / episode | Mean episode runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 20 | 0 | 0% | 7/20 | 4/20 | 2/20 | 0/20 | 150.00 | 15.0 | 4.41 s |
| 5 | 20 | 1 | 5% | 7/20 | 4/20 | 2/20 | 1/20 | 148.95 | 29.8 | 7.13 s |
| 2 | 20 | 0 | 0% | 6/20 | 3/20 | 3/20 | 1/20 | 150.00 | 75.0 | 14.58 s |
| 1 | 20 | 0 | 0% | 7/20 | 4/20 | 3/20 | 150.00 | 150.0 | 29.09 s |

The stage columns are diagnostic indicators: native ManiSkill grasp contact, cube height at least 2.5 cm above its initial height, targetward XY separation reduction of at least 3 cm, and the native `cubeA_on_cubeB` positional flag at least once. They are not task success. Native success requires the full StackCube success condition, including a stable, ungrasped placement.

### Paired comparisons

K=5’s only native success was seed 60; the same initial state failed at K=10, K=2, and K=1. Relative to K=10, its paired success result was one K=5-only success, zero K=10-only successes, and 19 ties. The exact two-sided paired sign/McNemar test for this single discordant pair gives p=1.0. This sample provides no statistically persuasive evidence of a general success-rate improvement.

Exact binomial 95% intervals are wide: 1/20 corresponds to approximately 0.13–24.87%, while 0/20 has an upper bound of about 16.84%. The 20-seed evaluation cannot tightly estimate small success-rate differences.

Paired stage differences versus K=10 were similarly mixed:

| K | Grasp: K-only / K10-only | Lift: K-only / K10-only | Transport: K-only / K10-only | Placement flag: K-only / K10-only |
|---:|---:|---:|---:|---:|
| 5 | 3 / 3 | 2 / 2 | 1 / 1 | 1 / 0 |
| 2 | 2 / 3 | 1 / 2 | 2 / 1 | 1 / 0 |
| 1 | 3 / 3 | 2 / 2 | 2 / 1 | 0 / 0 |

K=2 seed 52 briefly reached the native cube-on-cube positional flag at steps 127–129 but never reached task success; the cube was no longer stacked at timeout. This is partial placement progress only.

### Behavior and action analysis

K=5 seed 60 is the only successful episode. It reached native grasp at step 55, the 2.5 cm lift marker at step 67, and the transport marker at step 85. The cube-on-cube flag became true at step 120; the policy first issued an opening command at step 125, and native StackCube success/termination occurred at step 129. The corrected dual-camera recording and exact per-step trace are saved under `termination_fix_k5_seed60/`.

The action and positioning summaries do not show a consistent broad improvement as K decreases. Mean per-episode minimum XY target distance was 153.3 mm at K=10, 153.6 mm at K=5, 152.9 mm at K=2, and 152.2 mm at K=1. Median paired differences versus K=10 were zero at the recorded precision for all three alternatives. The mean of per-episode translation-action norms decreased from 0.0835 (K=10) to 0.0692 (K=5), 0.0563 (K=2), and 0.0503 (K=1). Mean cosine between the commanded translation vector and the horizontal target direction was -0.042, -0.068, -0.075, and -0.062, respectively. These are descriptive command-space comparisons and do not establish a more effective trajectory.

First positive gripper command after a native grasp was recorded in 3/20 K=10 episodes (median step 109), 5/20 K=5 (median step 78), 5/20 K=2 (median step 131), and 6/20 K=1 (median step 111.5). Only seed 60 at K=5 paired a release with native success. Frequent replanning did not consistently correct the earlier lift, transport, or release failure patterns.

### Inference overhead and GPU use

The final K=10 baseline used 300 inference calls across 20 episodes. The K=5 run used 596 calls (one successful episode ended at step 129); K=2 used 1,500; K=1 used 3,000. Mean measured inference latency per call was 180.4 ms (K=5), 170.9 ms (K=2), and 181.4 ms (K=1). K=10’s 20-episode M5.2 trace did not store per-call latency; its separate M5.4 pilot measured a 189.2 ms mean.

Mean episode runtime increased from 4.41 s at K=10 to 7.13 s at K=5, 14.58 s at K=2, and 29.09 s at K=1. Accumulated episode runtime for the 80 canonical episodes was approximately 18.4 minutes, excluding model-load and setup time; the resource-watch timestamps span about 19 minutes for the new full sweep. The single corrected replay took 6.50 s after load.

Physical GPU 1 was used as CUDA logical device 0 with `CUDA_VISIBLE_DEVICES=1` and the verified process-scoped NVML path `/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib`. PyTorch peak allocated/reserved memory was 6,808–6,809 MiB / 6,958–6,968 MiB across the configurations. The 15-second host monitor observed at most 8,014 MiB used and 41% utilization; GPU memory had 97,235 MiB free and 0% utilization with no compute processes after the runs. Initial preflight showed 472 GiB available system RAM and 114 GiB free disk; postflight showed the same GPU availability and approximately 114 GiB free disk. No other processes were terminated.

## Commands and artifacts

Representative pilot command (repeated for K=10, 5, 2, and 1):

```bash
LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib \
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD:${PYTHONPATH:-}" \
MPLCONFIGDIR=/tmp/rlinf-mpl MPLBACKEND=Agg \
timeout 1200s .venv/bin/python \
  outputs/stackcube_recap/milestone5_4_eval/evaluate_chunking.py \
  --phase pilot --episodes 3 --seed 200 \
  --actions-per-inference K --output-dir outputs/stackcube_recap/milestone5_4_eval/pilot_kK \
  --save-video
```

The full sweep used the same environment and checkpoint, 20 episodes with seeds 42–61, and a 3,600-second timeout per configuration for K=5, 2, and 1. K=10 used the compatible M5.2 paired SFT baseline. The corrected replay command was the same evaluator with `--phase replay --episodes 1 --seed 60 --actions-per-inference 5 --output-dir outputs/stackcube_recap/milestone5_4_eval/termination_fix_k5_seed60`.

All new experimental artifacts are under `outputs/stackcube_recap/milestone5_4_eval/`:

- `pilot_k*/`, `sweep_k*/`, and `termination_fix_k5_seed60/` hold per-step traces, episode metrics, reset signatures, and dual-camera NPZ videos.
- `logs/` contains per-configuration stdout, GPU/disk sampling, postflight, and corrected replay logs; `preflight_gpu.log` records initial resources.
- `aggregate_metrics.json`, `per_seed_metrics.csv`, and `paired_seed_comparison.csv` contain the aggregate and paired results.
- `chunking_comparison.png` summarizes task success, stage markers, and runtime.
- `k5_seed60_success_frames.jpg` contains the inspected success sequence from the original and corrected recordings.
- `experiment_manifest.json`, `trace_validation.json`, `analyze_chunking.py`, `validate_chunking.py`, and `tests/test_action_chunking.py` preserve provenance and analysis steps.

## Repository state and limitations

The work remained on `branch-openpi-recap`, HEAD `eeb5f39d`. No SFT checkpoint, dataset, normalization file, or prior evaluation was modified. New code and test utilities live under the ignored experiment output directory; no tracked project source code changed. `docs/CODEX_EXPERIMENT_PROGRESS.md` and `docs/milestones/milestone_5_4_report.md` were updated/created. The pre-existing modified progress file and untracked M5/M5.1/M5.2/M5.3 reports were preserved. No commit or push was made.

The policy uses stochastic flow-matching inference; the generator is deterministically seeded by episode seed and inference-call index. Since K changes the number and timing of inference calls, the random draws are reproducible within each K but are not common-random-number matched at the same simulator time across K. The paired initial simulator states are exact, but sampled action noise is not held equal across refresh schedules. Combined with only 20 seeds, this limits causal precision. K=10 per-call latency was not logged in its reused full baseline. The simulator evaluates cube motion using its native success state; no real-robot claim is made.

**Conclusion:** K=5 achieved one verified success and no other K did, while the overall grasp/lift/transport counts and target distances were similar. The result supports testing K=5 again on a larger, predeclared seed set, but does not justify calling it superior or optimal. Further evaluation, training, autonomous rollout collection, and RECAP remain unstarted pending approval.
