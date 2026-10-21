# Research Milestone 5 Report — ManiSkill StackCube Evaluation

## Status

**PASSED for the paired closed-loop evaluation.** Both policies ran in ManiSkill
StackCube-v1 on 10 matched seeds. Neither policy succeeded: the pretrained
checkpoint achieved 0/10 and the SFT checkpoint achieved 0/10. This is a
negative task-performance result, not a failed evaluation run. No improvement
in task success was observed.

## Starting State and Evaluation Scope

- Repository: `/home/muttaqien/Desktop/AIST-Tokyo/gamma/rlinf-project`
- Branch: `branch-openpi-recap`
- Starting commit: `eeb5f39d`; the starting worktree was clean.
- The actual M4 final checkpoint was verified at
  `outputs/stackcube_recap/full_sft/results/stackcube_panda_pi05_full_sft/checkpoints/global_step_1084/actor/model_state_dict/full_weights.pt`.
  It had already loaded through RLinf's OpenPI evaluation factory in M4.
- The official pretrained PyTorch weights exist at
  `checkpoints/torch/pi05_stackcube_panda_base/model.safetensors`; M4 records
  the upstream source as `gs://openpi-assets/checkpoints/pi05_base/params`.
- The supplied Milestone 5 message ended mid-sentence before its remaining
  episode/runtime and reporting constraints. After requesting the missing
  instructions, none arrived before execution. The evaluation was bounded to
  10 paired seeds (42–51), the native 50-step task limit, one environment, and
  one GPU.

The comparison used the official pretrained weights and the M4 SFT checkpoint
with the same OpenPI Pi0.5 architecture, StackCube prompt, normalization file,
environment resets, and inference settings. Both models loaded the existing
normalization statistics at
`outputs/stackcube_recap/openpi_norm/stackcube_panda/norm_stats.json`.

## Evaluation Setup

The LeRobot training representation was preserved: Panda qpos+qvel state
`(1,18)`, base and hand RGB cameras `(1,128,128,3)`, and
`pd_ee_delta_pose` actions `(1,10,7)`. The task prompt was “Stack the red cube
on the green cube”. Policy inference used RLinf's existing `get_model`,
`Pi0Eval.predict_action_batch`, and OpenPI transforms. The evaluator applied
each 10-action chunk open loop, then queried the policy again with the next
observation. Each environment was reset with the same seed for both models.

The generic RLinf evaluation wrapper does not expose the training's exact
qpos+qvel plus two-camera mapping for this dataset. The task-specific evaluator
therefore lives at
`outputs/stackcube_recap/milestone5_eval/evaluate_stackcube.py`; it calls the
existing RLinf model interface and the native Gymnasium ManiSkill task. The
simulator used `physx_cpu`, matching the expert replay backend, while rendering
and policy inference used the selected GPU. This kept physics on CPU and model
inference on one GPU.

The two policies each ran 10 episodes, seeds 42–51, at no more than 50 steps
per episode. Each episode used five policy calls, each returning a 10-action
chunk. The total was 20 episodes, 1,000 environment transitions, and 100
policy calls. Stage diagnostics used ManiSkill's grasp, on-cube, and success
signals; lift was recorded when the red cube rose at least 2.5 cm above its
reset height, and transport when its XY distance to the green cube decreased
by at least 3 cm from reset. These privileged signals were used only for
evaluation metrics, not policy input.

## GPU and Execution

The sandbox could not access the NVIDIA driver devices. In host context, the
process-scoped workaround
`LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib` allowed the
read-only preflight and CUDA check to pass. Immediately before evaluation,
physical GPU 1 had 97,235 MiB free of 97,887 MiB, 0% utilization, and no
compute processes. System memory had 473 GiB available and the filesystem had
114 GiB free. PyTorch was `2.11.0+cu130`, CUDA `13.0`.

The bounded command was:

```bash
timeout 3600s env \
  LD_LIBRARY_PATH=/tmp/rlinf_m31_nvml590:/opt/slurm/24.11.0/lib \
  CUDA_VISIBLE_DEVICES=1 HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets \
  MPLCONFIGDIR=/tmp/rlinf-mpl MPLBACKEND=Agg \
  PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 \
  .venv/bin/python outputs/stackcube_recap/milestone5_eval/evaluate_stackcube.py \
  --episodes 10 --seed 42 --max-steps 50 --action-chunk 10 --save-video \
  2>&1 | tee outputs/stackcube_recap/milestone5_eval/evaluation.log
```

The per-model peak CUDA memory was:

| Policy | Peak allocated | Peak reserved |
|---|---:|---:|
| Official pretrained | 6,808.2 MiB | 6,968.0 MiB |
| M4 StackCube SFT | 6,809.4 MiB | 6,958.0 MiB |

Evaluation setup metadata was written at 15:34:14 JST and the summary at
15:35:47 JST, about 1 minute 33 seconds apart after the environment was
created. A shell-level elapsed-time timer was not included. After evaluation,
GPU 1 returned to 97,235 MiB free, 0% utilization, and no compute processes;
system RAM had 473 GiB available and disk had 114 GiB free. No other process
was stopped and no system configuration was changed.

## Results

| Metric | Pretrained | M4 SFT |
|---|---:|---:|
| Episodes | 10 | 10 |
| Successes | 0 | 0 |
| Success rate | 0% | 0% |
| Episodes truncated at 50 steps | 10 | 10 |
| Episodes with grasp | 0 | 0 |
| Episodes with lift | 0 | 0 |
| Episodes with transport | 0 | 0 |
| Episodes placed on target | 0 | 0 |
| Mean normalized dense return | 2.0033 | 4.1842 |

Every episode for both policies was categorized as `no_grasp`. The red cube's
maximum height remained at approximately its initial 0.02 m, and its distance
to the target did not show the 3 cm reduction used for the transport marker.
All paired seeds had the same outcome, so the paired success-rate difference
was 0/10. The SFT checkpoint had a higher mean normalized dense return by
2.1808 in this sample, but it achieved no grasps or task successes. This
reward difference does not establish a task-performance improvement.

The evaluator saved both base-camera and wrist-camera frame sequences for all
20 episodes as compressed NPZ files. Spot-checked frames from pretrained seed
42 and SFT seed 50 show no completed grasp; the SFT robot moved toward the cube
in the inspected sample but did not lift it. These samples are illustrative,
not a substitute for the recorded episode metrics.

## Tests, Logs, and Git

- Evaluator `--help` and `py_compile` checks passed with the repository on
  `PYTHONPATH`. An initial help invocation without `PYTHONPATH` failed before
  environment creation with `ModuleNotFoundError: rlinf`; rerunning with the
  project path fixed it.
- Sandbox GPU preflight failed because NVIDIA device nodes were unavailable in
  the sandbox. Host preflight with the approved, process-scoped NVML workaround
  succeeded. Both outputs are retained in
  `outputs/stackcube_recap/full_sft/logs/m5_preflight_sandbox.log`.
- The paired evaluation command completed without model, environment, or
  non-finite action errors. The final post-run inventory showed the selected
  GPU idle.
- Detailed artifacts are under `outputs/stackcube_recap/milestone5_eval/`:
  `evaluation_config.json`, `evaluation.log`, `summary.json`,
  `pretrained_episodes.json`, `sft_episodes.json`, and `videos/`.
- `git diff --check` is run after finalizing the report and progress log. The
  evaluator and generated results remain in ignored output paths; the tracked
  changes are the progress log and this report.
- Starting commit was `eeb5f39d`. No Milestone 5 commit was created and none
  was pushed.

## Limitations and Next Milestone

This is a 10-seed paired pilot with the task's 50-step limit. Neither policy
grasped the red cube, so the experiment provides no evidence that SFT improves
StackCube task success. The higher SFT return is descriptive only. The result
suggests the policy action mapping, action cadence, and training setup merit
review before treating this checkpoint as a useful task policy; any follow-up
evaluation would need a separately approved scope.

Milestone 5 is complete. Autonomous rollout collection and RECAP training have
not started and require the user's explicit approval.
