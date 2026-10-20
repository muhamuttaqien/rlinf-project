# OpenPI + RECAP: Panda StackCube-v1

This branch isolates the **Franka Panda / ManiSkill StackCube-v1** experiment from the existing state-based SAC-Flow work.

## Status

**Phase 0 — simulator and data-interface smoke test.** This is not an end-to-end RECAP training implementation yet. The random-action probes below are **not expert demonstrations** and must **never** be passed to RECAP as the successful SFT dataset. RGB tensors are inspected but are not yet serialized into LeRobot episodes.

## Environment check

From the repository root in the existing embodied virtual environment:

```bash
git switch branch-openpi-recap
source .venv/bin/activate
PYTHONPATH="$PWD" python scripts/smoke_test_stackcube.py --robot panda_wristcam --obs-mode rgb --steps 10
PYTHONPATH="$PWD" python scripts/probe_stackcube_recap.py --episodes 3 --max-steps 100
cat outputs/stackcube_recap_probe/schema.json
cat outputs/stackcube_recap_probe/summary.json
```

The probe checks Panda camera observations, the 7D-ish end-effector action interface (inspect actual dimension), reset/step, and task-success reporting. The output directory is local experiment data; exclude it from commits.

## Planned progression

1. **Expert data:** Obtain successful StackCube trajectories via a validated ManiSkill scripted motion planner or teleoperation. Verify all successes by environment info and video; do not use random probes as demonstrations.
2. **LeRobot conversion:** Export the **full synchronized RGB observations, proprioceptive state, actions, timestamps, task text, and per-episode outcomes**, with correct feature schema, metadata, and episode indexing. Validate replay and inspect sample videos.
3. **π₀.₅ SFT:** Map the Panda image/state/action keys to the selected OpenPI RLinf model adapter; decide the actual action horizon and normalization using real dataset statistics.
4. **Rollouts:** Evaluate SFT policy using different seeds; collect success and failure episodes, preserving terminal outcomes. Measure baseline success rate before RECAP.
5. **RECAP stage 1:** `bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh recap_compute_returns` with dataset and tag set in `examples/offline_rl/config/recap_compute_returns.yaml`.
6. **RECAP stage 2:** `bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh recap_value_model_sft` with a valid SigLIP2/Gemma3 setup and matching tag.
7. **RECAP stages 3–4:** Compute advantages with the value-model checkpoint, then train the π₀.₅ CFG policy using `recap_compute_advantages.yaml` and `cfg_rl_openpi.yaml`. Consult the commands in the upstream RLinf RECAP guide for the exact checkout.
8. **Evaluation:** Same seeds and episode budget as baseline; log success, trajectory length, checkpoints, and simulation videos to W&B.

Reference: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html

## Caveats

- `StackCube-v1` is a ManiSkill **simulation** with Panda; this is not real-world FR3 support.
- RECAP is **offline**. Simulator interactions happen during demonstration and rollout collection, not inside the four optimization stages.
- The existing examples are LIBERO-oriented; Panda-specific LeRobot feature mappings and checkpoint compatibility require verification before training.
- A successful smoke test is necessary but not sufficient proof that full RECAP works.
