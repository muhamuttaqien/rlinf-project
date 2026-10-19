# World-model experiments: Phase 1 — StackCube simulator

This branch is isolated from the state-based SAC-Flow red-cube experiments on `main`.

## Scope

Verify that ManiSkill can initialize and step the built-in `StackCube-v1`
environment with the Franka Panda. The smoke test uses zero actions and
**does not** train a policy, collect demonstrations, or run Wan/OpenSora.

## Run

From the repository root, in an environment with the project's ManiSkill
dependencies installed:

```bash
git fetch origin
git switch branch-world-model
python scripts/smoke_test_stackcube.py --robot panda --obs-mode state --steps 10
python scripts/smoke_test_stackcube.py --robot panda_wristcam --obs-mode rgb --steps 10
```

For ManiSkill's interactive teleoperation (requires GUI/display support):

```bash
python -m mani_skill.examples.teleoperation.interactive_panda -e StackCube-v1
```

The smoke test prints the action space, observation type, and step outcomes.
A successful run verifies simulator functionality, **not cube stacking**.
It has not been executed on the local user's GPU.

## Next stages

1. Confirm the local RGB camera output and action conventions.
2. Generate and evaluate StackCube demonstrations via ManiSkill motion planning.
3. Train/evaluate an RGB-conditioned policy baseline.
4. Collect action-conditioned visual trajectories for Wan and OpenSora adaptation.
5. Integrate imagined rollouts and policy optimization in RLinf.

Existing LIBERO world-model checkpoints should not be considered validated
for ManiSkill StackCube without retraining or adaptation.
