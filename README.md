# RLinf + ManiSkill: Three-Cube Picking

Training a **Franka Panda** robot to grasp and lift a **red cube** while **green and blue cubes** act as distractors. Built on [RLinf](https://github.com/RLinf/RLinf) and [ManiSkill](https://maniskill.readthedocs.io/) using **SAC-Flow**. The project demonstrates reinforcement learning for target-specific robotic manipulation.

## Keyboard Teleoperation

ManiSkill-based keyboard teleoperation for controlling the Franka Panda in a three-cube environment, featuring an HD external view, wrist-mounted camera, and gripper control. Run:

```bash
PYTHONPATH="$PWD" python scripts/control_franka.py
```

**Controls:** `W/A/S/D/R/F` for movement, `I/J/K/L/U/O` for rotation, `Z/X` for wrist twist, `SPACE` for gripper, `C` to toggle wrist camera, and `Q` to reset.

## Demo

### Successful red-cube pick

<div align="center">
  <a href="docs/media/success.mp4">
    <img src="docs/media/success.gif" alt="Successful SAC-Flow red-cube picking" width="600" />
  </a>
  <p><em>Successful red-cube picking in ManiSkill. Click to watch the full MP4.</em></p>
</div>

[Watch the full successful rollout (MP4)](docs/media/success.mp4)

## Experiment

| Component | Configuration |
|---|---|
| Simulator | ManiSkill (SAPIEN / PhysX) |
| Robot | Franka Panda |
| Algorithm | RLinf SAC-Flow |
| Environment | `PickRedCube3-v1` |
| Target | Red cube; green and blue are distractors |
| Observations | Robot and object states |
| Training | 8,000 steps (~3 h 50 min) |
| GPU | NVIDIA RTX 4070 Ti SUPER |

**Scope:** The policy receives state information; this is **not yet vision-based color recognition**. The distractor locations are relatively constrained.

## Results

Independent evaluation of the **step-8000 SAC-Flow checkpoint** across five random seeds, with 100 episodes per seed (**500 episodes total**).

| **Evaluation** | **Success at least once** | **Success at episode end** |
|---|---:|---:|
| Seed 1000 (100 episodes) | 97% | 91% |
| Seed 2000 (100 episodes) | 100% | 94% |
| Seed 3000 (100 episodes) | 100% | 77% |
| Seed 4000 (100 episodes) | 94% | 84% |
| Seed 5000 (100 episodes) | 100% | 98% |
| **Overall (500 episodes)** | **98.2%** | **88.8%** |
| **Std. across seeds** | **2.68 pp** | **8.35 pp** |

The policy successfully grasped and lifted the target red cube in **98.2% of episodes**, while **88.8% maintained success until episode end**. Performance varies across random seeds, with seed 3000 exhibiting the lowest final success rate (77%).

These results are obtained in simulation using privileged state observations, rather than vision-based object recognition.

## Failure Analysis

Failure diagnostics from the 500 evaluation episodes reveal that **dropping the red cube after grasping or lifting** is the primary failure mode.

| **Failure category** | **Seed 1000** | **Seed 2000** | **Seed 3000** | **Seed 4000** | **Seed 5000** | **Total** |
|---|---:|---:|---:|---:|---:|---:|
| Red cube dropped | 9 | 6 | 23 | 14 | 2 | **54** |
| Missed red grasp | 0 | 0 | 0 | 2 | 0 | **2** |
| Wrong-cube grasp | 0 | 0 | 0 | 0 | 0 | **0** |

Of the 500 episodes, **56 (11.2%) did not satisfy the success condition at episode end**. The recorded failure categories indicate that maintaining a stable grasp is a greater challenge than initially reaching and lifting the target.

The absence of wrong-cube grasps suggests that the policy reliably distinguishes the target from distractors under the current state-based observation setup.

**Note:** Failure categories are based on the recorded diagnostic flags. They should not be interpreted as mutually exclusive unless confirmed by the evaluation implementation.

## Run

From the project directory, with dependencies installed:

```bash
source .venv/bin/activate
bash run_red_cube3.sh
```

To evaluate a saved checkpoint **without training**:

```bash
bash evaluate_red_cube3.sh
```

The current evaluation launcher points to the checkpoint from the October 15 experiment. Update its checkpoint path when evaluating another run.

Monitor training metrics with:

```bash
tensorboard --logdir logs --port 6006
```

## Key files

| File | Purpose |
|---|---|
| `custom_envs/pick_red_cube3.py` | Three-cube environment and reward |
| `run_red_cube3.sh` | Training launcher |
| `evaluate_red_cube3.sh` | Evaluation-only launcher |
| `summarize_red_cube_failures.py` | Failure analysis |
| `examples/embodiment/config/maniskill_sac_flow_red_cube3.yaml` | SAC-Flow configuration |
| `examples/embodiment/config/env/maniskill_pick_red_cube3.yaml` | ManiSkill environment configuration |

## Next steps

- Capture the actual dropped-cube failure on video.
- Evaluate with all three cube positions fully randomized.
- Extend from privileged state observations to RGB input.
