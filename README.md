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

### Failure: red cube dropped

The failure-tracking evaluation recorded **one episode where the red cube was grasped and lifted but was no longer grasped at episode end**. Its video was **not captured among the five available evaluation recordings**, so no failure clip is linked yet. We'll add a verified failure video when available, rather than showing an unrelated rollout.

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

Independent evaluation of the **step-8000 SAC-Flow checkpoint** across five random seeds, with 100 episodes per seed (500 episodes total).

| **Evaluation** | **Success at least once** | **Success at episode end** |
|---|---:|---:|
| Seed 1000 (100 episodes) | 100% | 90% |
| Seed 2000 (100 episodes) | 100% | 94% |
| Seed 3000 (100 episodes) | 100% | 74% |
| Seed 4000 (100 episodes) | 95% | 86% |
| Seed 5000 (100 episodes) | 100% | 98% |
| **Overall (500 episodes)** | **99.0%** | **88.4%** |
| **Std. across seeds** | **2.24 pp** | **9.21 pp** |

The policy successfully grasped and lifted the target red cube in **99.0% of episodes**, while **88.4% maintained success until episode end**. The difference highlights occasional failures to maintain a stable grasp or lift.

The variation across seeds, particularly the 74% final success rate for seed 3000, suggests that the policy's robustness depends on the sampled initial conditions.

**Evaluation setup:** ManiSkill simulation, 100 steps per episode, using the same trained checkpoint for all five seeds. These results use privileged state observations rather than RGB-based object recognition.

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
