# RLinf + ManiSkill: Three-Cube Picking

Train a Franka Panda robot with **SAC-Flow** to pick and lift the **red cube** while two distractor cubes (green and blue) are present. This project extends [RLinf](https://github.com/RLinf/RLinf) with a custom [ManiSkill](https://maniskill.readthedocs.io/) environment.

## Experiment at a glance

| Component | Setup |
|---|---|
| Simulator | ManiSkill (SAPIEN / PhysX) |
| Robot | Franka Panda |
| Learning | RLinf SAC-Flow, state observations |
| Environment | `PickRedCube3-v1` |
| Task | Grasp and lift the red cube; avoid the other cubes |
| Training | 8,000 steps (~3 h 50 min) |
| GPU used | NVIDIA RTX 4070 Ti SUPER |

**Note:** The current policy uses object and robot **state observations**, not camera-based color recognition. Distractor placements are relatively constrained; performance on fully randomized arrangements has not yet been measured.

## Quick start

Use the existing project environment (with RLinf and ManiSkill installed):

```bash
cd ~/Desktop/AIST-Tokyo/muhammad-ts/gamma/rlinf-project
source .venv/bin/activate
```

Train the three-cube policy:

```bash
bash run_red_cube3.sh
```

The script validates the custom environment, synchronizes the observation dimension, and launches SAC-Flow. Training configuration: `examples/embodiment/config/maniskill_sac_flow_red_cube3.yaml`.

Evaluate the saved step-8000 policy **without retraining**:

```bash
bash evaluate_red_cube3.sh
```

The evaluation launcher currently references the checkpoint from the October 15 run. If you train again, update its `CKPT` path before evaluating a different checkpoint. Evaluation outputs (including videos and failure reports) are saved under `logs/evaluation-red-cube3-failures-seed2000/`.

For training curves:

```bash
tensorboard --logdir logs --port 6006
```

## Results

The trained policy was tested on two independent sets of 100 evaluation episodes each.

| Evaluation | Success at least once | Success at episode end |
|---|---:|---:|
| Seed 1000 | 95% | 90% |
| Seed 2000 | 100% | 99% |

The **seed-2000 failure-tracking run** reported:

| Failure category | Episodes (out of 100) |
|---|---:|
| Missed red grasp | 0 |
| Wrong-cube grasp | 0 |
| Red cube dropped | 1 |
| Failed lift | 0 |

The one unsuccessful final state involved a previously lifted red cube that was no longer grasped at episode end. These figures are simulation results under the current reset distribution, not evidence of real-robot performance.

## Policy rollout videos

These clips are from **evaluation of the saved step-8000 policy**, not random actions or training-time demonstrations. The red cube is the target; green and blue are distractors.

| Successful pick | Failure: red cube dropped |
|---|---|
| [![Successful red-cube pick](docs/media/success.gif)](docs/media/success.mp4) | [![Red-cube drop](docs/media/failure.gif)](docs/media/failure.mp4) |
| **Success:** red cube grasped and held up at episode end. [Watch MP4](docs/media/success.mp4) | **Failure:** red cube grasped and lifted, but grasp lost before episode end. [Watch MP4](docs/media/failure.mp4) |

The failure occurred **once in 100 seed-2000 evaluation episodes** (batch 1, episode 6 within the batch, zero-indexed). The clips above must be selected from their corresponding evaluation rollouts; do not substitute an arbitrary video. GIFs provide inline previews on GitHub, while the linked MP4s provide higher-quality playback.

To add the clips, run `bash scripts/prepare_readme_videos.sh /path/to/success.mp4 /path/to/failure.mp4` from the repository root. This copies the selected clips to `docs/media/` and generates the previews using `ffmpeg`. Commit the four media files alongside this README. Evaluation logs and all other videos remain excluded from Git.

## Important files

| Path | Purpose |
|---|---|
| `custom_envs/pick_red_cube3.py` | Custom three-cube task and reward |
| `run_red_cube3.sh` | Training launcher |
| `evaluate_red_cube3.sh` | Checkpoint evaluation launcher |
| `summarize_red_cube_failures.py` | Failure summary generation |
| `scripts/prepare_readme_videos.sh` | Copy two selected evaluation clips and generate inline previews |
| `examples/embodiment/config/maniskill_sac_flow_red_cube3.yaml` | SAC-Flow experiment configuration |
| `examples/embodiment/config/env/maniskill_pick_red_cube3.yaml` | Simulator configuration |

## Next steps

1. Inspect the failure video and confirm grasp/lift behavior visually.
2. Randomize all three cube positions and evaluate on unseen layouts.
3. Extend from privileged state inputs to RGB observations.

*Based on the project run and evaluations completed October 14–15, 2026.*
