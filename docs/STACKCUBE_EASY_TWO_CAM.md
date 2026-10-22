# StackCube easy two-camera Pi0.5 SFT — Milestone 10

## Goal and constraints

Train a **new** Pi0.5 policy to stack the red 4-cm cube on the green one.
This baseline makes the task more repeatable without removing all variation:

- ManiSkill environment: `StackCubeEasyTwoCam-v1`. This inherits the
  original StackCube-v1 physics, success function, `pd_joint_pos` actions,
  and motion-planning expert interface.
- Robot: `panda_wristcam`. Actual `base_camera` and mounted `hand_camera`
  RGB, both requested at 224×224. **No duplicate or zero-padded wrist input**.
- Red cube A: (x,y) near (0.00, -0.10) m; green B near (0.00, +0.10) m,
  each with independent ±0.025 m random XY offsets and ±0.15 rad yaw.
  Regions remain nonoverlapping; robot initial qpos noise is 0.
- Observations: two RGB cameras, 9D Panda qpos, fixed language instruction.
  Actions: 8D (7 absolute Panda joint targets, 1 gripper command).
- Collect **100 successful new expert demonstrations**. These must be
  regenerated because the environment, robot camera and target distribution
  changed. Do **not** reuse the old 100-demo dataset, normalizer or checkpoint.
- Train Pi0.5 from the **original pretrained base** for 600 optimizer steps
  (new experiment); monitor loss and checkpoint. Evaluate with held-out seeds.

Two cameras are an **additional source of information**, not a guarantee that
behavioral cloning will work. A different camera configuration makes the
training and inference preprocessing a strict matching requirement.

## 1. Prepare lab workstation paths

Run from the GitHub branch's repository root on pca03395, with the existing
working GPU Python environment active:

```bash
git pull --ff-only origin branch-openpi-recap-chat
export HF_LEROBOT_HOME=/data1/Muttaqien/rlinf_data/lerobot
export STACKCUBE_DATASET=local/stackcube_easy_two_cam100
export STACKCUBE_RESULTS_ROOT=/data1/Muttaqien/rlinf_data/results
export STACKCUBE_NORM_STATS=/data1/Muttaqien/rlinf_data/pi05_stackcube_easy_two_cam_assets/local/stackcube_easy_two_cam100/norm_stats.json
export EMBODIED_PATH="$PWD/examples/sft"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$HF_LEROBOT_HOME" "$STACKCUBE_RESULTS_ROOT"
```

**Important:** set `PI05_MODEL_PATH` to your existing original pretrained
Pi0.5 **base** directory containing `model.safetensors`. Do not point it at
the failed SFT step-200 actor; do not overwrite any previous output.

## 2. Collect new expert demonstrations and normalization stats

```bash
PYTHONPATH="$PWD" python -m pytest   tests/unit_tests/test_stackcube_easy_two_cam.py -q

PYTHONPATH="$PWD" python toolkits/lerobot/collect_maniskill_stackcube_recap.py   --repo-id "$STACKCUBE_DATASET" --easy-two-cam   --num-episodes 100 --max-attempts 500 --seed 0
```

The collector rejects missing or identical wrist/base images, records
only successful solver episodes, and refuses to overwrite an existing
dataset. It collects trajectories using the same easy task as evaluation.
If fewer than 100 are successful, *stop* and inspect solver failures;
do not silently include failures or train on an incomplete dataset.

```bash
PYTHONPATH="$PWD" python toolkits/lerobot/calculate_norm_stats.py   --config-name pi05_stackcube_panda_easy_two_cam_state   --repo-id "$STACKCUBE_DATASET"   --output-root /data1/Muttaqien/rlinf_data/pi05_stackcube_easy_two_cam_assets

test -f "$STACKCUBE_NORM_STATS"
test -f "$PI05_MODEL_PATH/model.safetensors"
```

## 3. Start the new SFT run

Keep the previously successful isolated Ray environment settings used on
pca03395; avoid modifying shared Ray sessions. This example uses a separate
local Ray runtime and writes Ray temporary files to /data1:

```bash
export RAY_ADDRESS=local
export RAY_TMPDIR=/data1/Muttaqien/rlinf_data/ray_stackcube_easy_two_cam
export RLINF_RAY_DISABLE_DASHBOARD=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RLINF_RAY_LOCAL_NUM_GPUS=1
mkdir -p "$RAY_TMPDIR"

PYTHONPATH="$PWD" python examples/sft/train_vla_sft.py   --config-path "$PWD/examples/sft/config"   --config-name maniskill_stackcube_panda_sft_openpi_pi05_easy_two_cam
```

This new config uses two real images and discrete 9D state tokens,
`global_batch_size=16`, `micro_batch_size=1`, 600 steps, W&B and
TensorBoard logging. It writes results to `$STACKCUBE_RESULTS_ROOT`;
previous runs are unchanged.

## 4. Evaluate without retraining

After checkpoint 600 is written, use the new matching environment and
same two-camera observation path:

```bash
PYTHONPATH="$PWD" python scripts/evaluate_stackcube_pi05.py   --config examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_easy_two_cam.yaml   --checkpoint "$STACKCUBE_RESULTS_ROOT/maniskill_stackcube_easy_two_cam_pi05_state_sft/checkpoints/global_step_600/actor"   --norm-stats "$STACKCUBE_NORM_STATS"   --easy-two-cam --action-chunk 1   --episodes 3 --seed-start 10000 --max-videos 1   --label easy_two_cam_step600_replan1   --output "$STACKCUBE_RESULTS_ROOT/maniskill_stackcube_easy_two_cam_pi05_state_sft/evaluation/step_600/replan1"
```

Start with 3 held-out seeds, then expand to 20 after reviewing initial
success and rollout video. Training uses 10-step action targets; evaluation
replans after each executed action. Save the `RESULT` success line and
inspect the video before changing training settings.

## Reading the result

If the new 100-demo dataset and real two-camera SFT improve grasp and
stack success, expand evaluation to 20+ held-out randomized seeds. If it
still fails, test expert-action replay for the new environment and then
consider **shorter action-chunk targets**, additional expert data or a
pick-only curriculum. Do not add RECAP until the base imitation policy
performs nontrivial autonomous manipulation.
