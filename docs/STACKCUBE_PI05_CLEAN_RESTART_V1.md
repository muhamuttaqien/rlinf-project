# StackCube π₀.₅ — Clean Restart V1

## Goal

Restart ManiSkill StackCube π₀.₅ SFT from **fresh expert demonstrations**.
Keep the prior 100-demo dataset, its step-600 checkpoint and historical
evaluation results untouched.

This branch retains the working RLinf/OpenPI architecture and replaces only
the experiment identity and the known train-only image-cropping confound.
The new dataset stores successful collection seeds in an independent manifest.

## Fixed task contract

| Parameter | Definition |
| --- | --- |
| Robot | Panda wristcam in ManiSkill |
| Environment | StackCubeEasyEEDeltaTwoCam-v1; ±1 cm independent XY perturbations |
| Control | pd_ee_delta_pose, normalized controller actions |
| Camera | Base + wrist RGB, 224 × 224 |
| State | 8D world TCP XYZ + WXYZ quaternion + measured finger opening |
| Action | 7D normalized XYZ translation + XYZ rotation + gripper |
| Expert | Feedback expert, ManiSkill success required for every saved episode |
| New dataset | local/stackcube_pi05_clean_v1 |
| New experiment | maniskill_stackcube_pi05_clean_v1 |
| SFT start | Original pi05_base, NOT the failed step-600 checkpoint |
| Horizon | 10 |
| SFT image geometry | No training-only crop; train_image_augmentation=False |
| Initial SFT budget | 600 steps, same as prior for controlled comparison |

## One-time workstation setup

From the RLinf repository root:

~~~bash
git fetch origin
git switch --track origin/experiment/stackcube-pi05-restart-v1
export HF_LEROBOT_HOME=/data1/Muttaqien/rlinf_data/lerobot
export STACKCUBE_DATASET=local/stackcube_pi05_clean_v1
export STACKCUBE_RESULTS_ROOT=/data1/Muttaqien/rlinf_data/results
export STACKCUBE_NORM_STATS=/data1/Muttaqien/rlinf_data/pi05_stackcube_clean_v1_assets/local/stackcube_pi05_clean_v1/norm_stats.json
export EMBODIED_PATH="$PWD/examples/sft"
export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.590.48.01
# PI05_MODEL_PATH must point to the original pretrained base model.
test -f "$PI05_MODEL_PATH/model.safetensors"
~~~

## Gate 1: collect one successful expert demonstration

~~~bash
PYTHONPATH="$PWD" python toolkits/lerobot/collect_stackcube_ee_delta.py \
  --repo-id local/stackcube_pi05_clean_v1_acceptance \
  --num-episodes 1 --max-attempts 25 --seed 0

PYTHONPATH="$PWD" python scripts/audit_stackcube_pi05_clean_data.py \
  --repo-id local/stackcube_pi05_clean_v1_acceptance --episodes 1
~~~

Accept only if collection prints COLLECTION PASSED and the audit prints
DATASET AUDIT PASSED. and stores LeRobot metadata and
collection_manifest.json with an actual successful environment seed. The
collector neither overwrites existing data nor saves failed rollouts.

Before the full collection, confirm the controller action convention and the
stored observations/actions against the installed ManiSkill implementation;
do not assume the existing normalization/rotation comments are authoritative.

## Gate 2: collect 100 fresh successes and calculate fresh norm stats

~~~bash
PYTHONPATH="$PWD" python toolkits/lerobot/collect_stackcube_ee_delta.py \
  --repo-id "$STACKCUBE_DATASET" \
  --num-episodes 100 --max-attempts 500 --seed 100

PYTHONPATH="$PWD" python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_stackcube_panda_easy_ee_delta_two_cam_state \
  --repo-id "$STACKCUBE_DATASET" \
  --output-root /data1/Muttaqien/rlinf_data/pi05_stackcube_clean_v1_assets

PYTHONPATH="$PWD" python scripts/audit_stackcube_pi05_clean_data.py \
  --repo-id "$STACKCUBE_DATASET" --episodes 100
~~~

Acceptance: 100 success-verified episodes, valid 8D states and 7D actions,
two independent 224×224 RGB cameras, verified sample alignment and finite
normalization stats. Do not train if these checks fail.

## Gate 3: train from original π₀.₅ checkpoint

Only after Gate 2 passes:

~~~bash
export RAY_ADDRESS=local
export RAY_TMPDIR=/data1/Muttaqien/rlinf_data/ray_stackcube_pi05_clean_v1
export RLINF_RAY_DISABLE_DASHBOARD=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RLINF_RAY_LOCAL_NUM_GPUS=1
mkdir -p "$RAY_TMPDIR"

PYTHONPATH="$PWD" python examples/sft/train_vla_sft.py \
  --config-path "$PWD/examples/sft/config" \
  --config-name maniskill_stackcube_panda_sft_openpi_pi05_clean_v1
~~~

The new configuration uses the existing official OpenPI transform/loader
integration with an opt-in change to disable train-only image augmentation.
It preserves the pretrained architecture, original learning rate, action
dimensionality, padded 32D model action space and the 600-step budget.

## Gate 4: evaluate the new checkpoint

First perform offline teacher-forced comparison on saved expert observations,
including grasp and orientation phases. Then evaluate held-out simulator seeds
with an explicitly selected 10-action chunk. Report actual task success,
sustained grasp/lift and pose discrepancies, not only training loss.

The flow-matching loss is averaged across the 32-dimensional padded model
action representation. It must not be confused with accuracy on the seven
executable controller commands.

## Non-destructive restart policy

- Keep old datasets, stats, checkpoints and branch unchanged.
- Use new repo ID, assets path, log directory and branch.
- Do not change the controller, model architecture or all-zero padded-action
  loss without diagnosing actual runtime behavior.
- Prefer a small number of decisive checks to repeated smoke-test cascades.
