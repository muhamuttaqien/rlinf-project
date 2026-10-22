# Milestone 10.1 — Easy StackCube: two cameras and EE delta control

## Final experiment specification

This is a **separate training setup**; it does not modify the old joint-space
or +/-2.5cm StackCube runs.

| Signal | Definition |
|---|---|
| Task | Stack the red cube on top of green cube and release |
| Language | "stack the red cube on top of the green cube and release it" |
| Robot | ManiSkill panda_wristcam |
| Images | Real base_camera + real hand_camera, both RGB 224x224 |
| Robot state (8D) | TCP **world** XYZ (3), TCP quaternion **WXYZ** (4), measured gripper opening width in meters (1) |
| Policy action (7D) | normalized EE translation delta (3), normalized Euler XYZ rotation delta (3), gripper command (1) |
| ManiSkill control mode | pd_ee_delta_pose |
| Cube XY positions | separate red and green spawn regions, independent **±0.010 m** variation around each center |
| Cube yaw | ±0.15 rad (same as the easy 2-camera setup) |
| Robot start | fixed joint initialization (noise 0) |
| Pretrained model | original pi05_base checkpoint, NOT old 8D joint-policy checkpoint |
| Planned demos | 100 *successful* end-effector-controller demonstrations |
| Planned SFT | 600 optimizer steps, new output directory |
| Evaluation | 10 predicted EE actions per inference by default, with optional `--action-chunk 1` replanning |

**EE orientation encoding:** the observation is quaternion WXYZ (4D).
The action Δorientation has 3 normalized Euler-XYZ components; the
ManiSkill controller converts them to a relative rotation and applies
its own inverse kinematics. They are *not* absolute quaternion targets.
The action channels are normalized controller commands, not raw
meters/radians, and cannot be interchanged with previous 8D joint targets.

**Important compatibility:** ManiSkill's stock motion-planning StackCube
expert requires pd_joint_pos. The new collector implements a new
feedback-driven demonstration policy controlling the actual 7D
pd_ee_delta_pose environment. It uses the upstream Panda grasp-pose
construction but **is not yet proven successful** on this workstation.
Only simulator-verified successful episodes are written. If the
collector cannot complete stacking, do not train; fix the expert.

## 1. Prepare environment (from RLinf repository root)

\`\`\`bash
git pull --ff-only origin branch-openpi-recap-chat
export HF_LEROBOT_HOME=/data1/Muttaqien/rlinf_data/lerobot
export STACKCUBE_DATASET=local/stackcube_easy_ee_delta_two_cam100
export STACKCUBE_RESULTS_ROOT=/data1/Muttaqien/rlinf_data/results
export STACKCUBE_NORM_STATS=/data1/Muttaqien/rlinf_data/pi05_stackcube_easy_ee_delta_two_cam_assets/local/stackcube_easy_ee_delta_two_cam100/norm_stats.json
export EMBODIED_PATH="$PWD/examples/sft"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$HF_LEROBOT_HOME" "$STACKCUBE_RESULTS_ROOT"
\`\`\`

\`PI05_MODEL_PATH\` must point to the **pretrained original** pi05_base
directory containing \`model.safetensors\`. Do not point it to the
previous step-200 joint SFT actor.

## 2. Verify implementation and expert compatibility

\`\`\`bash
PYTHONPATH="$PWD" python -m pytest \
  tests/unit_tests/test_stackcube_pi05_ee_delta.py \
  tests/unit_tests/test_stackcube_easy_two_cam.py -q
\`\`\`

**Acceptance gate: one real successful EE-controller demonstration.**
The collector refuses to replace any existing dataset, so use a
separate small test dataset name that will not be reused as training data.

\`\`\`bash
PYTHONPATH="$PWD" python toolkits/lerobot/collect_stackcube_ee_delta.py \
  --repo-id local/stackcube_easy_ee_delta_acceptance \
  --num-episodes 1 --max-attempts 20 --seed 0
\`\`\`

Proceed **only** if the terminal prints \`COLLECTION PASSED: 1 successful
7D EE delta episodes\`. If it fails, review the first relevant error
and the expert/controller behavior. Failure of this gate means
the setup is NOT ready for SFT.

## 3. Collect 100 successful new demonstrations

\`\`\`bash
PYTHONPATH="$PWD" python toolkits/lerobot/collect_stackcube_ee_delta.py \
  --repo-id "$STACKCUBE_DATASET" \
  --num-episodes 100 --max-attempts 500 --seed 0
\`\`\`

This produces a new LeRobot v3 dataset with two embedded RGB images,
8D measured EE+gripper state and 7D actual executed EE delta actions.
Only successful episodes are saved. Previous datasets are never
overwritten. Note that the collection process can require changes to
the feedback expert if its closed-loop grasp strategy does not work
well in your installed ManiSkill version.

Calculate **new** normalization statistics:

\`\`\`bash
PYTHONPATH="$PWD" python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_stackcube_panda_easy_ee_delta_two_cam_state \
  --repo-id "$STACKCUBE_DATASET" \
  --output-root /data1/Muttaqien/rlinf_data/pi05_stackcube_easy_ee_delta_two_cam_assets

test -f "$STACKCUBE_NORM_STATS"
test -f "$PI05_MODEL_PATH/model.safetensors"
\`\`\`

## 4. Fine-tune Pi0.5 on 100 expert episodes

Keep the previously working isolated single-GPU Ray settings:

\`\`\`bash
export RAY_ADDRESS=local
export RAY_TMPDIR=/data1/Muttaqien/rlinf_data/ray_stackcube_easy_ee_delta
export RLINF_RAY_DISABLE_DASHBOARD=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RLINF_RAY_LOCAL_NUM_GPUS=1
mkdir -p "$RAY_TMPDIR"

PYTHONPATH="$PWD" python examples/sft/train_vla_sft.py \
  --config-path "$PWD/examples/sft/config" \
  --config-name maniskill_stackcube_panda_sft_openpi_pi05_easy_ee_delta_two_cam
\`\`\`

This configuration is independent of the prior 8D joint-policy
experiments; it fine-tunes a 7D delta-action policy from the base model.
Start training only after the acceptance gate and 100 successful demos.

## 5. Evaluate the new checkpoint on held-out seeds

\`\`\`bash
PYTHONPATH="$PWD" python scripts/evaluate_stackcube_pi05_ee_delta.py \
  --checkpoint "$STACKCUBE_RESULTS_ROOT/maniskill_stackcube_easy_ee_delta_two_cam_pi05_state_sft/checkpoints/global_step_600/actor" \
  --episodes 3 --seed-start 10000 \
  --output "$STACKCUBE_RESULTS_ROOT/maniskill_stackcube_easy_ee_delta_two_cam_pi05_state_sft/evaluation/step_600"
\`\`\`

The evaluator uses the *exact same* easy environment, 2 cameras, 8D
state, normalized 7D action control and the new dataset's stats.
By default it executes the full 10-step predicted action chunk before
resampling the model, matching the training action horizon. Pass
`--action-chunk 1` only to deliberately compare one-action replanning.
The output gives per-seed grasp and StackCube success, a rollout
video for the first seed, and \`results.json\`.

## Interpretation

Keep one short success gate and the resulting success rate as the main
signals. Additional demonstrations or SFT steps may help if there are
valid expert trajectories, but a stronger backbone cannot repair a
wrong controller/action representation or unsuccessful training labels.

## Investigation of step-600 policy failure (2026-10-22)

The initial evaluator only executed **one** action from each predicted
10-action trajectory, then injected fresh random diffusion noise on the
next simulator step. This was an unplanned mismatch to the policy's
10-action training horizon. It is not proof of an OpenPI training bug,
but it means the original 0/3 result was **only** for one-action
replanning. The updated evaluator defaults to chunk 10 and prints the
initial 7D EE command for each seed. The previous grasp metric also
silently read an optional info field, which may be missing; the new
version queries Panda `agent.is_grasping(cubeA)` and reports `None` if
unknown. Confirm chunk-10 performance on the existing step-600 checkpoint
**before any further training**.
