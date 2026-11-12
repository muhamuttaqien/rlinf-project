# Milestone 5 — Offline simulation checkpoint mathematical forward

This is an **offline-only computation** for the existing **StackCube Panda π0.5 SFT** weights (latest user-reported checkpoint: \`outputs/stackcube_recap/full_sft/results/stackcube_panda_pi05_full_sft/checkpoints/global_step_1084/actor/model_state_dict/full_weights.pt\`). The file is NOT a pretrained real FR3 policy. The separate Gemma-3-270M and SigLIP2 \`model.safetensors\` discovered by the user's \`find\` are component checkpoints, not full pi0.5 policies.

The script uses the existing 300-frame actual FR3 *stationary raw* episode, selects the first 2 RGB camera frames and candidate 19D FR3 state, feeds the existing \`RealworldInputs\` input transformer, then constructs the matching RLinf PyTorch Pi0 model, loads genuine \`full_weights.pt\` using the existing checkpoint helper, and computes **one actual π0.5 action prediction** on GPU with a single syntactically valid placeholder token.

**This is deliberately not a complete policy inference path**: no language tokenizer, no normalization stats from the simulation training dataset, no action-coordinate/frame validation, no live observation adapter and no NUC command transport. Output is inspected for shape and finiteness only; synthetic placeholder text and uncalibrated state mean it must not be interpreted as useful FR3 instructions. Simulator action semantics are not mapped onto FR3.

## Run on the GPU workstation (no ROS required)

\`\`\`bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
PYTHONPATH="$PWD" python -m toolkits.realworld_check.fr3_pi05_checkpoint_smoke \
  --raw "$HOME/fr3_demo_20261107_084244" \
  --checkpoint "outputs/stackcube_recap/full_sft/results/stackcube_panda_pi05_full_sft/checkpoints/global_step_1084/actor/model_state_dict/full_weights.pt"
\`\`\`

Expected: \`model_weight_load_success=true\`, \`pi05_forward_executed=true\`, \`raw_output_shape=[1,10,32]\`, \`decoded_first_seven.finite=true\`, \`robot_commands_sent=false\`.

The script requires CUDA. If loading fails because checkpoint backbone architecture or installed dependencies differ, report the exact error; do not force mismatched weights, download or auto-run another checkpoint, or silently claim success. \`--horizon\` only adjusts network horizon; it does not change checkpoint training semantics.

## Remaining before closing Milestone 5

- Real tokenizer and normalization assets appropriate to chosen policy/dataset; task-appropriate policy checkpoint.
- Policy inference from actual two-camera observation with full intended transform and latency checked.
- ROS2 action adapter with explicit FR3/Robotiq frame, units, limits, bounded rates, freshness and default-disabled control. Safety review is required before connecting outputs to physical robot.
