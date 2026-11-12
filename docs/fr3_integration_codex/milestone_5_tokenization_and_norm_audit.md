# Milestone 5: offline tokenization and normalization audit

The user's local files include a StackCube Panda normalization JSON plus Gemma-3-270M and SigLIP2 tokenizers. **Neither tokenizer is the OpenPI PaliGemma tokenizer.** The StackCube normalization is **not** an FR3 normalization. Do not silently substitute or apply them.

Standard OpenPI uses **PaligemmaTokenizer** from the official asset `gs://big_vision/paligemma_tokenizer.model`. This check requires the file already cached at `~/.cache/openpi/big_vision/paligemma_tokenizer.model` (or the corresponding OPENPI_DATA_HOME location). It does not download files and fails clearly when absent.

## Run one offline workstation command

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
PYTHONPATH="$PWD" python -m toolkits.realworld_check.fr3_pi05_checkpoint_smoke \
  --raw "$HOME/fr3_demo_20261107_084244" \
  --checkpoint "outputs/stackcube_recap/full_sft/results/stackcube_panda_pi05_full_sft/checkpoints/global_step_1084/actor/model_state_dict/full_weights.pt" \
  --norm-stats "outputs/stackcube_recap/openpi_norm/stackcube_panda/norm_stats.json" \
  --prompt "pick up the object"
```

The script uses the **real PaliGemma tokenizer** (not a placeholder) if its cached asset exists, audits the simulator statistics **without applying them**, then reruns one GPU π0.5 numerical inference on an existing FR3 recording. Model checkpoint is simulation-trained and not approved for real-robot motion. Expected success: `input_text_tokenization_verified=true`, `pi05_forward_executed=true`, `sim_norm_audit.applied_to_inference=false`, and `robot_commands_sent=false`. If official tokenizer absent, the test stops before attempting model inference.

No ROS connections, robot controller actions, or realtime inference are involved. Actual FR3 normalization/calibration and physical action-adapter safety remain outstanding.
