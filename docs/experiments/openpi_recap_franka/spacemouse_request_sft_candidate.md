# FR3 SpaceMouse request labels: independent offline SFT candidate

## Confirmed audit (2026-11-10)

The 100 raw FR3 teleoperation recordings have 21,134 observation
transitions, of which 15,366 carry complete 7D request candidates.
All 5,768 unknown gripper request labels form per-episode prefixes,
before the first button press. The 62 low-Twist-coverage transitions
overlap those unknown prefixes. NO missing gripper request is imputed.

The episode-disjoint, seeded 80/20 split is:

| Split | Original episodes | Complete 7D request rows | Contiguous 10-step windows |
|---|---:|---:|---:|
| Train | 80 | 12,288 | 11,568 |
| Validation | 20 | 3,078 | 2,898 |

**Action contract (candidate only):** Six time-averaged *normalized*
SpaceMouse input channels + last latched normalized gripper open request.
NOT a 7D metric TCP delta, NOT commanded physical velocity, NOT
controller-accepted torque, and NOT an executable robot policy.

The controller C++ inspected during the session used
\`new_position = position_ + desired_linear_position_update_\` and
\`new_orientation = orientation_ * desired_angular_position_update_quaternion_\`,
then IK and joint torque control. It computed these offsets from incoming
Twist with per-axis scaling factors **0.007 linear** and **0.06
angular**, not a direct conversion from a time-window average to a
per-frame 7D metric motion command. The controller reuses its latest
offset at each update. The actual deployed update rate, calibration,
frame alignment and latency have not been verified.

## Export two independent LeRobot datasets (offline research only)

Run from \`~/Desktop/AIST-Tokyo/gamma/rlinf-project\`:

\`\`\`bash
git pull --ff-only origin branch-openpi-recap-franka
python -m pytest -q tests/unit_tests/test_fr3_request_lerobot_export.py

python -m toolkits.realworld_check.fr3_request_lerobot_export \
  --candidate-root /data1/Muttaqien/fr3_peg_command_request_candidate_v1 \
  --quality /data1/Muttaqien/fr3_peg_command_request_quality_v1.json \
  --split train --validate-only

python -m toolkits.realworld_check.fr3_request_lerobot_export \
  --candidate-root /data1/Muttaqien/fr3_peg_command_request_candidate_v1 \
  --quality /data1/Muttaqien/fr3_peg_command_request_quality_v1.json \
  --split validation --validate-only
\`\`\`

After both preflights pass, export each split to its **new, separate**
location (will refuse overwrite):

\`\`\`bash
python -m toolkits.realworld_check.fr3_request_lerobot_export \
  --candidate-root /data1/Muttaqien/fr3_peg_command_request_candidate_v1 \
  --quality /data1/Muttaqien/fr3_peg_command_request_quality_v1.json \
  --split train \
  --output /data1/Muttaqien/fr3_peg_request_lerobot_train_v1 \
  --repo-id local/fr3_peg_request_train_v1

python -m toolkits.realworld_check.fr3_request_lerobot_export \
  --candidate-root /data1/Muttaqien/fr3_peg_command_request_candidate_v1 \
  --quality /data1/Muttaqien/fr3_peg_command_request_quality_v1.json \
  --split validation \
  --output /data1/Muttaqien/fr3_peg_request_lerobot_validation_v1 \
  --repo-id local/fr3_peg_request_validation_v1
\`\`\`

The exporter copies RGB images into separate LeRobot datasets, keeps
each original episode (or uninterrupted valid span) as an independent
LeRobot episode, preserves 19D candidate state and normalized 7D requests,
and marks terminal success **only** if the original episode was reviewed
successful and the exported span reaches its last available transition.
The incomplete gripper prefixes are completely excluded.

Outputs explicitly carry \`fr3_command_request_provenance.json\` with
\`sft_ready=false\`, \`recap_ready=false\`,
\`robot_execution_approved=false\`. These data are research-only.

## Still required before π₀.₅ SFT

1. Verify ROS input command and gripper bridge semantics, controller
   update rate, frame orientation, and command-age/exposure latency.
2. Implement a distinct *request-input* OpenPI data transform, stats
   computed **only from the training split**, compatible with 19D state,
   7D normalized requests and π₀.₅'s padded 32D action format.
3. Run OpenPI official loader tests for BOTH splits and ensure
   action windows never cross an episode boundary or truncated prefix.
4. Train offline SFT only after the above checks. No result should be
   interpreted as validated physical control or task-success prediction.

Never point the old \`pi05_fr3_peg_proxy_offline\` config at this export,
reuse \`fr3_peg_proxy_norm_v1\`, or deploy the proxy checkpoint.
RECAP additionally requires actual rollout/critic data and aligned
two-camera value-model inputs; no RECAP training is prepared here.
