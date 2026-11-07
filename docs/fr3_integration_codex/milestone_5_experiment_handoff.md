# Milestone 5 — Experiment handoff (no real experiment performed)

**Purpose:** Before starting real FR3 tasks, prepare a trustworthy route from approved
controller-reported actions to training data, FR3-specific pi0.5 SFT, and a
safe policy/control handoff. Do **not** perform demonstrations, train on fake
targets, or activate robot policy control during this milestone.

## What exists now

1. **Observation recording:** the actual stationary FR3/Robotiq/base+wrist
   camera capture passed raw validation, and all 300 samples were projected to
   19D candidate states. These are **not task demonstrations**.
2. **Future data export:** the new \`fr3_labeled_episode_export.py\` writes
   LeRobot from an **additional genuine, reviewed** metric 7D controller-target
   sidecar; it refuses missing or raw SpaceMouse-only labels. Output contains
   two RGB streams, 19D states, 7D accepted action targets, done/outcome flags,
   and provenance. The future sidecar must be obtained after a supervised
   collection session with verified clock calibration and accepted-command
   semantics. The current pilot has no such sidecar.
3. **FR3 SFT/normalization staging:** \`pi05_franka_pnp\` is registered; its
   official OpenPI two-camera realworld input transform exists and offline
   numerical inference has already been demonstrated with *simulation*
   weights. The SFT YAML now requires FR3-specific norm assets. The existing
   OpenPI calculation tool has an explicit local \`--output-dir\` to avoid
   accidental statistics writes into the dataset path.
4. **Policy execution boundary:** the existing actual NUC arm controller
   expects normalized six-axis SpaceMouse Twist, **not** the proposed metric
   7D action. ROS dispatch remains intentionally prohibited. It is **not**
   correct to claim autonomous FR3 policy execution readiness.

## One focused offline verification now

Run on the GPU workstation repository (the tests use CPU only and cannot
contact ROS or control the robot):

\`\`\`bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit_tests/test_fr3_labeled_episode_export.py \
  tests/unit_tests/test_fr3_sft_handoff_preflight.py
\`\`\`

**Do not create dummy real-world labels to force the exporter to pass.**
These unit tests use isolated synthetic fixtures only. There is no honest
end-to-end physical/training dataset test until a real approved task episode
has been recorded.

## After an approved demonstration exists (Milestone 6 only)

The external labels JSON schema is
\`fr3_controller_accepted_training_labels_v1\` and it must contain:

- \`raw_manifest_sha256\` (SHA-256 of the raw episode's exact
  \`manifest.json\`), matching \`task\`, \`state_contract\`
  (\`fr3_robotiq_policy_contract_candidate_v2\`), the exact
  \`ACTION_FIELDS\` from \`fr3_policy_contract.py\`, and
  \`action_source=controller_accepted_metric_tcp_delta_and_robotiq_knuckle_target\`.
- \`episode_success\` boolean, \`human_reviewed=true\`, \`reviewer\` string and
  \`controller_telemetry_source\` identifying the genuine controller records.
- A \`verification\` object with the **five exact keys** in
  \`fr3_labeled_episode_export.REVIEW_KEYS\`, all true **only when supported
  by real engineering evidence**. These are human attestations, not an
  independent cryptographic or physical-safety certification.
- \`clock\`: measured
  \`nuc_minus_workstation_monotonic_ns\`, \`uncertainty_ns\`, and
  \`max_skew_ns\`, whose tolerance really applies to this capture.
- One per-frame \`records\` entry with
  \`frame_index\`, \`controller_accepted_ns\`, unique \`source_record_id\`,
  and the seven approved metric action values. No inferred pseudo-targets.
  Explicitly establish whether an accepted controller setpoint can genuinely
  produce the chosen 7D Cartesian-delta representation. Today's normalized
  Twist and IK joint targets **do not automatically satisfy it**.

Only when these data exist and have been audited:

\`\`\`bash
python -m toolkits.realworld_check.fr3_labeled_episode_export \
  --raw /path/to/approved_raw_episode \
  --labels /path/to/reviewed_accepted_targets.json \
  --output /path/to/new_fr3_lerobot_dataset \
  --repo-id local/fr3_approved_episode

python toolkits/lerobot/calculate_norm_stats.py \
  --config-name pi05_franka_pnp \
  --repo-id /path/to/new_fr3_lerobot_dataset \
  --output-dir /path/to/new_fr3_pi05_norm_assets

python -m toolkits.realworld_check.fr3_sft_handoff_preflight \
  --dataset /path/to/new_fr3_lerobot_dataset \
  --norm-stats /path/to/new_fr3_pi05_norm_assets/norm_stats.json \
  --model-path checkpoints/torch/pi05_base
\`\`\`

The last command must successfully read **one real batch from the official
OpenPI SFT dataloader**, not merely check file names. The SFT YAML's
\`data.train_data_paths\`, \`actor.model.model_path\`, and
\`actor.model.openpi_data.norm_stats_path\` then need actual verified paths
before launching a training job. **No training starts in this preflight.**

## Policy-to-controller physical readiness is a separate gate

Do not reinterpret seven metric channels as six normalized SpaceMouse axes.
A hardware-valid bridge must be *separately qualified* with an operator-approved
exclusive publisher/control lease, base/TCP/rotation and gripper command
semantics, bounded actions and frequency, staleness watchdog, command-loss
hold, controller activation state, physical E-stop, supervisor approval and
tested rollback. The existing contract's \`dispatch_to_ros\` must keep raising
\`PermissionError\` until an independently reviewed adapter is implemented.
Do not swap, restart, or activate the active arm controller as part of this
offline check. This missing integration prevents any declaration that
physical policy execution is ready.
