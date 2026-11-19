# SpaceMouse request SFT — OpenPI offline preflight (NOT training)

This branch includes a **separate** OpenPI training contract named
\`pi05_fr3_peg_request_offline\`, independent of
\`pi05_fr3_peg_proxy_offline\`.

Request action 7D: [normalized Twist linear x/y/z,
normalized Twist angular x/y/z, binary gripper OPEN request].
No metric Cartesian pose delta, real robot target, executed torque, or
controller-accepted action is represented.

Recorded command averaging is a diagnostic approximation to the
high-frequency controller's held-input behavior. Source image receipts,
command latency, exact coordinate conventions, controller rates and
Robotiq bridge state/acknowledgement are **not yet verified**. This is
**offline research only**.

## 1. Generate train-only normalization (80 episodes; 12,288 rows)

\`\`\`bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
python -m pytest -q \
  tests/unit_tests/test_fr3_request_norm_stats.py \
  tests/unit_tests/test_fr3_peg_request_policy.py

python -m toolkits.realworld_check.fr3_request_norm_stats \
  --train-dataset /data1/Muttaqien/fr3_peg_request_lerobot_train_v1 \
  --output /data1/Muttaqien/fr3_peg_request_norm_v1
\`\`\`

Output:
\`/data1/Muttaqien/fr3_peg_request_norm_v1/norm_stats.json\`
and \`request_norm_provenance.json\`. This command **rejects**
validation provenance and refuses existing output.

Pi0.5 uses quantile stats. State 19D and request action 7D
are padded to model action_dim=32. Constant-value channels are
given nondegenerate normalization quantiles; inactive padded channels
have identity quantile ranges. Neither images nor validation data are
used for statistics.

## 2. Check both train and validation *official OpenPI* loader batches

\`\`\`bash
python -m toolkits.realworld_check.fr3_request_openpi_preflight \
  --train /data1/Muttaqien/fr3_peg_request_lerobot_train_v1 \
  --validation /data1/Muttaqien/fr3_peg_request_lerobot_validation_v1 \
  --norm-dir /data1/Muttaqien/fr3_peg_request_norm_v1 \
  --model-path /data1/Muttaqien/rlinf_data/checkpoints/torch/pi05_base_rlinf
\`\`\`

This checks two RGB views, model padding, stats, the 10x32 action
horizon and episode disjointness using the official RLinf/OpenPI
loader with *actual exported data*. No model training is performed.
A short failing traceback should be investigated before proceeding.

The exported LeRobot datasets have 80 train and 20 validation
episodes and 12,288 + 3,078 rows, with respectively 11,568 and
2,898 uninterrupted 10-step source windows.

**Caveat:** The default OpenPI loader can pad action windows at
episode ends. The audited number of uninterrupted windows is not
necessarily the number the generic loader will sample. Later
training/evaluation must explicitly decide whether to mask or exclude
boundary-padded windows, and confirm the selected loader's behavior.

## 3. Stop after preflight

This stage deliberately does NOT install a controller-action bridge
or run SFT. Before a meaningful offline SFT comparison, validate
gripper request vs measured knuckle and controller timing, and
confirm train/validation window handling. Never use the existing
proxy-action checkpoint or \`fr3_peg_proxy_norm_v1\` with these
request targets. A low imitation loss alone is not peg-insertion
success, and NO physical deployment is authorized.
