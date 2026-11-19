# FR3 request-action label gate (before 3000-step SFT)

## Source audit

The 100 reviewed raw episodes contain *observed* FR3 states, two RGB
views, received SpaceMouse Twist messages and received gripper button
requests. They do **not** carry verified controller-accepted commands.

Existing `fr3_command_request_alignment.py` creates a normalized six-axis
zero-order-held request average from each pair of **camera receipt timestamps**
(t_i, t_i+1), with a last-received gripper button value at t_i+1. The
observation image/state corresponds to t_i. Camera receipt is not exposure
time, nor is the workstation timestamp the controller acceptance timestamp.
The operator's request is not a metric TCP increment or velocity command.

The prior gripper audit reported 100 per-episode request transitions;
medians were 0.0 rad with open request and approximately 0.73 rad with
close request. This is plausible but cannot prove command delivery timing
or controller acceptance.

Historical source evidence is **inconsistent** about angular pose-update
scaling: the saved controller patch uses 0.007 translation / **0.03**
rotation, whereas a later inspection document records **0.06** rotation.
Only the exact active NUC controller binary/source and parameters can settle
the actual semantics; there is no authorization to replace binaries.

## 1. Replay-alignment and lag audit (read-only)

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
python -m pytest -q tests/unit_tests/test_fr3_request_label_validation.py

python -m toolkits.realworld_check.fr3_request_label_validation \
  --candidate-root /data1/Muttaqien/fr3_peg_command_request_candidate_v1 \
  --verify-source > /data1/Muttaqien/fr3_request_label_validation_stdout.json
```

The audit reproduces every candidate row from the raw recordings and tests
identical action labels. It also reports per-axis Pearson associations
between recorded normalized requests and observed TCP motion with lags 0–5
frames. No local or remote ROS access, no robot action, no new dataset,
no training, and no camera image decoding is needed. Output may be lengthy;
inspect `source_event_reconstruction_passed` and
`dataset_integrity_passed`, counts, and `association_only`.

A weak correlation does not necessarily mean a wrong label, while a high
correlation is *not* proof of accepted commands: multiple SpaceMouse axes,
IK behavior, update cadence, frame mounting transform, camera exposure,
contact and delayed feedback all confound the result. The audit is limited
to the original capture and does not invent missing controller timestamps.

## 2. Deployment identity audit (read-only, no restart)

The existing controller interface audit can inspect command topic endpoints
and timeout through the NUC read-only ROS CLI:

```bash
python -m toolkits.realworld_check.fr3_command_interface_audit --ssh-host nuc-franka
```

If available, inspect which controller library is mapped in the *existing*
`ros2_control_node` process using the already committed status script:

```bash
ssh nuc-franka 'python3 -' < toolkits/realworld_check/fr3_live_recorder_status.py
```

A reported process path, ROS topic, configuration parameter or source patch
alone does not prove the active controller accepted the recorded commands.
Do not relaunch ros2_control, move the arm, or send gripper goals.

## Decision gate

- Event reconstruction must PASS on all 100 original episodes.
- Explain any implausible correlation or mismatched request/gripper signs;
  do not automatically assume axes are aligned one-to-one.
- Identify deployed controller scaling, mounting frame and update cadence.
- Determine whether real accepted-reference telemetry exists; if absent,
  note explicitly that this request-only baseline is exploratory, not an
  accepted-action imitation target.
- For real-robot use, separately establish time calibration, output semantics,
  independent safety limits, human supervision, and authorization.

The prepared 3000-step request SFT YAML remains **on hold** pending this
gate. Do not conflate passing the one-step smoke test with reliable labels.
