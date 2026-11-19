# π₀.₅ Shutdown and SpaceMouse Execution Compatibility

This report explains the Episode 8 shutdown fix and the qualified 1.0 request path for the next separately approved rollout. It records the failure evidence, compares the policy interface with the installed SpaceMouse publisher, and lists the offline checks and operator preparation. [中文](pi05_episode8_shutdown.zh.md)

## Episode 8 and the Lost Acknowledgement

Episode 8 reached its time limit after 36 policy actions. The executor recorded one `episode_ended` event with 36 commands, 504 arm publications, and a maximum publish gap of 11,805,514 ns. The parent saved only the first 30 action/state records; the remaining six actions are present in `executor_events.jsonl`. Four observation files were saved, but there is no `outcome.json`.

The old `Executor.wait("chunk_done")` removed every event from a shared queue, including events it did not request. During a time-limit stop, it could remove `episode_ended`, discard it, then notice the stop flag and raise `RolloutStopped`. The following `wait("episode_ended")` then timed out despite the acknowledgement remaining visible in the audit log. A pipe-backed reproduction against the pre-fix code produced both original exceptions.

The revised waiter consumes only matching events and retains the terminal acknowledgement for repeated cleanup calls. Faults remain visible before a successful acknowledgement is returned. Sending a stop and revoking permission for new chunks share the same write lock. The parent sends `end_episode` once, and the executor emits its terminal acknowledgement once even if it receives duplicate stop messages. The ROS loop processes stop input before its next publication and clears the active schedule; no further motion or gripper goals follow a processed stop.

## Partial Results and Timeout Handling

The shutdown handshake now completes before the parent archives a stopped chunk's final action events. This lets it retain every executed action from a partial chunk rather than losing the last six as Episode 8 did. Command/state logs are flushed, and the full executor event stream is retained on exit.

If a chunk times out, the parent requests a stop. If shutdown acknowledgement times out or the executor faults/exits, it closes the command process before outcome review, preserves the received partial events, and records an unreviewed `aborted` outcome with the stop reason and error details. `shutdown_confirmed` reports whether an acknowledgement arrived; absent publication counters remain `null` rather than being estimated. A graceful time-limit or operator stop still requires an operator outcome label. EOF or interrupted outcome review leaves an unreviewed abort.

New outcome files are created exclusively and cannot overwrite an existing result. The registrar retains pending aborts, including a zero-action attempt with existing empty log files, without counting them toward RECAP. New manifests also record `execution_interface_qualified`: a reviewed legacy diagnostic result with this field set to `false` is recorded but does not count toward the 20 outcomes. Historical manifests retain their existing interpretation.

Episode 8's eight existing files were hashed before and after verification and remain unchanged. Its missing outcome and incomplete action logs were not reconstructed. The [evidence JSON](pi05_episode8_shutdown_evidence.json) contains the hashes, terminal event, SpaceMouse source identity, and offline test summary.

## SpaceMouse and Policy Requests

The installed SpaceMouse publisher matches its workspace source byte-for-byte (SHA-256 `2ed3e166966b31778264889301ef8e6dccba4c51a0ffed0ac0524fe46a801c87`). It publishes a normalized `Twist` every 0.01 s. Its HID mapping is `linear=(-y, x, z)` and `angular=(-roll, -pitch, -yaw)`, with the configured operator-position sign changes applied before publication.

The training alignment reads those published Twist fields and takes their time-weighted means over the 10 Hz observation windows. The FR3 OpenPI data transform preserves the six axis values and order, pads them for the model, and uses the training request normalization. The rollout inverse-normalizes actions once, then selects the seven request dimensions. It applies no second HID mapping, coordinate rotation, or SI conversion. The seventh dimension uses the separately documented [bounded gripper adaptation](pi05_gripper_validation.md).

| Property | Working SpaceMouse | Qualified π₀.₅ path |
| --- | --- | --- |
| Arm endpoint | `/franka_controller/target_cartesian_velocity_percent` | Same endpoint |
| Message and units | Normalized `Twist`, six request fields | Same fields and request units |
| Arm publication | 100 Hz | 100 Hz |
| Request multiplier | 1.0 | 1.0 in `identity_verified` |
| Axis order/signs | Published Twist axes after HID mapping | Decoded axes passed through unchanged |
| Action cadence | Current HID sample each timer tick | Trained 10 Hz samples, each held for its valid 100 ms slot |
| Gaps | Continuing device samples | Neutral refresh after the one-second chunk expires |
| Controller | Existing `JointImpedanceIKController` | Same controller and configuration |

Episode 8 used `legacy_limited`: its ready event reports gain 0.15, and every saved arm command equals the corresponding request multiplied by 0.15, with zero residual. That run did not test the unattenuated interface. The identity path already implements gain 1.0; this change retains it and identifies it as the intended qualified execution path. The default remains attenuated legacy diagnostics. Matching request units and publication timing does not prove equal trajectories: the policy predicts window averages and may choose different magnitudes from an operator.

## Offline Verification

The CPU checks run the real executor loop with fake ROS endpoints and the real parent IPC code with pipe-backed child processes. They load no checkpoint and activate no controller. All 259 component tests passed, including the Episode 8 race, terminal events arriving before stop handling, duplicate stops, partial chunks, missing acknowledgements, child faults/exits, preserved operator labels, and exclusive outcome creation.

The same suite verifies all twelve signed arm-axis cases at gain 1.0, record/fingerprint/consent gates, 100 Hz refreshes against distinct current 10 Hz samples, expiry during message construction, inference gaps, stale requests, neutral stop handling, watchdogs, and exclusive publishing. Run the checks from the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/unit_tests/test_fr3_pi05_ros_executor.py \
  tests/unit_tests/test_fr3_rollout_outcomes.py \
  tests/unit_tests/test_fr3_pi05_request_adapter.py \
  tests/unit_tests/test_fr3_pi05_deployment_audit.py

.venv/bin/ruff check --preview \
  toolkits/realworld_check/fr3_pi05_supervised_rollout.py \
  toolkits/realworld_check/fr3_pi05_ros_executor.py \
  toolkits/realworld_check/fr3_rollout_outcomes.py \
  tests/unit_tests/test_fr3_pi05_ros_executor.py \
  tests/unit_tests/test_fr3_rollout_outcomes.py

.venv/bin/ruff format --check \
  toolkits/realworld_check/fr3_pi05_supervised_rollout.py \
  toolkits/realworld_check/fr3_pi05_ros_executor.py \
  toolkits/realworld_check/fr3_rollout_outcomes.py \
  tests/unit_tests/test_fr3_pi05_ros_executor.py \
  tests/unit_tests/test_fr3_rollout_outcomes.py

git diff --check
```

Ruff and the diff check passed. Both language builds passed the repository docs harness with its default autodoc-import filtering. These Markdown reports sit outside the Sphinx source trees; their local links, commands, and English/Chinese parity were checked separately.

## Prepare One Separately Approved Rollout

Use `identity_verified` only after physical qualification. The existing record must cover all signed axes, controller acceptance, frames, 10 Hz/100 Hz timing, neutral stop, and gripper behavior, with hashed evidence and controller binary/configuration fingerprints. The operator must match the running installation to that record, stop the competing SpaceMouse publisher, clear the workspace, and supervise at the approved E-stop. This work grants no physical qualification and changes no verification record, controller setting, checkpoint, or safety limit.

The following is a launch template for that later operator-approved session; it was not executed. Set `PI05_CHECKPOINT` to the existing final `global_step_3000/actor/model_state_dict/full_weights.pt` and `OPERATOR_VERIFIED_MAPPING_JSON` to the actual qualified record:

```bash
PYTHONPATH=. .venv/bin/python -m toolkits.realworld_check.fr3_pi05_supervised_rollout \
  --checkpoint "$PI05_CHECKPOINT" \
  --root /data1/Muttaqien/fr3_pi05_real_rollouts_20 \
  --verified-request-mapping \
  --mapping-verification "$OPERATOR_VERIFIED_MAPPING_JSON"
```

The parent validates the record before model/hardware setup, and the child independently checks its digest. Per-episode consent remains `RUN VERIFIED <first 12 characters of record SHA-256>`. Existing attempt folders are skipped. The next episode may count toward RECAP only after the interface is qualified and the result is operator-reviewed. Work stops here after offline verification; whether the weak movement is resolved remains a physical test question.
