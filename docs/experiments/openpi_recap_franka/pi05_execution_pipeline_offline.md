# π₀.₅ Execution Pipeline: Offline Verification

This report explains the bounded command refresh implemented on 2026-11-18,
its completed mapping gate, offline timing evidence, and the checks required before one separately
operator-approved real-world test. Read the timing result first, then the
request-unit comparison and validation limits. [中文](pi05_execution_pipeline_offline.zh.md)

## Timing And Command Ownership

The verified gap comes from serial execution in
`fr3_pi05_supervised_rollout.py`: wait for `chunk_done`, acquire cameras and
states, transform observations, run inference, inverse-normalize, validate,
save the observation, then submit the next chunk. The old ROS executor
published ten actions at 100 ms intervals, sent one neutral at the one-second
boundary, then waited. Its stdin `readline()` could also block on a partial
message. That partial-message case is a verified blocking path, not evidence
that it caused the recorded gaps.

The executor process now owns a separate 10 ms publication schedule. A chunk
still contains exactly ten policy samples. Sample `i` is valid only in its
100 ms slot from chunk admission; refreshing it does not advance the policy,
produce another prediction, or advance the gripper confirmation filter.
After the one-second horizon, the executor refreshes **neutral** while waiting
for new inference. It never fills the gap with the last motion request.

Input reads are bounded to 64 KiB and do not wait for a complete line.
Stop/quit/parent EOF are processed before motion publication. A scheduling or
arm publication gap above 50 ms faults before resuming motion; missed slots
are never replayed in a burst. A final pre-publish expiry check rejects a
sample that expired during message preparation. The arm publisher keeps only
one queued request. These are application-side bounds, not hard real-time or
end-to-end DDS delivery guarantees.

The parent watches operator input/EOF and the episode deadline independently
of inference. A stop revokes later chunks and asks the executor to cancel the
active chunk. After the executor processes stop or fault, only neutral cleanup
requests remain; gripper goals and motion requests cease. Existing in-flight
gripper goals are not canceled by an arm neutral. The existing hardware E-stop
requirement remains in force.

## Full-Strength Gate And Latency Completion

`identity_verified` now sends `decoded_policy_action[:6]` unchanged, with the
original six-axis order and signs. The default `legacy_limited` still sends
those values times 0.15. All twelve signed-axis cases are exercised through
the real executor loop with fake ROS endpoints in both modes. The existing
100 Hz scheduler is retained.

Full-strength execution requires `--verified-request-mapping` together with
`--mapping-verification <record.json>`. The parent validates the record before
model or hardware setup, then requires the per-episode console challenge
`RUN VERIFIED <first 12 characters of record SHA-256>`. The child independently
validates the record and its digest before importing/initializing ROS, and
checks that digest again at `begin_episode`. Setting the mode environment
variable alone cannot enable full strength. This task creates no physical
approval record; synthetic test attestations exist only in temporary fixtures.

The verification record has the following required contract:

| Fields | Required evidence |
| --- | --- |
| `schema`, `mapping_mode` | `fr3_pi05_mapping_verification_v1`, `identity_verified` |
| `arm_axes` | The six `ARM_AXES` names in their original order |
| `policy_period_ns`, `publish_period_ns` | `100000000`, `10000000` |
| `operator`, `verified_at_utc` | Named reviewer and an actual past UTC qualification time |
| `controller_binary_sha256`, `controller_config_sha256` | Lowercase SHA-256 fingerprints of the physically qualified installation |
| `base_frame`, `tool_frame` | Physically reviewed base and calibrated tool/TCP frames |
| `checks` | Each of the 17 checks below has `passed: true`, `evidence_path`, and `evidence_sha256` |

The checks are each axis name suffixed with `_positive` and `_negative`
(twelve checks), plus `controller_acceptance`, `tool_frame_alignment`,
`timing_10hz_100hz`, `neutral_stop`, and `gripper_open_close`. Evidence files
must exist, be nonempty, and match their SHA-256 digests. Relative paths resolve
against the record directory. One synchronized trace can support several
checks if the operator has reviewed each claim. Hashes establish integrity;
they cannot authenticate physical measurements or inspect a live controller.
The operator must match the current running binary/configuration and frames
to the record before giving per-run consent. A changed installation requires
new qualification. The approved record path/digest is retained in the manifest.

A measured software bottleneck was observation compression before submission.
Ten offline writes of Episode 6's first saved observation (480×640 and 480×848
RGB, plus state19) took median **125.001 ms**, p95 **125.900 ms**, maximum
**126.417 ms** on workstation CPU. Formatting/writing/flushing ten saved
command/state pairs took median **0.536 ms**, maximum **0.682 ms**. All benchmark
writes went to temporary directories. These measurements isolate disk/CPU
work; they do not measure live acquisition or GPU inference.

The parent now submits validated actions before compressing the unchanged
observation, so archival overlaps the one-second command window. An archive
failure immediately revokes active/future chunks and preserves partial
executor events. Initial requests can precede an archive failure; such a
faulted attempt is not a usable outcome. Stop remains independent during a
blocked archive. `pipeline_timing.jsonl` records acquisition, inference,
submission, archive completion and wait-return timestamps for the next approved
qualification. Small command/state log writes remain synchronous.

Acquisition and inference still follow completion of the preceding chunk.
Predicting the next chunk from an earlier observation would shift the trained
observation/action alignment. No such prefetch, altered sampling steps, or
invented interpolation is introduced. Their live cost is still unmeasured;
neutral refresh safely covers that remaining wait.

## Before And After Evidence

The historical comparison reads all 100
`/data1/Muttaqien/fr3_peg_100/episode_*/operator_commands.jsonl` files and the
110 action events in
`/data1/Muttaqien/fr3_pi05_real_rollouts_20/episode_06/policy_command_log.jsonl`.
It computes consecutive receipt/event deltas separately inside each episode.
The revised result runs the real executor loop against fake ROS endpoints and
a deterministic clock, including two chunks separated by a 484 ms neutral wait.

| Quantity | Saved baseline | Revised offline result |
| --- | --- | --- |
| SpaceMouse receipts | 212,623; 100.00 Hz; maximum gap 13.845 ms | Reference only |
| Policy sample spacing within a chunk | About 100 ms | 100 ms; ten samples per chunk |
| Between-chunk action-event gaps | Mean 468.403 ms; maximum 484.156 ms | Inference wait remains neutral; no invented actions |
| Effective policy action-event rate | 7.474 Hz over Episode 6 | Still depends on acquisition/inference/storage time |
| Arm publication interval | 100 ms for policy actions; one neutral per boundary | 10 ms in deterministic executor tests, including inference waits |
| Gaps above 250 ms | Ten in the saved **action** log | Zero in the deterministic publication trace |
| Arm multiplier | 0.15 | Identity 1.0 behind physical-evidence and operator gates; legacy 0.15 retained |

The saved policy action log omits neutral requests and controller receipts.
Its ten long action gaps alone cannot establish ten controller freshness
violations. Live transport timing, command acceptance, and physical response
remain untested. Higher-frequency refresh can change controller response even
with unchanged numerical gain; qualify it before collecting more outcomes.

## Training Units And Scaling

The label path is `_mean_held_twist()` in
`fr3_command_request_alignment.py`, followed by `fr3_request_lerobot_export.py`.
It time-averages already axis-mapped, normalized Twist requests over observation
windows. The seventh channel is the latched gripper button request. Labels are
neither SI velocities nor measured robot motion. Rollout decoding still uses
train-only OpenPI `Unnormalize` before selecting the seven request channels.
The read-only deployment audit passes the disjoint 80/20 source split and
train-only normalization checks.

The reproducible read-only audit compares all 212,623 Twist requests and 200
button requests across the 100 demonstrations with all 110 saved Episode 6
actions. Legacy replay and identity replay both have maximum absolute mapping
error zero against their respective contracts. The saved gripper log and the
current offline debounce replay both contain 109 holds and one open goal.

The following mean absolute request magnitudes compare all recorded SpaceMouse
messages with Episode 6 predictions and their logged sent values. They are
sample-weighted statistics from different trajectories, not matched-state
accuracy or a physical calibration.

| Twist axis | Recorded SpaceMouse | Decoded prediction | Legacy sent | Identity sent, offline |
| --- | ---: | ---: | ---: | ---: |
| linear X | 0.1899 | 0.1296 | 0.0194 | 0.1296 |
| linear Y | 0.2097 | 0.3596 | 0.0539 | 0.3596 |
| linear Z | 0.3023 | 0.1850 | 0.0278 | 0.1850 |
| angular X | 0.0835 | 0.1325 | 0.0199 | 0.1325 |
| angular Y | 0.1227 | 0.1190 | 0.0179 | 0.1190 |
| angular Z | 0.1151 | 0.1218 | 0.0183 | 0.1218 |

Every saved Episode 6 arm command equals its decoded prediction times 0.15
(maximum absolute residual zero). This confirms post-training attenuation,
but does not establish that removing it is safe or sufficient. The default remains
`legacy_limited`; the strengthened `identity_verified` gate is described above.
No physical gain or mapping is automatically changed. The request data config
has no `DeltaActions`/`AbsoluteActions` transform; its output adapter only
slices the seven request channels. No duplicate HID transform, clipping,
coordinate rotation or SI conversion is applied by this execution pipeline.

For a future mapping review, first freeze the deployed binary/source identity,
mounting/tool frames, axis signs and update semantics. Then obtain separate
operator approval for bounded arm-only signed-axis calibration, with controller
receipt/reference diagnostics and measured TCP response. Verify neutral/stop
and hold timing at the current gain before considering a separately approved
gain change. One X pulse or matching topic names cannot qualify all six axes.
See [mapping review](pi05_real_rollout_mapping_review.md) for the existing gates.
The earlier X-only pulse reported +0.383 mm X and -0.539 mm Z; its unresolved
cross-axis response and uncalibrated flange-to-peg TCP prevent an offline claim
of physical direction compatibility. No guessed axis remapping is introduced.
The intentional gripper debounce also differs from direct button control:
three confident policy samples are required, with at least 0.8 s between goals;
100 Hz arm refresh never supplies extra confirmations.

## Offline Validation And Limits

Run the focused CPU checks from the repository root, one pytest process per file:

```bash
for component in fr3_pi05_ros_executor fr3_pi05_request_adapter \
  fr3_pi05_deployment_audit fr3_rollout_outcomes \
  fr3_command_request_alignment fr3_supervised_arm_pulse fr3_multimodal \
  fr3_peg_request_policy fr3_request_norm_stats fr3_request_lerobot_export \
  fr3_request_sft_offline_eval fr3_request_live_shadow fr3_request_chunk_eval; do
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
    "tests/unit_tests/test_${component}.py" || exit 1
done
```

This exercises the production scheduler and ROS loop at the process edge with
fake endpoints, plus real subprocess pipes for asynchronous stop/failure.
Coverage includes 300/484/1500 ms inference waits, chunk restart, stale
observation/issue receipts, partial IPC, parent death, operator stop during
caller inactivity, deadline stop, malformed events, lost exclusivity/subscribers,
publisher failure, preparation delay, stale pose, travel and duration faults.
The completed focused suite has **205 passing tests across 13 files** on
workstation CPU, Python 3.11.15, with each file run in its own pytest process.
It includes both mapping modes, all signed axes, evidence/consent failures,
archive overlap/failure and operator stop during slow archival, in addition
to the existing timing and safety regressions.
No policy inference, ROS discovery, controller activation, or hardware e2e ran.
OpenPI input/output transform and normalization tests exercise CPU arrays only.
Existing TorchScript deprecation warnings are unrelated to this change.

Observation age now follows the oldest of all four state-source and two camera
host receipts through inference and IPC; validation never replaces it with a
new timestamp. The admission limits remain one second for observations and
300 ms for issued requests. An admitted chunk has a bounded one-second horizon,
with live pose checked throughout; these are distinct freshness contracts.
The 250 ms pose watchdog, 8 cm measured travel limit, 20 s executor duration,
1–15 s rollout duration, request bounds, gripper thresholds and exclusivity
checks retain their existing values. Controller timeout, gains, IK, torque and
safety configuration, policy weights and existing collected data are untouched.

`executor_events.jsonl` records preserve partial actions and terminal faults.
`commands_sent` and the legacy manifest `robot_commands_sent` count **policy
samples**; `policy_actions_sent` makes that explicit. `arm_publications` and
`max_publish_gap_ns` in the episode-end event/manifest describe transport,
including neutral refresh up to the end acknowledgement, excluding shutdown
cleanup. Neither counter proves controller acceptance.

Ruff lint/format, Python compilation and whitespace checks pass on changed
Python files. This branch omits `.github/workflows/ci-tests.yml`; restore upstream
coverage before an upstream submission. The docs symbol checker reports 24
existing findings in unchanged Sphinx pages. The RST markup checker cannot run
without `docutils`; EN/ZH Sphinx builds are skipped because Sphinx is unavailable
and these bilingual experiment Markdown files are outside both Sphinx trees.
An AI assistant implemented and checked this change.

Reproduce the complete recorded-request comparison without ROS or inference:

```bash
.venv/bin/python -m toolkits.realworld_check.fr3_pi05_deployment_audit \
  --raw-root /data1/Muttaqien/fr3_peg_100 \
  --episode /data1/Muttaqien/fr3_pi05_real_rollouts_20/episode_06
```

This verifies train-only normalization and the 80/20 split, audits recorded
ranges/units/timing and replays both software mappings and gripper debounce.
The train/validation export preflights also pass read-only: 12,288/3,078 labels.
[Aggregated evidence](pi05_execution_compatibility_evidence.json) retains source
hashes and the offline archive profile; it is not a physical approval record.

The next step is **one separately operator-approved real-world qualification**,
recording actual publication/receipt gaps, neutral transitions, controller
acceptance and measured response. Retain legacy mode until physical mapping
evidence satisfies the identity gate; full strength then requires per-run consent. Its result must be
reviewed before the 20 RECAP outcomes proceed. This task stops at offline
verification; the new pipeline has no physical qualification yet.
