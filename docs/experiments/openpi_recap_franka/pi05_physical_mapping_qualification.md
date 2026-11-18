# FR3 π₀.₅ Physical Mapping Qualification

Use this checklist to finish the physical evidence for scaling 1.0, then authorize one supervised rollout. **Status: 0/17 physical checks passed; execution blocked.** Existing software evidence is reusable, but the unresolved X response prevents physical sign approval. This packet stops before deployment, activation, calibration or policy execution. [中文](pi05_physical_mapping_qualification.zh.md)

## Evidence You Can Reuse

The [mapping review](pi05_real_rollout_mapping_review.md), [controller investigation](fr3_controller_investigation.md), [execution report](pi05_execution_pipeline_offline.md), [gripper report](pi05_gripper_validation.md), and [Episode 8 report](pi05_episode8_shutdown.md) establish the following boundaries. The [evidence inventory](pi05_mapping_qualification_evidence.json) records current file hashes and fresh offline results.

- Training uses time-averaged, already mapped normalized Twist requests, in order `[linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]`; replay applies no second HID transform or SI conversion. The working SpaceMouse interface stays untouched.
- Identity sends all six decoded arm values unchanged. Episode 6 and Episode 8 used gain 0.15. Episode 3's maximum logged TCP displacement was 5.479 mm; it does not qualify mapping or acceptance.
- The reported +X pulse delivered requests, valid IK and reference advances. Its latest final delta `[+0.001,+0.001,-0.003]` mm and earlier `[+0.383,-0.002,-0.539]` mm do not establish a resolved, isolated +X response. Raw synchronized physical traces are absent from this packet.
- At preparation, the old diagnostic library was `fc5128dd…3925744`. The Release library `3b160839…7752b4` passed nine CPU controller cases but remained staged and unused. Both are historical identities; inspect the actual mapped library before qualification. Do not substitute a staged hash for the running binary.
- Saved source uses linear offset `0.007 * R_mountᵀ * request` m and postmultiplied flange RPY `0.03 * request` rad, measured-pose-relative IK at 50 Hz, nominal reference updates at 1 kHz, and timeout 0.25 s. Zero mounting RPY is recorded. Source/config inspection resolves the historical 0.03/0.06 discrepancy for this source, not for an unidentified live binary.
- Episode 8 recorded 504 publications with maximum gap 11.806 ms, at gain 0.15. This supports transport feasibility, not controller receipt timing or identity physical qualification. Gripper logs/filter tests establish request semantics, not accepted open/close goals.

## The 17 Required Checks

Each row below names the exact key required by `MAPPING_CHECKS`. “Partial” means supporting evidence exists; every row still has `passed=false` in the [draft record](pi05_mapping_verification.pending.json).

| # | Required key | Reusable support | Remaining physical evidence |
| --- | --- | --- | --- |
| 1 | `linear_x_positive` | Partial: reported pulse; software sign test | Resolve +X above baseline noise with synchronized acceptance and pose traces |
| 2 | `linear_x_negative` | Software sign test; CPU IK fixture | Isolated −X response |
| 3 | `linear_y_positive` | Software sign test | Isolated +Y response |
| 4 | `linear_y_negative` | Software sign test | Isolated −Y response |
| 5 | `linear_z_positive` | Software sign test | Isolated +Z response |
| 6 | `linear_z_negative` | Software sign test | Isolated −Z response |
| 7 | `angular_x_positive` | Software sign test | Positive right-hand flange X rotation |
| 8 | `angular_x_negative` | Software sign test | Negative flange X rotation |
| 9 | `angular_y_positive` | Software sign test | Positive flange Y rotation |
| 10 | `angular_y_negative` | Software sign test | Negative flange Y rotation |
| 11 | `angular_z_positive` | Software sign test | Positive flange Z rotation |
| 12 | `angular_z_negative` | Software sign test | Negative flange Z rotation |
| 13 | `controller_acceptance` | Partial: reported receipt/IK/reference counters | Joint/reference/effort, robot mode/errors and measured response together |
| 14 | `tool_frame_alignment` | Source: URDF `base` → `fr3_link8`; broadcaster `O_T_EE` | Actual base/mount, flange → EE → Robotiq peg TCP transforms and tool identity |
| 15 | `timing_10hz_100hz` | CPU scheduler tests; legacy Episode 8 publication gaps | Physical 100 ms sample holds, 10 ms refresh/receipt gaps, neutral during waits |
| 16 | `neutral_stop` | Source holds last joint reference; CPU stop tests | Measured residual after neutral/expiry/operator stop; approved hardware stop check |
| 17 | `gripper_open_close` | Binary labels; debounce/overshoot tests; logged open request | Accepted open and close goals, measured travel, hold/no-repeat and stop behavior |

## Remaining Operator Actions

The prepared [calibration interface and operator procedure](pi05_calibration_operator.md) now supplies single-case six-axis and gripper commands, synchronized capture and explicit evidence review. The inventory above preserves the November 18 evidence snapshot; no new physical checks have passed.

Start by inspecting any newly available session logs. Reuse a valid trace when it names the same binary, configuration, tool, pose and command timing and meets the criteria below; repeat only missing cases. The historical pulse alone does not meet them.

1. **Freeze the installation and workcell.** The local operator checks clear free space, mounting, attached Robotiq/peg geometry, camera roles, initial pose, hardware stop access and the approved stop procedure. Save current `/proc/<actual-arm-pid>/maps`, mapped `.so` SHA-256, selected launch/config/URDF, actual controller parameters, active controller and seven claimed effort interfaces. Save a config manifest containing file hashes plus captured runtime defaults, mounting and tool transforms; `controller_config_sha256` will hash that manifest's bytes. Current archived K/D are `[600,600,600,600,250,150,50]` / `[30,30,30,30,10,10,5]`; preserve all gains, limits, timeout and scales. Stop competing SpaceMouse/policy publishers through their owning interface, preserving its code/configuration. Any diagnostic deployment is a separate operator maintenance decision using the existing [deployment procedure](../../../outputs/fr3_step3_prepare_20261118/deployment_and_rollback.txt); it is not required merely because a Release artifact exists.
2. **Resolve frames before motion.** Confirm physical base axes from the installed URDF/mount. Linear requests are along base X/Y/Z at mounting `[0,0,0]`; positive angular requests are right-hand rotations about the current flange X/Y/Z because rotation is postmultiplied. Verify `O_T_EE`, flange and peg TCP are distinct or document their calibrated transforms. Recover flange motion from measured EE pose using those transforms. For signed rotations evaluate `log(R_flange_startᵀ R_flange_now)`; peg translation during rotation may result from its lever arm. Do not infer an axis swap from uncalibrated TCP translation.
3. **Establish baseline and capture.** Save at least two seconds of full pose orientation/position, measured and robot-desired joints/efforts, robot mode/errors and `FR3_DIAG_V1` output before and after each case. Use the existing telemetry bag topic list in the deployment procedure. Verify the installed meanings of `tau_J`, `tau_J_d` and broadcaster fields; robot `q_d` is not private controller `q_desired_`. Record sender and controller-local monotonic timing; do not subtract unsynchronized host clocks. Predeclare noise/resolution, allowed off-axis response, residual travel/rotation and settling time with the operator. Unknown thresholds or unresolved motion mean **inconclusive**, never pass. A recorder on the command topic adds a subscriber and breaks the existing X tool's exactly-one-subscriber guard; use sender logs and controller receipts instead.
4. **Resolve +X first and stop for review.** Use the new interface's read-only preflight, then one separately confirmed `linear_x_positive` case from the operator procedure. It retains the 0.05 request/0.5 s caps and 10 mm translation cap, adds a rotation guard, and captures fresh baseline, full pose/joint/state and diagnostic evidence. The five default 100 ms slots are `[0.030,0.024,0.018,0.024,0.030]` on one signed axis, refreshed at 100 Hz. These request offsets do not promise physical travel. If the response remains unresolved or cross-axis, end qualification and investigate the trace; do not increase or repeat automatically. The legacy X-only 20 Hz utility remains a separate diagnostic and cannot establish policy transport timing.
5. **Complete missing signed directions only after +X review.** Run `+X, −X, +Y, −Y, +Z, −Z, +Rx, −Rx, +Ry, −Ry, +Rz, −Rz` as separately confirmed cases, reusing any already valid trace. The [operator procedure](pi05_calibration_operator.md) lists exact case names and commands. Each case affects one request axis, keeps the arm-only gripper inactive, and observes neutral afterward. The default maximum angular offset is 0.0009 rad (0.052°); inadequate resolution is inconclusive. Preserve hardware stop supervision and review before the next case. Do not bypass the identity record or run policy predictions for calibration.
6. **Share acceptance, timing and neutral evidence with those cases.** The reviewed calibration interface must hold each request in explicit 100 ms slots and refresh it every 10 ms, with zero extra gain. Capture actual publication and controller receipt gaps, 50 Hz IK/reference gate activity, neutral at the final slot, a neutral inference-like wait of at least 0.5 s and an operator stop during a held sample. Preserve the existing 50 ms executor publication deadline; any gap beyond it fails timing. Verify controller freshness against its unchanged 250 ms timeout, no replay/catch-up, no new moving request after processed stop, and residual motion within the operator's declared clearance. A short pulse cannot prove all production deadlines. For controller acceptance correlate counter deltas and moving-gated joint-error/PD/total-effort peaks with per-joint robot-desired/measured effort, joint changes and status. Counters or independent scalar peaks alone cannot pass acceptance. Neutral/timeout holds the last reference and can permit continued tracking; it is not an E-stop. Complete the local approved hardware stop test/check without bypassing safety or inducing a hardware fault. After a fault or unexpected motion use the approved hardware procedure and stop qualification.
7. **Qualify the gripper with the arm neutral.** Inspect the domain-70 Float32 bridge and domain-71 Robotiq controller/action endpoint without changing them. In clear finger space, an operator-approved gripper calibration sends three confident 10 Hz open samples (`1`, goal on third), holds, then sends close samples (`0`) only after at least 0.8 s from the last goal. Check accepted/completed goals and measured opening/closing, actual model/units and empty-space clearance. Verify an intermediate request such as `0.5` holds, a stable latched request issues no repeated goals, 100 Hz arm refresh adds no confirmations, and processed stop issues no new goals. An in-flight Robotiq goal is not canceled by arm neutral or the current executor stop: inspect and approve its completion/independent stop behavior. Direct gripper bridge operation alone does not qualify the policy filter. Use the new interface’s separately confirmed open/close/hold cases; it captures measured/status evidence and never cancels an in-flight goal.
8. **Finish the record and independently review it.** Populate the pending JSON only from completed evidence: real operator name, past UTC qualification time, verified base/tool frames, actual mapped binary hash and config-manifest hash. For each of the 17 checks, attach a nonempty measurement/review file with `evidence_path` relative to the record and its actual SHA-256; the file must identify the raw traces, their hashes, case interval, measurements, criteria and operator verdict. One file may support several checks when it actually covers them. Set `passed=true` only for those reviewed checks. Keep incomplete records blocked. Copy a fully approved record to a new filename, preserving the pending snapshot. The loader validates completeness and file integrity; it cannot authenticate an operator, inspect the running binary, validate a measurement or certify safety.

## Validate the Record Before One Rollout

Run the following read-only validation on the completed record before connecting the rollout. Use the same record and evidence bytes for parent and child; changing them invalidates consent. With the supplied pending record, rejection is expected because operator/physical approval is absent.

```bash
export OPERATOR_VERIFIED_MAPPING_JSON=docs/experiments/openpi_recap_franka/pi05_mapping_verification.pending.json
.venv/bin/python - <<'PY'
import os
from toolkits.realworld_check.fr3_pi05_request_adapter import load_mapping_verification
r = load_mapping_verification(os.environ['OPERATOR_VERIFIED_MAPPING_JSON'])
print(r['record_sha256'])
print('Required consent: RUN VERIFIED ' + r['record_sha256'][:12])
PY
```

Only after 17/17 checks, local installation matching and separate per-run approval, replace that environment variable with the completed record. The future command below uses the same checkpoint recorded in Episode 6 and requests one three-second supervised episode at gain 1.0; it was not executed. The operator must first confirm the trained starting setup and clearance for policy requests throughout `[-1,1]`. Small calibration pulses establish direction and operation, not safety for arbitrary policy trajectories. The executor's 8 cm guard observes translation only; rotation/contact clearance remains an operator responsibility under the unchanged hardware safeguards.

```bash
export PI05_CHECKPOINT='/data1/Muttaqien/logs/20261111-04:55:26-realworld_sft_openpi_fr3_peg_request_full3000_wandb/fr3_peg_pi05_request_offline_full3000_wandb/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt'
PYTHONPATH=. .venv/bin/python -m toolkits.realworld_check.fr3_pi05_supervised_rollout \
  --checkpoint "$PI05_CHECKPOINT" \
  --root /data1/Muttaqien/fr3_pi05_real_rollouts_20 \
  --verified-request-mapping \
  --mapping-verification "$OPERATOR_VERIFIED_MAPPING_JSON" \
  --max-episode-seconds 3
```

Type the exact `RUN VERIFIED <12-character digest>` at the later per-episode prompt. Require the ready/event logs to show `identity_verified`, `arm_gain=1.0`, 100 ms policy steps, 10 ms refresh and the approved record digest. ENTER stops new commands even during inference; retain hardware stop supervision. Existing gates also cover oldest observation receipt ≤1 s, issue age ≤300 ms, live pose ≤250 ms, exclusive arm/gripper publishers and available subscribers, bounded requests, one-second chunks, 20 s child duration and CLI duration within 1–15 s. No full-strength fallback is selected when verification fails. The installed identity still needs manual matching: the loader does not compare live `/proc` maps/configuration.

After shutdown, inspect `executor_events.jsonl`, `pipeline_timing.jsonl`, command/state logs and `outcome.json`. Require every logged sent arm sample to equal its decoded request, no stale or lost-acceptance interval, confirmed shutdown and an operator-reviewed outcome. Compare measured flange/peg movement, baseline noise, joint/effort response and task progress with the saved legacy evidence; these are different trajectories, not a controlled gain-only comparison. Diagnose “barely move” as resolved only from meaningful measured response and operator review, not request count. If inconclusive, preserve the attempt and stop before more rollouts. Existing Episode 1–8 files remain intact.

## Offline Verification in This Packet

The November 18 workstation CPU snapshot passed **269 tests**, one pytest process per file: request adapter 95, ROS executor 136, deployment audit 9, outcomes 19, X pulse 10. They use fake ROS/process-edge endpoints and pipes; there was no ROS discovery, controller activation, new inference or movement. Test fixtures contain synthetic approvals solely for unit tests; none were copied into the pending physical record. Reproduce the tests and dataset audit as follows:

```bash
for component in fr3_pi05_request_adapter fr3_pi05_ros_executor \
  fr3_pi05_deployment_audit fr3_rollout_outcomes fr3_supervised_arm_pulse; do
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
    "tests/unit_tests/test_${component}.py" || exit 1
done
.venv/bin/python -m toolkits.realworld_check.fr3_pi05_deployment_audit \
  --raw-root /data1/Muttaqien/fr3_peg_100 \
  --episode /data1/Muttaqien/fr3_pi05_real_rollouts_20/episode_06
```

The fresh audit passed the disjoint 80/20 source split, train-only normalization and recorded request replay in both modes, while retaining `physical_identity_execution_approved=false`. Evidence hashes and the pending record's expected rejection are checked separately. An AI assistant prepared this packet; production code, controller settings, weights and the SpaceMouse interface were not changed.

Local links, the 17 checklist keys and identical EN/ZH command blocks passed inspection. Both Sphinx builds passed the repository harness with zero reported warnings under its default autodoc-import filtering; these Markdown files are outside the Sphinx trees. The broader symbol check still reports 24 findings, and the RST markup check reports 61 warnings and zero errors, all in unchanged pages. Logs are under `outputs/fr3_mapping_qualification_20261118/`.

The prepared interface and its current regression boundary are documented in the [operator procedure](pi05_calibration_operator.md). All 17 physical checks remain pending.
