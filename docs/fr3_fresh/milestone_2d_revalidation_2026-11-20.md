# Milestone 2D Follow-up — FR3 Passive Recovery Revalidation (2026-11-20)

This follow-up records the **operator-provided November 20, 2026 passive recovery revalidation** after the failed-acquisition investigation in [Milestone 2D](milestone_2d_passive_readiness_gaps.md). It does **not** replace, erase, or invalidate the earlier audit evidence of FR3 hardware `unconfigured`, frozen device time, and libfranka I/O failure. The earlier failure was real for its captured controller session. The reason for recurring `Move command aborted!` has **not** been established, and long-term acquisition stability is not qualified.

The source is a workstation-generated `fr3_passive_readiness_v2` report, `fr3_readiness_recheck.json`, assessed **2026-11-20T16:20:25.736+09:00**. The complete raw JSON was supplied by the operator and should be retained outside Git with the original byte SHA-256 `bbe691a3dd3101d8171229eb11b41e82e2ff6768bd2538b6fe23a482b0c740ee` (332,749 bytes). A compact provenance and measurement summary is committed separately in [revalidation evidence](evidence/milestone_2d_revalidation_2026-11-20.json). This digest identifies the exact source file; it does not attest the robot's safety or authorize control.

## Passive Readiness Assessment

The merged report passes all six passive evidence categories for the captured session. Physical qualification remains outstanding, as shown separately below.

| Category | 2026-11-19 Milestone 2D | 2026-11-20 recheck | Scope |
|---|---|---|---|
| Observation readiness | FAIL | **PASS** | Includes NUC-side device-time progression, not only fresh ROS headers |
| Controller-status visibility / consistency | FAIL | **PASS** | NUC-side Humble manager, hardware and resource claims agree in the sampled controller epoch |
| Gripper-status visibility | PASS | **PASS** | Typed reported fault/status is accessible; serial-device freshness is not attested |
| Interface identity consistency | PASS | **PASS** | Expected ROS types and configuration; no reproducible build/physical attestation claim |
| Timing evidence availability | PASS | **PASS** | Timestamps are observable; worst-case timing and inter-host clock bounds not qualified |
| Ownership evidence availability | PASS | **PASS** | Command-client graph visible; **not** exclusive command authority |
| Physical safety qualification | NOT_QUALIFIED | **NOT_QUALIFIED** | No operator-authorized physical safety qualification |

**Physical command authorization: `NOT_GRANTED`. Qualification gate: `NO_GO`.** The revalidation does not change either value.

## Fresh NUC FR3 Evidence

All three bounded samples used ROS 2 Humble domain 70 and the same controller-manager PID **516931**, with start ticks **321456315** and identical before/after process identity in each window. This differs from the November 19 manager PID 232045 and identifies a new controller session. The observed `joint_impedance_ik_controller` was `active`, `FrankaHardwareInterface` was `active`, and seven named FR3 `/effort` command interfaces were `available` and `claimed`.

| Window | Collected (JST) | Manager PID | Hardware state | FR3 effort interfaces | Device time (s) |
|---|---|---|---|---|---|
| 1 | 2026-11-20T16:20:11.944+09:00 | `516931` | `active` | 7/7 available, 7/7 claimed | 2535.880 → 2537.787 |
| 2 | 2026-11-20T16:20:14.085+09:00 | `516931` | `active` | 7/7 available, 7/7 claimed | 2537.930 → 2539.926 |
| 3 | 2026-11-20T16:20:16.229+09:00 | `516931` | `active` | 7/7 available, 7/7 claimed | 2540.075 → 2542.070 |

The three NUC windows reported **1869, 1976, and 1896** typed FR3 health samples, respectively. Device time advanced in the sampled windows, with no detected reversals. The latest source-reported FR3 `current_errors` flags were all false, but that observation is not an independent safety verdict. The Workstation was also receiving valid standard joint, pose, and wrench streams during this capture.

The Workstation's direct Jazzy inspection still had **`UNKNOWN`** entries for remote controller-manager listing and the missing `franka_msgs`/`robotiq_msgs` packages. These limitations were resolved **for this merged passive evidence assessment** through the existing bounded NUC-side Humble collector, not by making the Jazzy and Humble message interfaces interchangeable.

## Operator-Reported Context (Not a Passive Test)

The operator separately reported running libfranka `communication_test`, which **physically moved the robot** and reported a **10,000-sample success rate of 1.00**. This is operator-reported context, separate from the passive JSON evidence. It was not run by the read-only RLinf preflight or during this documentation follow-up, and its output was not independently remeasured here. It does not authorize a repeat, establish the cause of earlier aborts, or qualify the complete control system.

The previous libfranka `Move command aborted!` events remain historically valid. Their root cause has **not been established**. The earlier report's source-supported explanation of frozen acquisition and divergent software status does not establish why all prior aborts occurred. A coherent new session with advancing device time does not establish a permanent repair or long-term stability.

## Remaining Blockers and Next Step

Physical operation continues to require operator-reviewed diagnosis/handling of prior FCI aborts; approved FR3 frame/TCP and Robotiq width/effort calibration; independent protective and emergency safeguards; qualified command/source expiry, transport timing and clock uncertainty; explicit bridge/goal handover and command-authority exclusivity; and named operator approval. See [physical qualification plan](fr3_physical_qualification_plan.md).

This follow-up ends with recording, signing and pushing the documentation. Any future Milestone 3 work requires a separate request. The PASS snapshot does not authorize live RLinf actions, command ownership, movement or physical data collection. Reassess readiness for each new physical session and retain later failures alongside this evidence.

## Validation of This Documentation

The source file's byte SHA-256 matched the expected digest before using the supplied local revalidation bundle. The compact evidence preserves exact wall/monotonic timestamps, host/boot clock identities, controller name/type and library hashes, manager epochs, named FR3 effort claims, hardware state and measured device-time progression. Empty asserted-error lists replace repeated all-false flags while preserving their field count. Monotonic values remain associated with their source host and are not compared across hosts.

The fresh branch and initial clean working tree matched baseline `9f5128ffc23cd7eb57ed9997d3cf279f9813aa72`. Fetching the remote confirmed the same baseline without requiring reconciliation. An AI assistant prepared this documentation for the human submitter using the supplied bundle and source JSON. The upstream open PR/issue search for `FR3 revalidation` returned no matches.

Source-hash verification, strict JSON parsing, every compact snapshot/decision comparison, Markdown links/report consistency and whitespace diff checks passed. Both EN and ZH Sphinx builds produced zero warnings under the repository harness's default autodoc filtering; the existing bilingual guide markup/symbol checks also passed. No doc-code or EN-ZH consistency issues found in checked scope. These standalone milestone files remain outside the bilingual source trees. No runtime source, tests or action functionality changed, and no robotics regression or hardware-control test is claimed for this documentation-only follow-up.
