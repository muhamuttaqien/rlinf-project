# FR3 and Robotiq Physical Qualification Plan

Use this plan to review evidence before authorizing any future physical action
integration. It defines required records, responsible roles and acceptance gates;
it does not authorize or execute a physical test. Start from the
[Milestone 2C report](milestone_2c_passive_readiness.md), resolve passive status
contradictions, then have the named operators approve separate test procedures
and acceptance thresholds before any physical interaction.

**Current decision: NO_GO / NOT_QUALIFIED. Physical command authorization:
NOT_GRANTED.** No named operator has signed this plan. Blank records and missing
thresholds are blocking evidence, never assumed passes. Classifications are
**Verified runtime**, **Source-supported**, **Proposed**, and
**Unverified / blocked**, as defined in the milestone report.

**2D evidence update:** [Passive gap resolution](milestone_2d_passive_readiness_gaps.md)
corroborates the arm I/O failure and frozen acquisition under a stable manager.
P1 remains FAIL; P3 requires independently restored device acquisition as well as
visible ROS status. No recovery, physical test or operator signoff is supplied.

## 1. Qualification Record and Responsibilities

Assign accountable people before reviewing physical evidence. The human safety
lead owns permission to perform any separately planned test; software cannot
replace that permission or independent hardware safeguards.

| Required Record | Value to Record | Current State |
|---|---|---|
| Qualification/session identifier | Unique record and authority epoch | Unassigned; NOT_QUALIFIED |
| Date, location and validity period | UTC interval and task/cell scope | Unassigned; NOT_QUALIFIED |
| Responsible robot operator | Name, role, training/authorization and signature | Unassigned; NOT_QUALIFIED |
| Safety lead / stop operator | Name, independent stop responsibility and signature | Unassigned; NOT_QUALIFIED |
| Calibration owner | Name, instruments, traceable uncertainty and signature | Unassigned; NOT_QUALIFIED |
| Control/software reviewer | Name, deployment and interface review signature | Unassigned; NOT_QUALIFIED |
| Existing command-owner representative | Name, bridge/client inventory and handover signature | Unassigned; NOT_QUALIFIED |
| Evidence bundle | Immutable paths, hashes, raw measurements, decisions and deviations | 2C passive report only; insufficient |
| Final approval | Named operator, safety lead, approved procedure/scope and expiry | Not granted |

For each checklist row record **PASS / FAIL / NOT_QUALIFIED**, evidence reference,
UTC date, responsible person's name and signature. Use `FAIL` for contradictory
results and `NOT_QUALIFIED` for missing evidence. A passive `PASS` never authorizes
action. Retain rejected results and rationale; do not overwrite them with later
favorable snapshots.

Re-review after a controller/driver/bridge binary or configuration change, robot
or tool remounting, calibration change, safety setting change, unexplained
restart, ownership change, timing violation or fault. New sessions remain
disarmed; approval is not inherited from a previous process or test run.

## 2. Passive Preconditions

Resolve current status and identity gaps before planning any physical procedure.
These rows require read-only evidence and operator review, with no automatic
activation, recovery, switching or safety-limit changes.

| ID | Evidence Required | Responsible Role | Acceptance Criterion | Current Record |
|---|---|---|---|---|
| P1 | Controller-manager, controller, component and resource-list snapshots from the same deployment/epoch | Control reviewer | Expected controller/type active; expected resources consistently available/claimed; no conflicting or duplicate manager evidence | FAIL: arm active controller conflicts with unconfigured/unavailable hardware reports |
| P2 | Current loaded-library identity, deployment/configuration subset, message definitions and source/build provenance | Control reviewer | Review signed for the exact deployed contract; unexplained identity/drift resolved | NOT_QUALIFIED: current hashes observed; reproducible source/binary attestation outstanding |
| P3 | Workstation and NUC graph/status visibility, typed fresh faults, raw and normalized measured state | Control reviewer | Required signals available with valid identity, semantics, freshness and error handling | NOT_QUALIFIED: workstation custom types/remote identities/listing services incomplete |
| P4 | All arm publishers, raw/normalized gripper clients, internal relay and bridge pending/goal state | Existing owner representative | Complete inventory and approved future handover; graph absence alone insufficient | NOT_QUALIFIED: bridge and internal relay observed; pending bridge state unresolved |
| P5 | Exact action semantics and expected interface mapping | Control reviewer | Normalized measured-FK offsets distinguished from velocity/absolute pose; gripper radians/closure/width distinguished | Source-supported in 2B.1; physical qualification outstanding |

## 3. FR3 Frame and Tool Calibration Contract

Establish geometry explicitly before interpreting measured poses or action axes.
Each transform record must name parent/child frames, direction, metres/radians,
quaternion **x,y,z,w**, measurement method, uncertainty, mounting identity and
validity interval. Approved residual/uncertainty thresholds must be chosen by the
calibration owner for the intended task, before measurements are accepted.

| ID | Evidence Required | Responsible Role | Acceptance Criterion | Current Record |
|---|---|---|---|---|
| F1 | Relationship between `base` and `base_link`, independently checked descriptions/transforms and physical reference | Calibration owner | Direction and transform established with uncertainty; no inference from similar names | NOT_QUALIFIED |
| F2 | Measured `current_pose` / `o_t_ee` versus FK of named joints to `fr3_link8`, and configured `f_t_ee` / `ee_t_k` | Calibration owner + control reviewer | Same-time observations compared using verified transforms; flange, measured EE and stiffness roles separated | NOT_QUALIFIED; published fixed joint8 geometry alone is insufficient |
| F3 | Flange-to-mounted-Robotiq and flange-to-actual-TCP transforms | Calibration owner | Actual installed tool, adapter/fingers, payload and mounting orientation identified; traceable tool geometry and uncertainty | NOT_QUALIFIED; no transform inferred |
| F4 | Translation axes, inverse mounting rotation, RPY convention and right-composed orientation offsets | Control reviewer + robot operator | Audited 0.007 m / 0.03 rad full-scale semantics tied to approved frames and physical evidence under a separate procedure | Source-supported only; NOT_QUALIFIED |
| F5 | `fr3_joint1`–`fr3_joint7` identities/order, units and quaternion decoding | Control reviewer | Name-based order and finite dimensions verified; quaternion normalization/order preserved through recording | Code-level tests pass; physical frame association NOT_QUALIFIED |
| F6 | Task frame, workspace, forbidden regions, joint and tool constraints, reachable scope and uncertainty margins | Safety lead + calibration owner | Signed task-specific limits compatible with independent hardware safeguards; no invented or default Panda limits | NOT_QUALIFIED |

Do not turn description values into approved operational limits. Do not substitute
an arbitrary pose-to-Twist conversion for this contract. Future qualified
geometry must explicitly address temporal mismatch between joints and measured
pose; a single snapshot cannot prove a transform over the operating scope.

## 4. Robotiq Calibration and Status Contract

Treat raw knuckle angle, normalized closure and measured jaw width as separate
representations. A future calibration artifact must identify gripper serial/tool
configuration, finger pads, method, direction, fit validity, repeatability,
hysteresis/backlash, uncertainty, contact conditions and review expiry.

| ID | Evidence Required | Responsible Role | Acceptance Criterion | Current Record |
|---|---|---|---|---|
| G1 | Raw revolute knuckle angle in radians versus normalized closure | Calibration owner + control reviewer | Verified profile/register mapping and valid range; 0/1 direction and raw endpoints independently established | Source-supported only; NOT_QUALIFIED |
| G2 | Physical jaw opening width in metres across the intended operating range | Calibration owner | Traceable measurements and approved fit/error bounds; no nominal 85 mm or unverified constants used as calibration | NOT_QUALIFIED |
| G3 | Raw/normalized command direction and both action entry points | Robot operator + control reviewer | Approved representation and range agree with physical outcome under a separately authorized procedure | NOT_QUALIFIED; no goals sent |
| G4 | Object detection codes, contact while opening/closing, stalled/rejected/timeout/result handling | Robot operator + control reviewer | Signed interpretation separating reported contact, action success and actual grasp success; unknown/fault states block | Source-supported only; NOT_QUALIFIED |
| G5 | Force/effort path, retained commands, current telemetry and independent pinch hazards | Safety lead + control reviewer | Physical effort enforcement established through approved evidence/procedure; motor current is not force; `max_effort` echo is insufficient | NOT_QUALIFIED; active position path does not establish force control |
| G6 | Driver communication freshness, unknown fault 255, object timeout and missing serial readings | Control reviewer + safety lead | Device-data validity distinguishable from fresh ROS publication; operator-approved response to unknown/fault/stale measurements | NOT_QUALIFIED |

**Source-supported:** raw `0..0.7929` and normalized `0..1` do not share identical
register endpoint scaling in the deployed driver. The 2B.1 audit gives the source
mapping; this plan supplies no width conversion and no permission to calibrate by
moving the jaws. Any physical measurements requiring actuation need a new,
explicitly approved procedure.

## 5. Timing and Communication Qualification

Approve timing budgets and methods before accepting a latency claim. Preserve
source ROS stamps, local monotonic receipt stamps, clock identity, sequence and
observation provenance. Cross-host monotonic values are incomparable; use a
qualified synchronization/error model for ROS wall-clock differences.

| ID | Evidence Required | Responsible Role | Acceptance Criterion | Current Record |
|---|---|---|---|---|
| T1 | Clock synchronization health, relative host offset and uncertainty over load/restarts | Control reviewer | Signed maximum uncertainty and validity monitor; normal chrony/RMS alone insufficient | NOT_QUALIFIED |
| T2 | Source measurement age, callback receipt age, stream skew, IPC and diagnostic timing under representative load/loss | Control reviewer | Approved freshness/skew/uncertainty budgets, supported by recorded distributions and justified bounds | Short passive samples only; NOT_QUALIFIED |
| T3 | Command source-age lease, queue bound, ROS QoS/lifespan and controller timeout semantics | Control reviewer + safety lead | Per-source expiration enforced independently of resend/parent liveness; no stale replay; verified timing of controller behavior | Source-supported timeout parameter only; NOT_QUALIFIED |
| T4 | Gripper goal acceptance/result deadlines, preemption and retained-target behavior | Control reviewer + existing owner | UUID/correlation and bounded waits; submission/acceptance separated from execution/grasp success; unresolved result disarms | Offline mock tests only; NOT_QUALIFIED |
| T5 | Network loss, IPC failure, parent/worker death and restart handling | Control reviewer + safety lead | Reviewed fault injection plan and results; pending commands discarded; independent hardware response understood | Offline IPC tests pass; physical effects NOT_QUALIFIED |

The current Twist has no source timestamp, sequence, authority or acceptance
acknowledgement. A future upstream adapter lease cannot make the NUC transport
authenticated or safety rated. Timing of actual command effects requires separate
operator-approved instrumentation/procedures; this milestone published no latency
test command. Choose no timing bound from a short quiet-network sample.

## 6. Command Authority and Failure Responsibilities

Separate connecting to observations from enabling commands. A future action
session must require a reviewed deployment identity, task/limit/calibration
profile, fresh observation/fault evidence, exclusive handover, operator identity,
explicit scope/expiry and a fresh authority epoch. Synthetic profiles from 2B.2
can never supply physical approval.

| Lifecycle / Failure | Proposed Software Responsibility | Operator / Hardware Responsibility |
|---|---|---|
| Connect/readiness | Read-only attach; fail closed on inconsistent or incomplete evidence; no activate/restart/recover/switch | Review current mode, cell and independent safeguards |
| Handover | Inventory arm publishers, both gripper servers, internal relay, external bridge and pending work; require an explicit lease/arbiter model beyond graph counts | Existing owner and new operator agree on exclusive authority and a separately approved transition; this tool never takes ownership |
| Enable | New signed, time-limited session after all gates; explicit operator request separate from connect | Confirm scope, responsible stop operator, independent protections and procedure |
| Send action | Validate kind/dimensions/finite values/range, identity, sequence, source lease, observation freshness and approved limits; never clip silently | Remain responsible for the authorized physical scope |
| Invalid/stale/fault evidence | Reject, disarm, discard pending actions and report uncertain submission/result state | Review fault; recovery/re-enable requires fresh approval, never automatic recovery |
| Communication loss or process death | Stop scheduling/publication; worker must not refresh an old request; drop queues and log last known status | Apply the independently qualified response to the cell condition; software cessation is not an emergency stop |
| Reset/disconnect/reconnect | Non-moving, disarmed cleanup; new authority epoch; no queued replay, automatic neutral publish or implicit goal cancel | Review retained targets/unknown goals and any necessary separately authorized resolution |

Future authority must cover both raw and normalized action paths. The internal
relay is part of controller implementation, while an external bridge is another
command source. Do not equate a ros2_control resource claim with exclusive ROS
publisher/action-client ownership. Never deactivate an external source through
a readiness checker.

### Four Different Stop Claims

Assign each stop function independently. Source-supported controller behavior is
not proof of the physical response of the complete cell.

| Claim | Meaning and Owner | Current Qualification |
|---|---|---|
| Stop software publication | RLinf ceases scheduling/writing; in-flight delivery and retained targets can remain | Offline cleanup tested; no physical stopping claim |
| Hold controller reference | NUC controller retains its reference; arm impedance/coriolis torques continue; gripper can retain a position command after failure | Source-supported; physical behavior NOT_QUALIFIED |
| Protective stopping | Robot/cell protective safety function with its own conditions and validated response | Safety lead must identify and qualify; NOT_QUALIFIED |
| Hardware emergency stopping | Independent emergency hardware chain, accessible operator and defined cell procedure | Safety lead/stop operator responsible; NOT_QUALIFIED |

The qualification record must name the actual protective and emergency-stop
systems, their inspection/test evidence and responsibilities. It must not claim
that a watchdog, zero Twist, action cancellation or process kill implements those
functions. No emergency/protective stop was exercised by this milestone.

## 7. Formal Go/No-Go Checklist

Apply an AND gate to the following requirements in an operator-reviewed record.
Every row needs a signed `PASS` with evidence for the exact deployment, tool,
task and session. Any `FAIL`, `UNKNOWN`, `NOT_QUALIFIED`, missing owner, expired
record or unresolved conflict yields **NO_GO**. This planning document cannot
grant `GO` or enable code, even if later records are filled.

| Gate | Required Evidence / Responsible Signer | Current Decision |
|---|---|---|
| Action semantics and controller identity | P1/P2/P5; control reviewer | NO_GO: arm listing contradiction and incomplete attestation |
| Frames, TCP and task constraints | F1–F6; calibration owner + safety lead | NOT_QUALIFIED |
| Gripper representation/contact/effort | G1–G6; calibration owner + control reviewer + safety lead | NOT_QUALIFIED |
| Independent hardware safety controls | Protective/emergency system records; safety lead + stop operator | NOT_QUALIFIED |
| Exclusive command authority | P4 and reviewed handover/lease covering raw+normalized routes; existing owner + robot operator | NOT_QUALIFIED |
| Observation and source-command freshness | P3/T2/T3; control reviewer | NOT_QUALIFIED |
| Network, timing and clock uncertainty | T1–T5 with approved limits; control reviewer + safety lead | NOT_QUALIFIED |
| Failure/reset/disconnect handling | Reviewed failure procedures and physical evidence; control reviewer + safety lead | NOT_QUALIFIED |
| Explicit operator approval | Named operator/safety lead signatures, scope, procedure, expiry and authority epoch | NOT_GRANTED |

Record the final decision, all blocking rows, evidence hashes, reviewer signatures,
and the authorized next step. Current final decision is **NO_GO**. The next step
is operator review and closure of passive inconsistencies/calibration/safety
requirements. A future supervised physical qualification milestone requires new
explicit authorization and approved procedures. Live action integration, policy
inference and real-world collection remain outside this plan's permission.
