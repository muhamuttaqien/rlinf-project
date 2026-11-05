# Milestone 2B — Robotiq hardware and DDS qualification

Milestone 2B is **PARTIAL**: the existing NUC bridge passed a bounded serial-disabled runtime test and delivered 180 matching observation messages to the Jazzy workstation. Physical FC03 feedback is **NOT TESTED**, with the physical phase **BLOCKED** by incomplete serial-owner visibility. No physical serial port was opened, no actuation command was issued, and no bridge or FR3 production source was changed.

Date: **2026-10-27 JST**. The measured publication interval was **08:22:06.458183–08:22:24.358317 JST**. This report distinguishes application-topic success, a discovery-metadata incompatibility, and the independent physical-ownership gate. It does not declare the system fully hardware-qualified.

The [ownership continuation](#8-ownership-continuation--administrative-decision-required) establishes why the ordinary account cannot satisfy the current gate, verifies the ownership limits with five focused PTY tests, and specifies the administrator decision needed before physical testing. Production ownership behavior remains fail-closed; no privileged deployment was authorized or introduced.

## 1. Scope and evidence

The [Milestone 1 audit](milestone_1_system_interface_audit.md), [Milestone 2A report](milestone_2a_robotiq_observation_report.md), [bridge README/schema](../../toolkits/robotiq_ros2/README.md), bridge sources/configuration and [historical hardware baseline](../../../fr3-system-audit/docs/AUDIT_BASELINE.md) were read before execution. Historical gripper readings were not reused as current measurements.

Qualification scripts and raw evidence are retained in [`outputs/fr3_milestone2b`](../../outputs/fr3_milestone2b). This output directory is ignored by Git; retain its contents with the report when sharing the evidence. The scripts create only temporary ROS observers and the serial-disabled bridge process. They do not import RLInf hardware classes or invoke gripper APIs.

| Evidence | Content |
| --- | --- |
| [`phase1_nuc.json`](../../outputs/fr3_milestone2b/phase1_nuc.json) | Source/executable identities, device metadata, visible owners, procfs errors and initial dependency inspection |
| [`artifacts.json`](../../outputs/fr3_milestone2b/artifacts.json), [`patch_identity.json`](../../outputs/fr3_milestone2b/patch_identity.json) | Sourced dependencies, private vendor-copy hashes, build settings and equivalence of the final patch |
| [`phase2_nuc.json`](../../outputs/fr3_milestone2b/phase2_nuc.json) | All NUC messages, 18 descriptor snapshots, node log, command, shutdown result and serial-path trace checks |
| [`workstation_messages.jsonl`](../../outputs/fr3_milestone2b/workstation_messages.jsonl) | Topic/publisher discovery, all workstation messages and receipt clocks |
| [`summary.json`](../../outputs/fr3_milestone2b/summary.json) | Message equality, schema assertions, rates, intervals and receipt-age measurements |
| [`postcheck_nuc.json`](../../outputs/fr3_milestone2b/postcheck_nuc.json), [`local_identity_cleanup.json`](../../outputs/fr3_milestone2b/local_identity_cleanup.json) | Full syscall trace, static vendor linkage, cleanup, package versions and both installed graph schemas |
| [`orchestration.json`](../../outputs/fr3_milestone2b/orchestration.json) | Actual sourced execution commands and exit codes |

## 2. Environment and source/build identity

The qualification used the existing Humble/Jazzy installations and domain-70 Fast DDS environment. No middleware, ROS distribution, system service, installed robot binary or hardware setting was changed.

| Item | Workstation | NUC |
| --- | --- | --- |
| Host | `pca03395` | `aistfranka-NUC13ANK-B` |
| ROS distribution | Jazzy | Humble |
| Runtime account | Existing workstation user | `aist-franka`, UID 1000 |
| `rclcpp` | 28.1.21 | 16.0.19 |
| `diagnostic_msgs` | 5.3.8 | 4.9.2 |
| `rmw_fastrtps_cpp` | 8.4.4 | 6.2.10 |
| Fast DDS / Fast CDR | 2.14.6 / 2.2.7 | 2.6.12 / 1.0.29 |
| `rmw_dds_common` | 3.1.1 | 1.6.0 |
| Process environment | `ROS_DOMAIN_ID=70`, `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` | Same three variables |

The integration checkout remained at HEAD `cdc99764dc4ae78bd25add1b83eb908da540d252`, with the previously uncommitted Milestone 2A package and earlier artifacts retained. Runtime logs were directed to `/tmp/fr3_m2b_workstation_ros_logs` and `/tmp/fr3_m2b_nuc_ros_logs`.

The existing NUC executable was:

```text
/tmp/rlinf_robotiq_m2a_3krbaoaq/build/robotiq_observation
SHA-256 b005b04bf7e2ac8484d3e412d22df1aac56aa1875d1419abdefe633176a40ce8
```

Identity verification is **PASS**:

- All eight vendor source/header hashes matched [the pinned identities](../../toolkits/robotiq_ros2/cmake/vendor.cmake) against `/home/aist-franka/robotiq/robotiq-gripper-interface-main`.
- The executable was available, and the link command contained the private `librobotiq_vendor.a`. The existing installed vendor `.so` was not loaded.
- The staged package's implementation sources matched the integration checkout. Its patch file had the earlier context-bearing formatting: staged SHA `1fc14f358c4918a0d2496ab33b2d0d9d442e244587cea102611a29d31963aca8`, final checkout SHA `b72c029e2e4c9af108bd617793e642b847f560a63685f685905fad710e19a747`.
- Applying the current patch with zero fuzz to the hash-verified source snapshot produced **all eight staged vendor files byte-for-byte**. This is patch-format drift, not a change in the compiled implementation. No rebuild was needed or performed.
- Initial unsourced `ldd` output listed unresolved ROS libraries. After sourcing `/opt/ros/humble/setup.bash`, all dependencies resolved; the actual node run also loaded and exited successfully. No dependency installation was needed.

## 3. Phase 1 — ownership and permissions

The stable device path resolved to the expected FTDI adapter, but the intended account could not establish complete process ownership. This independently blocks physical qualification.

| Check | Result | Observed evidence |
| --- | --- | --- |
| Stable device identity | PASS | `/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DA95C62Y-if00-port0` → `/dev/ttyUSB1`, character device major/minor `188:1` |
| USB identity | PASS | `ID_VENDOR_ID=0403`, `ID_MODEL_ID=6015`, `ID_SERIAL_SHORT=DA95C62Y`, `ID_MODEL=USB_TO_RS-485`, `ftdi_sio` |
| Device access bits | PASS | Owner UID 0, group 20, mode `0660`; intended account reports read/write access through `os.access` |
| Visible serial owners | PASS within inspected scope | No matching character-device descriptors in inspectable processes; `fuser -v` returned 1 with empty stdout/stderr |
| Visible YUBI/Robotiq processes | PASS within inspected scope | No matching driver/bridge candidates found; this does not certify inaccessible processes |
| Complete `/proc` owner inspection | **BLOCKED** | **395 visibility errors**; first recorded error: `[Errno 13] Permission denied: '/proc/1/fd'` |
| Physical-device flock state | PASS for file inspection | `/tmp/rlinf-robotiq-188-1.lock` was absent before and after the disabled test; `/proc/locks` was inspected |
| Physical `TIOCEXCL` behavior | NOT TESTED | Would require opening the physical port; prohibited while the ownership gate fails |
| Existing claim/exclusivity implementation | IMPLEMENTED | Source still uses major/minor flock, fail-closed procfs scanning and private-library `TIOCEXCL`; prior 2A PTY tests remain historical evidence and were not repeated |

The first inaccessible descriptor directory is sufficient to make the existing `PortClaim` reject a physical connection. The 395 errors are recorded paths from the inspection, not a claim that 395 processes own the adapter. Empty visible-owner and `fuser` results do not establish absence of an owner in inaccessible processes.

No permissions were bypassed, process terminated, lock removed, service changed, or serial device opened. The serial-disabled test does not enter `SerialObserver` or acquire this claim, so it could proceed safely despite the physical gate.

## 4. Phase 2 — serial-disabled runtime and DDS

One bounded session exercised the existing C++ node on the NUC, a temporary local Humble observer, and a temporary workstation Jazzy observer. The node was run directly from the verified build, with its existing parameter file and an explicit `enabled:=false` override. No package installation, ROS CLI daemon, custom message or extra DDS bridge was introduced.

| Required observation | Result | Measured evidence |
| --- | --- | --- |
| Disabled node startup | PASS | Log: `Serial observation disabled; publishing missing state` |
| No serial opening | PASS | Full child syscall trace covered open/openat/openat2/ioctl/close; zero physical serial-path, physical-lock or TIOCEXCL/TIOCNXCL hits; 18 descriptor snapshots contained no serial descriptor |
| Topic and type | PASS | Workstation discovered `/robotiq_observation/state` as `diagnostic_msgs/msg/DiagnosticArray` |
| NUC publication/reception | PASS | 180 messages observed during approximately 18.03 s of node runtime |
| Humble → Jazzy application DDS | PASS | Workstation received 180 messages; all 180 source timestamps and complete message payloads matched NUC observations |
| Disabled schema and validity | PASS | Every sample had schema version `1`, `state=missing`, `communication_valid=false`, `last_attempt_valid=false`, sequence `0`, historical timestamps `0`, age `-1` and metric-width availability `false` |
| No fabricated measurements | PASS | Every sample omitted measured/requested positions, current, activation/object/fault measurement fields; diagnostic severity remained STALE (numeric value 3) |
| Source timestamps and encoding | PASS | Header seconds/nanoseconds and all key/value fields survived delivery; empty frame ID retained |
| Workstation receipt-age measurement | PASS | Local monotonic receipt times recorded; final age since last receipt was **9.4418 s** after the bounded publisher stopped |
| Clean shutdown | PASS | SIGINT to the owned node PID `724816`; exit code 0 in **0.1141 s**, no forced cleanup; both observer/orchestration processes exited 0 |
| Final process cleanup | PASS | No remaining bridge executable or workstation qualification observer; physical lock still absent |

The publisher used reliable, volatile, keep-last-1 QoS as implemented. The workstation recorded a compatible reliable/volatile publisher endpoint. No subscription or publisher was added to a gripper-command or Franka-control topic.

### Rates and timing

The actual application stream matched the configured 10 Hz publication timer. Serial polling stayed disabled; a physical 5 Hz poll rate was not exercised.

| Quantity | Observed value |
| --- | --- |
| NUC publication rate from header intervals | **9.999925 Hz** |
| Workstation receipt rate from local monotonic intervals | **10.001005 Hz** |
| Publication interval, min / median / max | 0.099627 / 0.100002 / 0.100356 s |
| Workstation receipt interval, min / median / max | 0.097861 / 0.099994 / 0.100841 s |
| Workstation wall receipt minus NUC publication stamp, min / median / max | 0.000571 / 0.000871 / 0.002976 s |
| Physical polling rate | NOT TESTED; configured 5 Hz, disabled for this session |

The cross-host wall-clock differences are observed timestamp skew, **not calibrated network latency**. Clock offset/synchronization was not qualified. Workstation-side age since local receipt is independent of inter-host clock alignment. No missing message was observed between the two recorded sets; this one short run does not establish sustained delivery guarantees.

### Discovery metadata warning

Application-message qualification passed, while remote node-name metadata was incomplete. The first workstation publisher-discovery record reported `_NODE_NAME_UNKNOWN_` and `_NODE_NAMESPACE_UNKNOWN_`. The NUC node log contained this exact line:

```text
sequence size exceeds remaining buffer
```

Remote node-name discovery is therefore **FAIL/incomplete in the recorded sample**, separately from successful topic/type discovery and payload delivery. No later endpoint-name sample was captured, so this report does not assert that the names stayed unknown throughout the session.

File inspection identified a concrete private graph-schema incompatibility: installed Humble `rmw_dds_common/msg/Gid.msg` is `char[24] data`, while Jazzy's is `char[16] data`. `NodeEntitiesInfo` and `ParticipantEntitiesInfo` embed these GIDs. This is evidence of a graph-metadata wire-layout difference and supports a discovery-deserialization explanation for the warning and incomplete names. The exact warning call site was not stack-traced, so attribution of that particular log line remains an inference. Both installed schema sets and package versions are retained in the evidence.

No production fix was necessary for the successful standard observation-topic delivery. Middleware upgrades, schema substitutions and a replacement DDS bridge would change infrastructure beyond the demonstrated application need; none was performed. Future work that depends on complete cross-distro node-name graph metadata must resolve that compatibility issue explicitly.

## 5. Phase 3 — physical gate

Physical FC03 feedback is **NOT TESTED**, and the phase is **BLOCKED** by the failed ownership-visibility precondition. No onsite confirmation was supplied in this turn. Because ownership was already blocked, no physical session was proposed or started; onsite confirmation alone would not remove that blocker.

No current measured raw position, requested-position echo, activation/object status, current, device fault, physical feedback sequence/freshness or physical polling rate is available from this milestone. Earlier raw positions and fault codes remain historical baseline evidence only. A disabled `missing` sample is not a fault-free gripper reading.

Physical port release after an enabled session is also **NOT TESTED**: the port was never acquired. This report instead verifies that the disabled process never opened it and left no physical claim. No activation, reset, open/close, target-position, fault clearing, FR3-controller startup or arm motion command was issued.

After ownership visibility is resolved through approved laboratory administration, any physical continuation must first request explicit confirmation of all four user-specified conditions: authorized operator physically present, gripper condition approved for read-only diagnostics, no other adapter owner, and communication permitted under current procedures. A later physical session must remain bounded and FC03-only; the current evidence does not authorize it.

## 6. Actual commands and harnesses

The commands below identify the actual bounded execution. The linked scripts retain the inspection logic, timing, process ownership and cleanup details; they are qualification artifacts under the ignored output directory, not production bridge changes.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 nuc-franka 'python3 -' \
  < outputs/fr3_milestone2b/inspect_nuc.py \
  > outputs/fr3_milestone2b/phase1_nuc.json

ssh -o BatchMode=yes -o ConnectTimeout=10 nuc-franka \
  'bash -c "source /opt/ros/humble/setup.bash && python3 -"' \
  < outputs/fr3_milestone2b/artifact_check.py \
  > outputs/fr3_milestone2b/artifacts.json

python3 outputs/fr3_milestone2b/run_disabled_qualification.py
python3 outputs/fr3_milestone2b/analyze.py

ssh -o BatchMode=yes -o ConnectTimeout=10 nuc-franka 'python3 -' \
  < outputs/fr3_milestone2b/postcheck_nuc.py \
  > outputs/fr3_milestone2b/postcheck_nuc.json
```

The [orchestrator](../../outputs/fr3_milestone2b/run_disabled_qualification.py) sourced `/opt/ros/jazzy/setup.bash` locally and `/opt/ros/humble/setup.bash` remotely; exported the three specified domain/discovery/RMW variables; launched the [workstation observer](../../outputs/fr3_milestone2b/dds_subscriber.py) for 28 s; and streamed the [NUC harness](../../outputs/fr3_milestone2b/disabled_nuc_runtime.py) over SSH. The NUC harness's exact publisher command was:

```bash
strace -f -ttt -e trace=open,openat,openat2,ioctl,close \
  -o /tmp/fr3_m2b_disabled_fc64sgni/syscalls.log \
  /tmp/rlinf_robotiq_m2a_3krbaoaq/build/robotiq_observation \
  --ros-args \
  --params-file /tmp/rlinf_robotiq_m2a_3krbaoaq/robotiq_ros2/config/observation.yaml \
  -p enabled:=false
```

The harness also subscribed locally, sampled only its node's descriptors, sent SIGINT to that node after 18 s, waited for exit, destroyed its own observer and retained logs under `/tmp/fr3_m2b_disabled_fc64sgni`. Final inspection copied the full 436-line syscall trace into the local evidence JSON. It did not signal any pre-existing process.

Additional identity inspection used `ldd`, `udevadm info --query=property --name=<stable-device>`, `fuser -v <stable-device>`, `/proc/*/fd` metadata, `/proc/locks`, installed `.msg` files, CMake cache/link files, SHA-256 and `dpkg-query -W`. The current final patch was applied only to a disposable local `/tmp/fr3_m2b_patch_identity_*` source copy and compared with the staged vendor files. No compile, package install or CPU-suite rerun occurred.

The initial SSH command encountered a sandbox socket denial, then succeeded through the approval mechanism. This was an execution-environment restriction, not a NUC hardware failure. The on-NUC procfs permission errors remained real and were not bypassed.

## 7. Changes, remaining risks and next work

The qualification harness/evidence is **IMPLEMENTED** under `outputs/fr3_milestone2b`, and this report is the only new production-tree documentation deliverable. Bridge sources, launch/configuration, vendor source, installed binaries, FR3 controls and services remain unchanged. Existing uncommitted 2A changes were preserved. The 2A nine-test suites were deliberately not repeated because no relevant implementation changed.

Remaining qualifications are physical ownership visibility, an authorized bounded FC03 session, physical-device claim/release behavior, actual device faults and timing, exact model/metric calibration, sustained delivery and complete cross-distro graph metadata. The read-only-by-default design and absence of a command path were preserved. Full repository/GPU tests and published EN/ZH Sphinx builds were not relevant to this report-only change.

Recommended next work is to resolve the procfs visibility requirement with the responsible administrator without weakening the owner check or using a permission bypass, then obtain the four-point onsite authorization before a separately approved 10–20 s FC03-only qualification. Retain exact faults, compare accepted feedback sequence rate with configured polling, verify workstation receipt-age handling and prove port release after shutdown. Complete graph-metadata compatibility work only if the subsequent integration depends on that metadata; standard observation-topic delivery already passed this bounded test.

Prepared with an AI assistant. **Milestone 2B stops here as PARTIAL. Physical qualification, gripper commands, policy control and subsequent integration milestones have not been performed.**

## 8. Ownership continuation — administrative decision required

The ownership issue is now identified as a mismatch between the global inspection contract and the ordinary account's authority. **Disposition: BLOCKED pending administrator review, not resolved by a safe user-only code change.** The existing protection is retained. The physical qualification gate is **not ready** for separately authorized testing yet.

### Root cause and practicality

[`PortClaim::acquire()`](../../toolkits/robotiq_ros2/src/observation.cpp) first locks a file keyed by character-device major/minor. It then enumerates every numeric `/proc` entry, lists each `fd` directory, and follows its descriptor links with `stat()` to detect another open handle to that device. It accepts disappearing entries (`ENOENT`) as process/descriptor races, but throws on permission failures. `SerialObserver::connect()` opens the vendor connection only after that scan returns successfully.

The scan asks for negative evidence across the process view, so an unreadable process cannot be classified as harmless. That is why the first `/proc/1/fd` denial blocks connection even when all inspected processes and `fuser` show no adapter owner. This is a deliberate fail-closed decision, not a Modbus, DDS or adapter-access failure. File-descriptor link access is subject to Linux's `PTRACE_MODE_READ_FSCREDS` check. [Linux proc_pid_fd manual](https://man7.org/linux/man-pages/man5/proc_pid_fd.5.html).

A read-only NUC metadata inspection during this continuation confirmed:

| Observation | Actual value |
| --- | --- |
| Bridge account | `aist-franka`, UID/EUID 1000 |
| Effective capabilities | `CapEff: 0000000000000000` |
| PID 1 | `systemd`, UID 0 |
| `/proc/1/fd` | UID/GID 0, mode `0500` |
| Listing `/proc/1/fd` | `[Errno 13] Permission denied` |
| Dereferencing `/proc/1/fd/0` | `[Errno 13] Permission denied` |
| `/proc` mount | `rw,nosuid,nodev,noexec,relatime`; no `hidepid` option recorded |
| Current process security profile | `unconfined` |
| Yama `ptrace_scope` | `1`; recorded context, not established as the cause of the directory denial |
| Account device group | Group 20 already present; the prior device read/write-access check passed |

The root-owned `0500` directory alone explains why this ordinary account cannot enumerate PID 1's descriptors. Global visibility is therefore impractical under the current account policy. Changing serial-device group membership would not repair it. Merely relaxing Yama or procfs mount options would not establish the necessary directory and descriptor-access rights, and no such changes were attempted.

The new metadata evidence is [`ownership_continuation/nuc_policy.json`](../../outputs/fr3_milestone2b/ownership_continuation/nuc_policy.json). It was collected with:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 nuc-franka 'python3 -' \
  < outputs/fr3_milestone2b/ownership_continuation/inspect_policy.py \
  > outputs/fr3_milestone2b/ownership_continuation/nuc_policy.json
```

This inspection read process status, mount policy and permission metadata only. It neither opened nor queried the physical adapter.

### What each ownership mechanism establishes

The three mechanisms cover different conflicts; none is a substitute for the others under the current ownership contract.

| Mechanism | Protection supplied | Limitation |
| --- | --- | --- |
| Major/minor-keyed `flock` | Rejects another cooperating bridge using the same lock inode; aliases contend for the same device claim; close releases the bridge's claim | Advisory: a legacy serial driver can ignore it. Separate mount namespaces or replacing the lock inode can undermine coordination. It does not report terminal open-handle count |
| Procfs descriptor inspection | Detects already-open matching device handles in the inspected process view, including noncooperating programs | Denial is uncertainty, not absence. Namespace limits, process/FD churn and concurrent opens mean the scan is not an atomic global snapshot |
| `TIOCEXCL` | Blocks subsequent terminal opens by processes without `CAP_SYS_ADMIN`; retained in the private vendor patch before serial configuration | Does not evict earlier descriptors or stop I/O through them. Privileged actors can bypass subsequent-open exclusion |
| `TIOCGEXCL` | Reports the terminal's exclusive-mode flag | Is not an owner count or proof of sole ownership, and querying it requires a descriptor |

The advisory nature and close-based release of `flock` are documented by the [Linux flock manual](https://man7.org/linux/man-pages/man2/flock.2.html). The subsequent-open rule, privileged exception and exclusive-mode query are documented by the [Linux TIOCEXCL manual](https://man7.org/linux/man-pages/man2/TIOCEXCL.2const.html).

Detectable conflicts include a second cooperating bridge, an existing terminal handle visible in the checked process view, and an unprivileged later open after exclusive mode is set. Conflicts not reliably excluded include an inaccessible existing owner, an open racing the pre-open scan, a passed/inherited existing descriptor, and privileged intervention. Unrestricted privileged activity is outside the protection these mechanisms can supply.

The new PTY counterexample demonstrates the key distinction: an independently opened slave descriptor remains usable after the observation session sets `TIOCEXCL`. It can still write to the simulated master. That test deliberately omits the holder from the synthetic process view to demonstrate why incomplete inspection must not be accepted. It does not introduce a production option for ignoring owners.

Consequently, removing the scan, skipping root-owned processes, treating `EACCES` as success, or trusting an empty `fuser` result would weaken the current gate. No such correction was made. Complete inspection authority would repair observability, but would not by itself eliminate the existing scan-to-open race; operational control over legacy writers remains necessary.

### Precise administrator requirement

An administrator must approve a deployment authority model before the physical gate can be cleared. A normal unrestricted-root ROS bridge is not the proposed default. Two reviewable routes exist; neither was configured or implemented in this continuation.

1. **Constrained launch of the existing executable.** An administrator could review a dedicated identity/service with sufficient host-process visibility and tightly bounded authority. The access review must cover both directory read/search permissions and the procfs ptrace access check. Linux provides `CAP_DAC_READ_SEARCH` and `CAP_SYS_PTRACE` for these classes of access, respectively; they are broad powers, not capabilities scoped to this one serial device. `CAP_SYS_PTRACE` also permits cross-process memory operations, so it is not a read-only inspection grant. Exact sufficiency and containment depend on namespaces and security policy and require administrator verification. Granting these powers to the normal ROS process is not a device-scoped least-privilege solution; do not attach them to the node, an interpreter or an account-writable binary without separate security review. [Linux capabilities manual](https://man7.org/linux/man-pages/man7/capabilities.7.html).
2. **Narrow supervisor, preferred if privilege separation is required.** A separately reviewed, administrator-owned component could hold the required inspection authority while the ROS process remains ordinary. It must validate the fixed adapter identity, inspect the relevant host process view, coordinate the shared claim and kernel exclusion, and retain ownership across the session. A one-shot privileged scan followed by an unrelated bridge launch is insufficient: the snapshot goes stale and the current bridge would still fail its own scan. A real split would require either a reviewed already-open-descriptor handoff with vendor adoption rather than reopen, or keeping the existing vendor observation lifecycle inside the supervisor. Neither interface currently exists; implementing an unreviewed IPC/privilege path would exceed a minimal permission correction.

For either route, administrator acceptance must specify:

- The responsible identity and immutable executable, library/header identities and fixed configuration, deployed to administrator-managed paths rather than the current account-writable `/tmp` build.
- Complete relevant host PID/procfs visibility and sufficient directory/descriptor inspection rights; visibility errors remain failures. PID/mount/user namespaces and security policy must not hide potential owners.
- A common, stable lock inode/location for all cooperating owners. Do not rely on identical `/tmp` path strings across private mount namespaces as proof of shared locking.
- Sole operational control of legacy gripper access during the session, preserving flock and `TIOCEXCL`; no need to grant `CAP_SYS_ADMIN` to bypass exclusive mode.
- How ownership is retained, how failures and cleanup release only the bridge's resources, and which privileged actors remain outside the exclusivity guarantee.

No service definition, capability grant, sudo rule, procfs permission edit, udev/ACL change or supervisor was installed. A privileged inspection snapshot was not used to override the existing gate. **The administrative decision and deployment verification are still required; onsite presence alone cannot clear this blocker.**

### Focused verification

Only ownership tests were selected. No ROS node was started, no DDS qualification repeated, and none of the feedback-mapping, CRC, timeout, cache or message-schema tests was executed in this continuation.

Two cases were added to [`test_observation.cpp`](../../toolkits/robotiq_ros2/test/test_observation.cpp): a real permission-denial test within the synthetic procfs boundary, and the pre-existing-descriptor counterexample. The permission test restores permissions only on its own temporary fixture directory; no system permission is modified. Tests use freshly allocated PTYs and an ordinary workstation account (effective UID 1004). The deliberate owner-side test payload is the ASCII string `owner`, sent only between the PTY endpoints.

| Selected test | Result | Established behavior |
| --- | --- | --- |
| `ClaimRejectsAliasesExistingOwnersAndMissingProcVisibility` | PASS | Cooperating/alias conflict, visible existing owner and unavailable process view rejected |
| `DestructionAndFailedConnectReleaseClaimsWithoutCommands` | PASS | Reconnection/destruction and failed connection cleanup; no wire command on connection |
| `KernelExclusiveAccessAndFailedOpenReleaseClaim` | PASS | Existing kernel-exclusive claim rejects open; failed bridge open releases its claim; own cleanup permits reopen |
| `PermissionDeniedOwnerInspectionReleasesClaimBeforeOpen` — new | PASS | Actual directory denial blocks before serial open; a replacement can acquire the claim **before explicit disconnect**; repeated connection/cleanup after fixture restoration emits no request |
| `KernelExclusiveModeDoesNotRevokePreexistingDescriptor` — new | PASS | Exclusive mode is set but a prior holder can still perform PTY I/O; closing the bridge does not close another holder's descriptor |

The five-test selection passed **5/5 in 53 ms** on the workstation. The permission case was then strengthened to test release before explicit disconnect, rebuilt, and rechecked individually: **1/1 passed in 10 ms**. There were no failed or skipped selected cases. This final single-case rerun verifies the only assertion changed after the initial selection; it is not an unrelated suite repeat.

Actual filtered execution, after sourcing Jazzy and building only the test target, was:

```bash
source /opt/ros/jazzy/setup.bash
cmake --build /tmp/fr3_m2a_build_jazzy --target test_observation -j2
/tmp/fr3_m2a_build_jazzy/test_observation \
  --gtest_filter='ObservationTest.ClaimRejectsAliasesExistingOwnersAndMissingProcVisibility:ObservationTest.DestructionAndFailedConnectReleaseClaimsWithoutCommands:ObservationTest.KernelExclusiveAccessAndFailedOpenReleaseClaim:ObservationTest.PermissionDeniedOwnerInspectionReleasesClaimBeforeOpen:ObservationTest.KernelExclusiveModeDoesNotRevokePreexistingDescriptor' \
  --gtest_output=xml:outputs/fr3_milestone2b/ownership_continuation/ownership_tests.xml
```

The final permission-only execution used `--gtest_filter=ObservationTest.PermissionDeniedOwnerInspectionReleasesClaimBeforeOpen` and wrote `permission_final.xml`. [Initial log](../../outputs/fr3_milestone2b/ownership_continuation/ownership_tests.log), [initial XML](../../outputs/fr3_milestone2b/ownership_continuation/ownership_tests.xml), [final permission log](../../outputs/fr3_milestone2b/ownership_continuation/permission_final.log) and [final XML](../../outputs/fr3_milestone2b/ownership_continuation/permission_final.xml) retain the results. These selected tests do not call a connected `read()` path, activation/reset or position APIs; no FC03 request was sent, including on physical hardware. The existing disconnected-read rejection test makes no wire request.

### Change scope and stop decision

**IMPLEMENTED:** two focused ownership tests, a README pointer to this administrative requirement, and this report continuation. **Production ownership source changes: none.** `PortClaim`, `SerialObserver`, vendor patch/framing, feedback schema, ROS message contract, launch/configuration and disabled-by-default behavior are unchanged. No new actuation endpoint exists. No code or binary was deployed on the NUC.

**PASS:** root-cause inspection and selected simulated ownership tests. **BLOCKED:** administrator-approved authority/supervisor deployment and its ownership verification. **NOT TESTED:** physical exclusive claim, physical FC03 feedback, actual gripper behavior and privileged deployment. The previous 180/180 DDS result stands without repetition.

Milestone 2B remains **PARTIAL**. Per the request to stop when a correct resolution requires administrator configuration or operational approval, this continuation ends at that requirement. Proceed to neither physical diagnostics nor later integration until the administrative gate is resolved and the separate onsite authorization is obtained.

## 9. Administrator-Managed Deployment Review — 2026-10-27

**Milestone 2B remains PARTIAL.** A controlled deployment proposal is **IMPLEMENTED**, and unprivileged staging/static checks **PASS**. Installing and verifying the privileged profile is **BLOCKED** pending explicit authorization and an authenticated administrator session. Physical feedback remains **NOT TESTED**. No physical adapter was opened and no FC03 or actuation request was sent during this review.

The operator reports that `aist-franka` has `(ALL : ALL) ALL` sudo rights, can list `/proc/1/fd` with sudo, and found no owner with sudo fuser. These establish available administrative authority, not complete inspection or exclusive ownership. In this automation session, the bounded probe `sudo -n /usr/bin/id -u` returned exit 1, `sudo: a password is required`. No password handling, permission bypass or privileged development-binary execution was attempted. The subsequent limited-capability probes were therefore not executed.

### Proposed Resolution and Boundaries

The smallest change retaining the existing lifecycle is a constrained systemd launch of an administrator-owned observation-only executable. A new descriptor-handoff supervisor would require a larger vendor/lifecycle change. The proposed dedicated nonlogin identity `rlinf-robotiq` receives `CAP_DAC_READ_SEARCH` and `CAP_SYS_PTRACE` only inside this service; no file capabilities or unrestricted-root bridge are proposed. These powers address the directory and procfs inspection checks but are broad host inspection privileges. Their sufficiency under the actual security policy and their remaining exposure require administrator acceptance and verification.

The reviewed unit bounds runtime to 30 seconds, disables restart and boot enablement, keeps the host process view and shared `/tmp`, and preserves flock, the complete fail-closed procfs scan and the private vendor `TIOCEXCL`/cleanup path. It grants no `CAP_SYS_ADMIN`. A syscall deny list restricts direct ptrace/process-memory operations; it does not make inspection authority device-scoped. Strict device policy excludes serial adapters, and both fixed YAML and the command line set `enabled=false`. The deployment is intentionally suitable only for metadata/preflight and serial-disabled startup. Systemd documents these capability and execution controls in [systemd.exec](https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html), and device policy in [systemd.resource-control](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html). Static acceptance was checked against the actual NUC systemd 249 rather than inferred from the latest manual alone.

`ExecStartPre` runs a small fixed-device native helper using the frozen Milestone 2A `PortClaim`: stat the adapter, take the existing cooperative lock, inspect procfs, release the claim. It never constructs a vendor serial connection. Errors still fail closed. This preflight must succeed under the same identity/profile; an independently privileged snapshot cannot override the live gate. A successful scan only describes that instant. Existing scan-to-open races, hidden process namespaces, inherited/passed descriptors and privileged actors remain limitations. Operational exclusion of legacy writers is still necessary for any later physical session.

The original observation-only build is selected because the current development checkout contains subsequent offline command implementation. This task neither deploys that implementation nor recreates its removed report. Production bridge source, Modbus implementation, schema and DDS contract were not changed in this continuation.

### Source, Build and Environment Evidence

Read-only inspection identified NUC systemd `249.11-0ubuntu3.23`, unified cgroup v2 and seccomp support. The root-owned native system utilities and `/opt/ros/humble/lib` directory metadata were inspected. A complete trust audit of all resolved runtime libraries, ancestors and ACLs remains an administrator prerequisite.

| Artifact/check | Identity or result |
| --- | --- |
| Original observation-only node | `/tmp/rlinf_robotiq_m2a_3krbaoaq/build/robotiq_observation`; UID 1000, mode 0775; **ineligible for privileged execution** |
| Original node SHA-256 | `b005b04bf7e2ac8484d3e412d22df1aac56aa1875d1419abdefe633176a40ce8` |
| Original ELF RUNPATH | `/opt/ros/humble/lib:`; empty final component permits working-directory library search |
| Ordinary-user candidate staging | `/tmp/fr3_m2b_admin_review_2eqvpi33/release` |
| Staged installed node SHA-256 | `3cd0c07c9621e7683038e1e489838209c1218d40a5322dcf01f85c625490ce57` |
| Staged ELF RPATH/RUNPATH | Neither present after CMake installation |
| Preflight executable SHA-256 | `23c01f3477b06133d576dc3b7caf15145d7bae7285682aa9ec277321130b8dc8` |
| Vendor provenance | Frozen 2A private patched archives; source/header and build identities recorded in inspection/staging evidence |
| Proposed installed release | `/opt/rlinf/robotiq-observation/m2a-20261027` |
| Proposed fixed configuration | `/etc/rlinf/robotiq-observation/observation-disabled.yaml` |
| Proposed unit | `/etc/systemd/system/rlinf-robotiq-observation.service` |

The build-only install removed the unsafe RUNPATH; the changed executable hash reflects this installation transformation. The original `/tmp` executable was not patched in place or run with privileges. No files were installed in privileged paths.

Evidence is under [`outputs/fr3_milestone2b/admin_deployment`](../../outputs/fr3_milestone2b/admin_deployment): `inspection.json` includes account, native-tool metadata, frozen node source and package/vendor hashes; `staging.json` records commands, ELF output, dependency resolution and staged manifest; `proposal_hashes.json` pins proposal files; `unit_verify_final.json` records final static validation. These are local generated artifacts and may not be present in another checkout.

### Actual Focused Commands and Results

The ordinary-user inspection and staging harnesses were invoked through `ssh -o BatchMode=yes -o ConnectTimeout=10 nuc-franka 'python3 -'`, with their local scripts supplied on stdin. Exact subprocess arguments and outputs are retained in the evidence JSON and harness sources. The principal NUC commands were:

```bash
sudo -n /usr/bin/id -u
/usr/bin/cmake --install /tmp/rlinf_robotiq_m2a_3krbaoaq/build --prefix /tmp/fr3_m2b_admin_review_2eqvpi33/release
/usr/bin/g++ -std=c++17 -Wall -Wextra -Werror /tmp/fr3_m2b_admin_review_2eqvpi33/proposal/port_claim_check.cpp -I/tmp/rlinf_robotiq_m2a_3krbaoaq/robotiq_ros2/include -I/tmp/rlinf_robotiq_m2a_3krbaoaq/build/vendor/include /tmp/rlinf_robotiq_m2a_3krbaoaq/build/libobservation_core.a /tmp/rlinf_robotiq_m2a_3krbaoaq/build/librobotiq_vendor.a -lboost_system -lpthread -o /tmp/fr3_m2b_admin_review_2eqvpi33/release/lib/rlinf_robotiq_observation/port-claim-check
/usr/bin/readelf -d /tmp/fr3_m2b_admin_review_2eqvpi33/release/lib/rlinf_robotiq_observation/robotiq_observation
/usr/bin/env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/opt/ros/humble/lib /usr/bin/ldd /tmp/fr3_m2b_admin_review_2eqvpi33/release/lib/rlinf_robotiq_observation/robotiq_observation
/usr/bin/systemd-analyze verify --man=no /tmp/fr3_m2b_admin_review_2eqvpi33/rlinf-robotiq-observation.service
sha256sum /tmp/fr3_m2b_admin_review_2eqvpi33/proposal/rlinf-robotiq-observation.service /tmp/fr3_m2b_admin_review_2eqvpi33/proposal/observation-disabled.yaml
```

The static verifier used a **path-only variant** pointing executable/configuration paths at staging; it did not load or start a service. An initial attempted `--root` verification was unsupported by this systemd version. An initial condition-key typo was corrected to `ConditionFileIsExecutable`, and final verification returned zero with no warning about the proposed unit. Two unrelated host-unit diagnostics remain: permission denied opening `netplan-ovs-cleanup.service`, and the existing snapd unit's unsupported `RestartMode` key. These are retained verbatim in the evidence; exit zero is not a privileged runtime qualification.

| Focused behavior | Category | Evidence/limit |
| --- | --- | --- |
| Frozen observation build identity checked | PASS | Original hash matched; account-writable executable excluded from privileged execution |
| Ordinary-user staging and helper compilation | PASS | CMake install and strict-warning C++ compile returned zero |
| Installed-candidate RUNPATH and dependency resolution | PASS | Empty RUNPATH removed; ordinary-user `ldd` resolved dependencies |
| NUC systemd property/static verification | PASS | Final path-only variant returned zero; unrelated host warnings above |
| Staged service/config match reviewed hashes | PASS | SSH SHA-256 check matched both proposal identities |
| Root-owned installation and controlled profile | IMPLEMENTED | Reviewable proposal files only; not installed |
| Noninteractive sudo inspection | BLOCKED | Password required; no capability-limited scan performed |
| Same-profile complete ownership preflight | NOT TESTED | Helper compiled but never executed against the adapter metadata |
| Privileged startup, device-policy enforcement and cleanup | NOT TESTED | Requires authorized installation and authenticated administrator verification |
| Physical `TIOCEXCL`, FC03, feedback/fault/rate behavior | NOT TESTED | No physical open or request |
| DDS and existing CPU suites | NOT TESTED | Deliberately not repeated; previous 180/180 DDS and ownership results stand |

### Changed Files and Authorization Gate

New review assets are [`deployment/port_claim_check.cpp`](../../toolkits/robotiq_ros2/deployment/port_claim_check.cpp), [`deployment/observation-disabled.yaml`](../../toolkits/robotiq_ros2/deployment/observation-disabled.yaml), [`deployment/rlinf-robotiq-observation.service`](../../toolkits/robotiq_ros2/deployment/rlinf-robotiq-observation.service), and [`deployment/INSTALL_PLAN.md`](../../toolkits/robotiq_ros2/deployment/INSTALL_PLAN.md), plus this report update. The plan gives exact installation, installed-hash checks, trust review, bounded startup and rollback commands. No privileged service, account, permission, installed binary, FR3 configuration or device ownership was modified. Neither the helper nor the bridge was launched in this continuation.

**Can the gate be satisfied?** The proposed authority model can supply the missing Linux inspection rights in principle while preserving all three ownership checks. It has not yet demonstrated complete visibility under the actual service profile, so the blocker is not cleared and physical qualification is not ready. Administrator approval must accept the host-inspection exposure, root-owned artifacts/dependencies, shared lock handling and serial-denying service policy. After explicit installation/startup authorization, an authenticated administrator must verify those properties and obtain a passing same-profile preflight. Any denial or detected owner remains a blocker.

Stop here for authorization to install the exact reviewed files and perform bounded **serial-disabled** startup. Physical access, FC03 testing, actuation and later milestones remain outside that authorization. Milestone 2B cannot be called fully hardware-qualified until a separate authorized physical read-only session meets its original completion criteria.
