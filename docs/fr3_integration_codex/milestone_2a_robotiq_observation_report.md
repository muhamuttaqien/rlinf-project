# Milestone 2A — Robotiq ROS 2 observation bridge

Milestone 2A implements an observation-only ROS 2 bridge using the existing NUC Robotiq C++ implementation. The node, launch file, configuration, observation schema and focused tests are available for review. **Both Jazzy and Humble builds pass all nine CPU tests; physical feedback and operational node behavior remain unverified.** No gripper actuation or later RLInf integration was implemented.

Date: **2026-10-27 JST**. Integration checkout: `/home/muttaqien/Desktop/AIST-Tokyo/gamma/rlinf-project`. The [Milestone 1 audit](milestone_1_system_interface_audit.md), [hardware baseline](../../../fr3-system-audit/docs/AUDIT_BASELINE.md), [baseline configuration](../../../fr3-system-audit/config/system.json) and relevant RLInf interfaces were read before implementation. The baseline's latest saved full report was IDLE; it is not a current physical-readiness verdict.

## 1. Implementation and reuse

The bridge is a standalone `ament_cmake` package under [`toolkits/robotiq_ros2`](../../toolkits/robotiq_ros2/README.md). This keeps ROS Humble dependencies and hardware ownership separate from RLInf's Python/worker lifecycle. It does not register or instantiate an RLInf robot, arm, environment or gripper.

The existing [`RobotiqGripper`](../../rlinf/robotics/parts/end_effectors/grippers/robotiq.py) activates the gripper from `_open()`. [`BaseGripper`](../../rlinf/robotics/parts/end_effectors/grippers/base.py) represents metric width and includes commanding/reset behavior; its state contract cannot represent this uncalibrated observation source faithfully. [`FrankaRobot`](../../rlinf/robotics/robots/franka.py) supports composing separate devices, but no such integration is needed in 2A. [`FrankaROSArm`](../../rlinf/robotics/parts/arms/franka_ros.py) is a ROS 1 controller-launching path. Those upstream interfaces were reused as design references, not executed or modified.

The bridge compiles the audited NUC library's original implementation files rather than writing another Modbus stack. A narrow patch adds missing observation guarantees in a disposable build copy. No deployed source file or installed library is overwritten. The bridge has a read-only facade exposing connection, FC03 feedback and disconnect; the existing vendor activation/motion methods are never called and have no ROS endpoint.

Serial polling is disabled in the default configuration. When explicitly enabled in a later reviewed launch, one worker obtains the port claim, opens a fresh library object, reads feedback and closes/retries on communication failure. A separate timer assembles status at 10 Hz without waiting for serial reads. The configured settings are 115200 8N1, fixed library slave 9, 5 Hz polling, 100 ms response deadline, 0.5 s freshness threshold and 1 s reconnect delay. Parameters are read-only after construction.

## 2. Exact library inspection

The NUC source tree `/home/aist-franka/robotiq/robotiq-gripper-interface-main` was inspected over SSH using file reads only. A read-only source snapshot was copied to workstation `/tmp/fr3_m2a_library` for compilation/tests. Eight compiled source/header identities are pinned in [vendor.cmake](../../toolkits/robotiq_ros2/cmake/vendor.cmake), with build failure on any mismatch.

Key original source hashes are:

| Original NUC source | SHA-256 |
| --- | --- |
| `src/robotiq_gripper_interface.cc` | `f156ab602ba9d766cdb4812ebd93bbb6e133365feea8c84c54d586798b7c9979` |
| `src/helpers.cc` | `2a212b59cd8be97d8c722439b0a83eea9900ac18747b210b5313742a9cf17570` |
| `src/timeout_reader.cc` | `dc2b2369e2b2f0e58595771968b5baeb0be643d0122fbfbd3f5bdceecc2595d3` |

The connection and feedback paths support reuse, but the inspected API requires a small extension to report failure correctly.

| Inspected behavior | Finding and resulting implementation |
| --- | --- |
| Construction | Allocates Boost.Asio state; does not open serial or activate the gripper |
| `connect()` | Opens/configures serial only; no activation/reset write. Originally lacked kernel exclusivity and could retain `is_connected=true` after closing an earlier connection and failing to reopen |
| `get_feedback()` | Sends fixed request `090307D00003040E` (slave 9, FC03, three holding registers at 0x07D0); decodes measured/requested positions, normalized current and detailed status |
| Failed feedback | Original API returns default numeric fields without a validity flag and with uninitialized detailed status. The overlay adds `valid=false` by default, initializes status and marks valid only after a correctly sized/prefixed response |
| Raw fields | Original raw measured/requested position is retained. Overlay also preserves full fault byte and raw current; no metric-width conversion is added |
| Finger status | The original named `ACTIVATION_COMPLETE` enum value was implicitly 2, although decoding returns raw gSTA 3. The private overlay assigns that name value 3 |
| Echo/CRC | Existing `extract_response` discards an exact local request echo, checks slave/function, derives response length, checks CRC and resynchronizes after corrupt bytes. These algorithms are reused unchanged |
| Exception/invalid response | Existing helper returns no usable feedback for Modbus exception/malformed/timeout cases. The bridge reports common communication failure, not an invented detailed transport diagnosis |
| Timeout | Original timeout is per character, with up to 256 bytes read. Overlay uses one steady-clock response-read deadline and passes remaining time to the existing `TimeoutReader` |
| Destruction | Original destructor closes serial without reset/activation commands; overlay uses nonthrowing close and releases `TIOCEXCL`. Partial connection failures destroy a fresh object and release the external claim |
| Build rules | Original CMake writes its library into the vendor source tree and builds actuation examples. The bridge bypasses those build rules, compiles only the three audited implementation files into a private static library and never builds/runs those examples |

The [build-only patch](../../toolkits/robotiq_ros2/cmake/observation.patch) changes two headers and two implementation files in the copy. It extends the connection signature with an exclusivity option, adds validity/raw fields, fixes the named enum, closes safely, and bounds response reception. Its ABI is private to this build; it must not be mixed with the old installed `.so` or old headers. No replacement CRC, framing, echo-filtering or position-decoding implementation was added.

## 3. Observation contract

The full [schema and deployment notes](../../toolkits/robotiq_ros2/README.md) are the interface reference. The default topic is `/robotiq_observation/state`, type `diagnostic_msgs/msg/DiagnosticArray`, with one status entry named `robotiq/observation`. QoS is reliable, volatile, keep-last 1. Node-name/topic remapping remains standard ROS behavior.

The payload preserves measured `position` and `raw_position` (gPO), `commanded_position` and `raw_commanded_position` (gPR echo), normalized/raw/register-derived ampere current, gACT/gGTO/gSTA/gOBJ, full fault byte, low-nibble fault code and decoded fault name. Normalized positions mean closure fraction, **0 open and 1 closed**. The commanded-position fields report the device's request echo; this bridge never sends a position request.

Standard diagnostic key/value fields avoid a custom message build across Jazzy/Humble and support detailed validity/fault metadata. No `JointState` topic is emitted: a normalized register position is neither a calibrated jaw width in metres nor a known joint angle. Consumers must follow schema version `1`, not assume arbitrary diagnostic keys map directly into RLInf's metric gripper interface.

Publication `header.stamp` is NUC UNIX wall time, not ROS simulation time or a sensor stamp. `last_success_unix_ns`, `last_success_monotonic_ns`, `request_started_unix_ns`, `sequence` and `age_s` retain explicit timing/provenance. Source measurement timestamps are unavailable. Steady-clock ages are local to the NUC; workstation consumers must additionally monitor their own receipt age and cannot compare host monotonic clocks.

| State | Published validity and data |
| --- | --- |
| `missing` | No successful sample yet, including default-disabled operation; measurement fields omitted, communication invalid |
| `fresh` | Latest completed attempt valid and its receipt age within threshold; all measurement fields present |
| `stale` | Latest successful receipt has aged out or monotonic time moved backwards; measurement fields omitted, communication invalid |
| `failed` | Connection or communication attempt failed; immediate invalidation even if the previous success was recent; measurement fields omitted |

Historical last-success times and sequence remain available on failure, without presenting old register values as current measurements. A valid response with device fault `0x09` remains communication-valid and reports `COMM_TIMEOUT` with diagnostic WARN; transport failure is diagnostic ERROR. Missing/stale status is STALE. Fault-free fresh feedback is OK. No fault clearing or automatic activation occurs.

## 4. Ownership and lifecycle

The serial claim uses the device's major/minor number, so different path aliases contend for the same flock under `/tmp`. A pre-open `/proc/*/fd` scan rejects existing serial owners and fails closed if process visibility is incomplete. The private library requests `TIOCEXCL` before baud/line configuration to prevent later unprivileged opens. Cleanup closes only the bridge's own session, then releases the claim; repeated connect/disconnect are idempotent.

Procfs/ptrace permission policy is a real deployment prerequisite. An ordinary account may be unable to inspect another user's descriptors, including root-owned processes. In that case the bridge will publish failed status without opening serial. No permission bypass, automatic `sudo`, process termination or service modification was introduced. The intended NUC account's complete ownership visibility remains NOT TESTED; resolve this with the responsible administrator before an enabled launch.

The test suite substitutes a temporary procfs view at the OS boundary and uses fresh PTY pairs for every serial session. The real library's framing, timeout and decode paths are exercised. A separate test checks kernel exclusivity and successful release after failed open/destruction. This establishes the process-edge behavior in simulation, not exclusive access to the physical USB adapter under all host conditions.

The procfs precheck and kernel request are not atomic with unrelated legacy software's open, and privileged opens can bypass exclusivity. Operational coordination and stable-device identity remain necessary. No arbitrary software-only lock can certify safety against an independent privileged hardware process. The bridge never stops or reconfigures another owner.

The response-read deadline begins after the original library's synchronous FC03 write. It is not a hard upper bound on OS open/write, procfs traversal, scheduling or shutdown. FC03 has no transaction/source timestamp; receipt freshness does not prove acquisition age or exclude a delayed device response. These limits are stated rather than silently treating host receipt as sensor time.

## 5. Model and calibration status

The existing [hardware qualification](../experiments/openpi_recap_franka/milestone_1b_1_report.md) records the label “Robotiq 2F Adaptive Gripper C-49010” and considers 2F-85 likely, with the exact 85/140 variant unconfirmed. Neither FC03 register feedback nor the library's historical 2F-85 test description identifies the mounted model. No additional physical inspection or hardware query was performed to claim otherwise.

The implementation publishes `model=unconfirmed_2F_variant` and `metric_width_available=false`. Before any metric-width integration, obtain the full unit/model identification and fingertip configuration, review the applicable manufacturer calibration, and qualify jaw opening versus gPO with units and uncertainty. Do not substitute the Python driver's default `max_width=0.085` m or the C++ library's arbitrary alpha/beta scale for measured calibration. Gripper calibration exercises and FR3 TCP/load calibration remain outside 2A.

## 6. Changed files

Only new bridge/report files and a scoped ignore exception were added during 2A. Existing Milestone 1 and startup-fix artifacts were retained.

| File | Purpose |
| --- | --- |
| [`.gitignore`](../../.gitignore) | Allow this package's `CMakeLists.txt` through the repository-wide `*.txt` ignore rule |
| [`CMakeLists.txt`](../../toolkits/robotiq_ros2/CMakeLists.txt) | Standalone/ament build, private static vendor/core/message targets, ROS executable and CPU test target |
| [`package.xml`](../../toolkits/robotiq_ros2/package.xml) | ROS package/dependency metadata |
| [`cmake/vendor.cmake`](../../toolkits/robotiq_ros2/cmake/vendor.cmake) | Eight audited hashes, disposable copy and fail-on-drift patch application |
| [`cmake/observation.patch`](../../toolkits/robotiq_ros2/cmake/observation.patch) | Minimal private observation/lifecycle extensions to the existing library |
| [`include/.../observation.hpp`](../../toolkits/robotiq_ros2/include/rlinf_robotiq_observation/observation.hpp) | Serial claim/session and observation-cache contracts |
| [`src/observation.cpp`](../../toolkits/robotiq_ros2/src/observation.cpp) | Exclusive observation lifecycle and freshness/cache handling |
| [`include/.../message.hpp`](../../toolkits/robotiq_ros2/include/rlinf_robotiq_observation/message.hpp), [`src/message.cpp`](../../toolkits/robotiq_ros2/src/message.cpp) | Standard diagnostic-message mapping |
| [`src/node.cpp`](../../toolkits/robotiq_ros2/src/node.cpp) | Disabled-by-default ROS publisher, worker, reconnect and cleanup |
| [`launch/observation.launch.py`](../../toolkits/robotiq_ros2/launch/observation.launch.py), [`config/observation.yaml`](../../toolkits/robotiq_ros2/config/observation.yaml) | Isolated launch and static defaults |
| [`test/test_observation.cpp`](../../toolkits/robotiq_ros2/test/test_observation.cpp) | Nine focused CPU tests using real patched library sources and simulated serial/process boundaries |
| [`README.md`](../../toolkits/robotiq_ros2/README.md) | Schema, lifecycle, calibration and deployment/dependency notes |
| This report | Implementation/verification record and 2B recommendations |

No FR3 controller, startup file, hardware workspace, installed robot binary, system service or RLInf operational driver was changed. Temporary source/build/test files were written only under workstation/NUC `/tmp`. No installation was performed; the node executable was compiled but never launched.

## 7. Verification record

Verification exercised only CPU and simulated serial behavior. No physical serial port was opened and no ROS node or controller was started by the tests.

| Verification | Status | Evidence |
| --- | --- | --- |
| Jazzy build, workstation `pca03395` | PASS | GCC 13.3, ROS `rclcpp` 28.1.21, `diagnostic_msgs` 5.3.8; all targets built in `/tmp/fr3_m2a_build_jazzy` |
| Humble build, NUC | PASS | GCC 11.4, Python 3.10.12, `rclcpp` 16.0.19, `diagnostic_msgs` 4.9.2; isolated `/tmp/rlinf_robotiq_m2a_3krbaoaq/build` |
| Final Jazzy CPU suite | PASS | **9/9**, approximately 0.32 s; one CTest target containing nine GTests |
| Final Humble CPU suite | PASS | **9/9**, approximately 0.31 s; `/tmp/rlinf_robotiq_m2a_3krbaoaq/verification_final.log` |
| Feedback fields, echo and fault | PASS | Distinct measured raw 225/request echo 18, normalized positions, raw/current conversion, activation/action/finger/object bits, full fault byte `0xA9`, decoded low-nibble `COMM_TIMEOUT` |
| Invalid communication | PASS | Bad CRC, wrong slave, wrong function, wrong byte count, Modbus exception and no response rejected; noise/corrupt-frame resynchronization accepts a later valid reply |
| Deadline, freshness and recovery | PASS | Slow partial reply bounded; fresh → stale/failed/missing behavior, backward monotonic time, explicit clocks, successful reconnection and sequence advancement |
| Lifecycle and ownership | PASS | Repeated connect/disconnect; no activation writes; alias contention, existing owner, incomplete process view, failed open, TIOCEXCL rejection and destructor claim release on PTYs |
| Standard message mapping | PASS | Publication stamp, status severity, raw fields/faults, absence of unavailable measurements, historical last-success retention |
| Audited-source drift guard | PASS | Deliberately modified temporary `constants.h` rejected during configure with `Robotiq source changed` |
| Humble/Jazzy message definitions | PASS | Installed `DiagnosticArray.msg`, `DiagnosticStatus.msg` and `KeyValue.msg` SHA-256 values match byte-for-byte across hosts |
| Launch Python lint/format | PASS | Ruff check with preview and format check on the launch file |
| Final patch and document checks | PASS | Context-free patch applies with zero fuzz and produces byte-identical tested vendor sources; 31 local README/report links resolve; XML, launch syntax and new-file whitespace checks pass |
| Physical model identification/calibrated width | NOT TESTED | Existing evidence remains insufficient; no metric width is published |
| Actual NUC procfs ownership visibility and physical USB exclusive access | NOT TESTED | CPU tests simulate the OS process view and serial endpoint |
| Operational ROS startup, timers/reconnect under real scheduling, shutdown and delivered rate | NOT TESTED | No node executable was run; tests exercise core lifecycle and message assembly, not ROS runtime composition |
| Humble → Jazzy DDS reception | NOT TESTED | Build/schema compatibility is established separately from network/runtime delivery |
| Physical serial feedback, activation, reset, opening/closing, grasp, FR3 motion | NOT TESTED | Explicitly excluded; no such calls/commands were made |

The same CPU suite was run on each host with:

```bash
source /opt/ros/humble/setup.bash  # jazzy on the workstation
cmake -S toolkits/robotiq_ros2 -B /tmp/robotiq-observation-build \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build /tmp/robotiq-observation-build -j2
ctest --test-dir /tmp/robotiq-observation-build --output-on-failure
```

The recorded executions used the isolated paths above; workstation configure additionally selected `/tmp/fr3_m2a_library` as `ROBOTIQ_SOURCE_DIR`. Humble used the original NUC source path read-only. GTest summary output confirmed nine cases on each host. Tests assert the exact outbound FC03 request and exercise the real C++ implementation; no alternate Modbus stack supplies the expected behavior.

Two development build failures were corrected: mixed keyword/plain CMake link signatures with `ament_target_dependencies`, and a rebuild attempted without sourcing the ROS environment. Final builds and tests were rerun with the environment sourced. An intermediate CTest invocation used the prior executable after the failed rebuild and is not the final verification evidence. Original vendor Boost global-placeholder deprecation notes remain; no compiler diagnostic was suppressed. The initial Humble configure also reported an unused `PYTHON_EXECUTABLE` option; the documented build uses `Python3_EXECUTABLE`.

Public ROS documentation pages were inaccessible during lookup; installed Humble/Jazzy message definitions and actual compilation were used as the schema evidence. This experiment branch has no `.github/workflows/ci-tests.yml`; CI path/job coverage for the new C++ package must be supplied before upstream submission. Full repository/GPU suites and EN/ZH Sphinx builds were not run: no RLInf runtime or published RST trees changed. Standalone report/README references and new-file whitespace were checked separately.

Upstream overlap searches for `robotiq` returned no open PR and the existing [#974](https://github.com/RLinf/RLinf/issues/974) and [#1012](https://github.com/RLinf/RLinf/issues/1012) issues. No upstream content, commit or PR was created or changed. Prepared with an AI assistant.

## 8. Remaining risks and Milestone 2B recommendations

Milestone 2A supplies reviewable implementation and simulated evidence. It does not establish continuous physical observation or clearance to run operational hardware.

Before a later **observation-qualification** milestone, review the private library patch, confirm the intended NUC account's procfs visibility, establish sole serial ownership with the responsible operator, and identify the mounted model from product records. Review whether a reviewed enabled observation session is permitted under the current hardware state. Keep the Franka controller, its startup configuration and FCI owner untouched.

If that review authorizes 2B observation work, qualify one bounded FC03-only session, recording communication faults, sequence/timing, disconnect/reconnect and cleanup, then validate DDS reception and workstation-side freshness with the existing Humble/Jazzy/Fast DDS baseline. Verify worker/timer/shutdown behavior separately from the already-passing core tests. Treat device fault codes as reported evidence; do not clear faults or activate the gripper to make a test pass.

Metric-width calibration, actuation endpoints, policy commands and an RLInf `EndEffector`/robot integration require separate reviewed milestones. The original library's raw-feedback model and host timestamps cannot resolve source acquisition age or calibrated width by themselves. Do not silently promote the diagnostic schema to the metric `BaseGripper` contract.

**Stop record: Milestone 2A implementation and CPU verification are complete. No operational hardware process, physical serial read, activation/reset/motion, FR3 change, policy action path or Milestone 2B work was performed.**
