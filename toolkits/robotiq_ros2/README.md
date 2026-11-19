# Robotiq ROS 2 observation bridge

This package publishes measured Robotiq feedback through standard ROS 2 messages while reusing the existing NUC C++ library. Optional normalized position commands share the same serial owner and require an explicit owner lease. Serial polling and commands are both disabled by default. Startup, connection, reconnection, observation and shutdown never activate, reset or move the gripper. The position path has offline verification only.

The implementation targets ROS 2 Humble on the NUC and builds against Jazzy on the workstation. Milestone 2B qualified the standard observation topic across hosts with serial disabled. Physical feedback/commands and command-service DDS communication remain unqualified. The [Milestone 2A report](../../docs/fr3_integration_codex/milestone_2a_robotiq_observation_report.md) records the original observation implementation.

## Build and dependencies

Build from this package with C++17, CMake 3.16+, Boost.System, pthreads, `patch`, ROS 2 `ament_cmake`, `rclcpp`, `rcl_interfaces` and `diagnostic_msgs`. Launch-file dependencies are `launch_ros` and `ament_index_python`. CPU tests additionally use GTest and Linux PTYs (`libutil`). These dependencies were already available for the recorded workstation/NUC builds; no packages were installed by this milestone.

The optional command interfaces add `builtin_interfaces`, `rosidl_default_generators` at build time and `rosidl_default_runtime` at runtime. Matching generated service types are required for command clients on either host; observation subscribers still need only standard `diagnostic_msgs`. Deploy the package and its generated shared type-support libraries together, not a bare copied executable. CTest configures the build-tree library lookup for offline service serialization; a direct test invocation also needs the build directory on `LD_LIBRARY_PATH`.

`ROBOTIQ_SOURCE_DIR` defaults to `/home/aist-franka/robotiq/robotiq-gripper-interface-main`. CMake checks eight source/header hashes in [vendor.cmake](cmake/vendor.cmake), copies those files into the build directory, applies [observation.patch](cmake/observation.patch), and compiles a private static library. It neither invokes the vendor's examples/install rules nor modifies or loads the existing installed `.so`. Changed source hashes cause a build failure pending review. The patch changes the private ABI, so its headers must only be used with this build's library.

For an isolated NUC build, use the package directory in a staging checkout and place all outputs under `/tmp`:

```bash
source /opt/ros/humble/setup.bash
cmake -S toolkits/robotiq_ros2 -B /tmp/robotiq-observation-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build /tmp/robotiq-observation-build -j2
ctest --test-dir /tmp/robotiq-observation-build --output-on-failure
```

This builds the node and runs only simulated CPU tests. Tests allocate fresh PTYs, use a simulated procfs view for ownership cases, and never open `/dev/serial/*` or `/dev/ttyUSB*`. They do not initialize ROS or launch any node. With a copied vendor source tree on another machine, supply `-DROBOTIQ_SOURCE_DIR=/path/to/audited/source`. A non-ROS build of the core/tests is available with `-DROBOTIQ_WITH_ROS=OFF`; the message test then requires the ROS build.

An optional staging install is `cmake --install /tmp/robotiq-observation-build --prefix /tmp/robotiq-observation-install`. The package is also an `ament_cmake` package for a dedicated colcon workspace. Use isolated build/install bases, not either Franka workspace. Installation and operational launch were not performed in this milestone.

## Configuration and ownership

The supplied [configuration](config/observation.yaml) has `enabled: false`. Its other settings are the stable serial by-id path, 115200 8N1 (fixed by the audited library), slave 9 (the library's fixed request), 100 ms response-read deadline, 5 Hz polling, 0.5 s freshness threshold and 1 s reconnect delay. The deadline covers the response read following the library's synchronous request write; it is not a hard bound on OS opens, writes, scheduling or shutdown.

The [launch file](launch/observation.launch.py) accepts a `config` argument and launches only this node. For later reviewed deployment, `ros2 launch rlinf_robotiq_observation observation.launch.py` loads the disabled configuration. Enabling physical polling requires a separately reviewed parameter file. Parameters are read-only after node creation. Command services and command-status publication are absent unless `commands_enabled` is explicitly configured true; that setting also requires serial enable and a nonempty configured owner. The ordinary ROS parameter services cannot enable commands after startup.

Before any later enabled launch, confirm the original Robotiq/YUBI processes are stopped by their responsible owner. The bridge claims a major/minor-device lock under `/tmp`, checks `/proc/*/fd` for existing owners, and requests kernel `TIOCEXCL` before serial configuration. Aliases share the same lock. It refuses access when ownership inspection is incomplete, the port is already open, or another bridge holds the lock. It never stops another process. Lock files remain after shutdown; the kernel releases the flock on close/process exit, so a leftover filename is not a stale owner.

**Procfs visibility is a deployment prerequisite.** Normal Linux ptrace/procfs permissions can prevent an ordinary account from inspecting other users' descriptors, including root processes. The bridge then reports failure and leaves the port unopened. This policy was not relaxed or worked around; an administrator must review whether the intended account has sufficient inspection visibility. No automatic privilege escalation or service changes are provided. The tests substitute only that OS process-visibility boundary, not the vendor protocol implementation.

The NUC account was confirmed to lack that visibility during Milestone 2B. The [ownership continuation](../../docs/fr3_integration_codex/milestone_2b_robotiq_hardware_dds_report.md#8-ownership-continuation--administrative-decision-required) records the exact administrative requirement and focused PTY evidence. An empty `fuser` result or successful `TIOCEXCL` is insufficient: kernel exclusive mode blocks later unprivileged opens but leaves earlier descriptors usable. A separately privileged inspection snapshot cannot replace the bridge's lifetime claim. No approved supervisor or privilege configuration is supplied by this package; physical qualification remains blocked pending administrator review.

The preflight scan and TIOCEXCL cannot provide an atomic guarantee against an unrelated legacy process racing the open, nor against privileged processes able to bypass exclusivity. Operational ownership coordination is still required. Protect the stable device path from reassignment during connection. After read/connection failure, the worker closes its own session, releases its claim and waits before reconnecting with a fresh library object; it never activates or resets the device. Cleanup is idempotent. Receipt freshness is evaluated by the independent publication timer while a read is outstanding.

Preserve the existing ROS distributions and DDS baseline (`ROS_DOMAIN_ID=70`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`) for any later cross-machine qualification. This package does not set discovery ranges, change ROS domains, launch a daemon, or alter Franka controller/FCI ownership.

## Observation message schema, version 1

The observation topic is `/robotiq_observation/state` with the default node name. Its type is `diagnostic_msgs/msg/DiagnosticArray`; QoS is reliable, volatile, keep-last 1. The 10 Hz publisher is independent of the configured serial polling interval. Receivers may request reliable or best-effort delivery and must enforce their own local receipt-age limit if the publisher or network stops. The optional command interface below does not change this schema or QoS.

Each array contains exactly one `DiagnosticStatus`: name `robotiq/observation`, hardware ID equal to the configured serial path. `header.stamp` is publication UNIX wall time from `system_clock` on the NUC, encoded as ROS seconds/nanoseconds; it deliberately does not use simulation time. Frame ID is empty because these are gripper registers, not spatial coordinates. The header is **not** measurement/exposure time. All keys below are `DiagnosticStatus.values` string key/value pairs; decimals use C++ precision 17, booleans are `true`/`false`, integers are base 10.

| Key | Meaning |
| --- | --- |
| `schema_version` | `1` |
| `state` | `missing`, `fresh`, `stale` or `failed` |
| `communication_valid` | True only when a successful response is still fresh and no subsequent attempt failed |
| `last_attempt_valid` | Whether the latest completed connection/read attempt succeeded; can remain true after a sample becomes stale |
| `sequence` | Number of successful reads in this node lifetime; publication does not increment it |
| `last_success_unix_ns` | Host wall receipt time after the last valid library read; `0` before any success |
| `last_success_monotonic_ns` | Host steady-clock receipt time after that read; `0` before success, meaningful only within this host/boot |
| `request_started_unix_ns` | Wall time before the last successful read request; retained with its historical receipt timestamps |
| `age_s` | Age since last success using the NUC steady clock, or `-1` if none; a negative clock discontinuity invalidates the sample |
| `position_units` | `normalized_closure_0_open_1_closed` |
| `model` | `unconfirmed_2F_variant` |
| `metric_width_available` | Always `false` |

The following keys are present **only in fresh samples**. Their omission on missing/stale/failed communication means unavailable, not zero. Historical receipt times/sequence remain available for diagnostics, without republishing old register values as current measurements.

| Key | Meaning |
| --- | --- |
| `position`, `raw_position` | Measured gPO: raw byte divided by 255, and unchanged raw byte |
| `commanded_position`, `raw_commanded_position` | gPR requested-position echo: raw byte divided by 255, and unchanged raw byte; these are not commands issued by this bridge |
| `current_normalized`, `raw_current`, `current_ampere` | Library current (gCU/255), original gCU byte, and register current in amperes (gCU × 0.01); not force/grasp estimation |
| `activation_status` | gACT: 0 not activated, 1 activation requested; gACT alone does not prove completed activation |
| `action_status` | gGTO: 0 stopped, 1 go-to requested |
| `finger_status` | Raw gSTA: 0 reset, 1 activation in progress, 3 complete; value 2 is reserved |
| `object_status` | Raw gOBJ: 0 in motion, 1 stopped opening, 2 stopped closing, 3 requested position reached; interpret with activation/action state, not as independent grasp success |
| `raw_fault`, `fault_code` | Full gFLT byte and its low nibble, respectively |
| `fault_status` | Decoded low-nibble status: `NONE`, `ACTION_DELAYED`, `ACTIVATION_NEEDED`, `MAX_TEMP_EXCEEDED`, `COMM_TIMEOUT`, `UNDER_VOLTAGE`, `AUTOMATIC_RELEASE_IN_PROGRESS`, `INTERNAL_FAULT`, `ACTIVATION_FAULT`, `OVERCURRENT`, `AUTOMATIC_RELEASE_COMPLETED` or `UNKNOWN` |

Diagnostic severity is OK for fresh fault-code 0, WARN for a fresh nonzero device fault, ERROR for communication/connection failure, and STALE for missing/stale feedback. `status.message` contains the state plus the failure reason when available. A device fault with a valid response is distinct from failed communication. The library collapses malformed responses, Modbus exceptions and response timeout into invalid feedback; the bridge reports that common failure without inventing a more precise cause.

Before the first attempt, including when disabled, state is `missing`. A failed attempt invalidates data immediately, even within the freshness interval. If no attempt has failed but time since the last success exceeds `max_age_s`, state becomes `stale`. Reconnection cannot reuse the previous measurement as fresh. Source stamps from the gripper are unavailable; all measurement timestamps are host receipt/request times, not sensor clocks or measured network latency. Downstream consumers cannot compare monotonic clocks across machines.

## Model and calibration

The available label is “Robotiq 2F Adaptive Gripper C-49010”; prior records call 2F-85 likely, while the exact mounted 85/140 variant remains unconfirmed. The six feedback registers do not establish that variant. This bridge therefore publishes normalized closure and raw feedback, with no metric joint-state/width topic.

Before adding metric width, identify the exact unit/model and fingertip configuration from the full nameplate/product records, confirm the applicable vendor calibration, and qualify measured jaw opening versus gPO across the relevant travel. Record the mapping, uncertainty and units; a default 0.085 m linear scale is not measured calibration. Activation/open/close exercises for calibration are outside Milestone 2A. Robotiq tool/TCP/payload calibration remains separate from jaw-width calibration.

## Optional position-command contract — offline implementation

The position interface uses the existing library and serial worker. It adds no second device connection and leaves the ownership blocker and its fail-closed checks intact. No live command node or physical operation is authorized by these instructions.

The immutable settings `commands_enabled: false` and `command_owner: ""` keep command endpoints absent by default. The remaining settings cap normalized speed/force at `0.1` and position change at `0.1`. These are software register-domain limits, not calibrated physical safety limits. Both the requested and byte-quantized target must satisfy the step limit against fresh measured position. Feedback age for commands is at most the smaller of observation `max_age_s` and 0.5 s.

When a future reviewed deployment explicitly enables the interface, two typed services become available:

| Service | Request | Response and meaning |
| --- | --- | --- |
| `~/authorize_commands`, [`AuthorizeCommands`](srv/AuthorizeCommands.srv) | `enable`, configured `owner`, `authorization_id`, `lease_ms` | `success`, `authorization_id`, `reason`. Grant uses an empty ID and lease 1–1000 ms. Requires a connected, fresh, already-activated and fault-free device. Returns a random 128-bit bearer ID; a live lease cannot be silently renewed or replaced. Revocation requires the owning ID and sends no stop/motion command |
| `~/set_normalized_position`, [`SetNormalizedPosition`](srv/SetNormalizedPosition.srv) | `owner`, bearer `authorization_id`, increasing nonzero `sequence`, UNIX `issued_at`, `valid_for_ms` 1–500, normalized `position`, `speed`, `force` | `accepted`, `sequence`, `stage`, `reason`. Acceptance means one queued request only. It is not register acknowledgment, motion completion, request echo or grasp confirmation |

Position means normalized closure: 0 open, 1 closed. Speed and force are raw-register fractions; all three are finite and in [0,1]. Encoding truncates `fraction * 255` to a byte. Register zero means minimum speed/force, not zero movement/force. The native go-to request retains `rACT=1` and sets `rGTO=1`; it is permitted only after measured activation completion. No activate, reset, fault recovery, homing, auto-release or separate open/close endpoint is supplied.

Only the configured cooperative owner with the current bearer ID may submit commands. Requests are rejected for missing/expired leases, duplicate/out-of-order sequences, another pending request, future/expired/invalid source stamps, range/limit violations, stale feedback, reset/activation-in-progress, reported motion, faults or communication failure. Wall-clock and monotonic expiry are both checked; clocks must be qualified before cross-host use. At dispatch the same serial worker has just polled feedback and rechecks all guards. It sends at most one FC16 request and validates its acknowledgement using the existing echo/CRC/framing implementation. The new vendor method does not enter the legacy blocking motion-completion loop.

`~/command_status` is a separate reliable/volatile/keep-last-1 `DiagnosticArray`, published at 10 Hz only when commands are configured. It contains `schema_version=1`, `stage`, `sequence`, configured `owner` and `grasp_outcome=unknown`; its status text adds the reason. `accepted` means queued, `acknowledged` means an FC16 register-write reply, `dropped` means queued but not sent, `failed` means no valid write acknowledgment with an unknown device outcome, and `revoked` is a lifecycle/authorization event. It reports latest state, not a durable event history. Bearer IDs are not included. Service rejection is returned to that caller and does not replace another accepted request's status.

An acknowledged command is not retried. A later feedback read must expose the requested raw position echo before another request is admitted, and reported motion continues to block new requests. Observation `raw_commanded_position` remains gPR, `raw_position` remains gPO; neither is overwritten with the accepted target. The protocol has no unique per-command echo identifier, so matching gPR is not proof of execution or grasp.

Reconnection, communication failure, unhealthy device state, explicit revocation and shutdown discard queued work and invalidate its lease. A new connection needs fresh feedback and explicit authorization again. No lifecycle operation submits a replacement command. Revocation/expiry prevent future dispatch; they **do not stop an already-issued move**, cancel autonomous re-grasp or implement a safety stop. Dispatch and revocation are serialized; revocation cannot recall bytes already being transmitted.

Bearer leases and owner names are software coordination, not authenticated operator identity. Future hardware deployment requires administrator-reviewed ownership configuration, approved onsite authorization, authenticated/authorized DDS service access, clock qualification and physical limits/calibration review. The existing procfs blocker remains unresolved. The original synchronous serial write is not covered by the response-read deadline, so this is not a hard real-time transaction or shutdown guarantee. Fault/acknowledgment failure revokes authorization, closes the worker's own connection and never retries a position request automatically.
