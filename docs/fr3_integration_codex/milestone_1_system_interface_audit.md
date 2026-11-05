# RLInf Milestone 1 — System and Interface Audit

This report identifies the RLInf components that can be reused with the AIST FR3 system, the missing interfaces, and the work needed before integration. The recommended architecture keeps the existing NUC ROS 2 controller and Robotiq library as hardware owners and connects RLInf through narrow adapters. **The system is not yet qualified for an RLInf control session.**

Audit date: **2026-10-27 JST**. Scope: source inspection, historical hardware evidence, upstream comparison, and CPU-only tests. Milestone 1 ends with this report; the steps below are recommendations, not implementation or authorization for Milestone 2.

## 1. Scope and evidence boundaries

The requested `docs/fr3_integration_codex/` instruction pack was absent when the audit started. This directory was created only to deliver this report. The audit used the supplied task boundaries, repository instructions, existing experiment records, and `../fr3-system-audit` as the hardware baseline.

| Evidence | Identity and interpretation |
| --- | --- |
| Audited checkout | `/home/muttaqien/Desktop/AIST-Tokyo/gamma/rlinf-project`, branch `branch-openpi-recap-franka`, HEAD `cdc99764dc4ae78bd25add1b83eb908da540d252` |
| Baseline repository | [`fr3-system-audit` README](../../../fr3-system-audit/README.md), [historical baseline](../../../fr3-system-audit/docs/AUDIT_BASELINE.md), [configured interfaces](../../../fr3-system-audit/config/system.json) |
| Latest saved hardware report | [2026-10-27 03:16:16 +09:00 full report](../../../fr3-system-audit/reports/audit_20261027_031616_+0900_full.md): inventory `DETECTED`, operating mode `IDLE`, operational readiness `IDLE`; serial feedback and physical motion `NOT_TESTED` |
| Current upstream RLInf | GitHub API resolved `RLinf/RLinf` main to `fab7f407fbc611085eb163962e3c89c0ab424c1a`; eight source files were fetched at that immutable revision and compared byte-for-byte |
| Separate baseline RLInf path | Hardware-audit configuration points to sibling `../RLinf`, whose local HEAD is `0067f7d5ce525c18bda439ed8dbb638abafd6a06`. It is a separate directory, not the checkout audited here |
| Existing SpaceMouse workspace | `../franka_spacemouse_ws/src/franka_spacemouse`, local HEAD `1d041da5a34bb040f8a97192504c9137eb0b8923`; its publisher source and configuration were read without execution |
| NUC source inspection during this audit | SSH read-only Python file reads inspected the existing controller, Robotiq C++ header, and selected implementation lines. No ROS environment was sourced, driver imported, or hardware device opened |

Historical observations are identified as such throughout. No current hardware-health verdict is inferred from file presence, successful SSH, or the saved reports. The system-audit CLI was not rerun because existing evidence suffices for this interface audit.

Existing untracked items `assets/pi05_stackcube_panda_joint` and `docs/experiments/openpi_recap_franka/fr3_startup_fix/` were present before this work and were left untouched. The latter contains useful historical evidence but is not part of the recorded HEAD.

## 2. Hardware baseline and outstanding qualifications

The hardware baseline establishes topology and prior communication results. Its latest saved state is idle, so it does not establish fresh arm, camera, or input reception.

| Component | Baseline evidence | Qualification still missing |
| --- | --- | --- |
| Workstation `pca03395` | Four RTX PRO 6000 Blackwell Max-Q GPUs, driver `590.48.01`; ROS 2 Jazzy | RLInf hardware integration and current resource availability were not measured |
| NUC `aistfranka-NUC13ANK-B` | ROS 2 Humble; Franka hardware/controller overlays; historical 1000 Hz controller manager and FIFO priority 50 | Running binary/overlay identity, cycle timing and physical behavior |
| Network | Workstation `10.42.0.1`, NUC `10.42.0.158`, DDS domain `70`, `rmw_fastrtps_cpp`; historical seven-joint reception | Sustained state streams, QoS, clock relationship, reconnect behavior, and cross-distro custom messages |
| FR3 | Historical seven-joint position/velocity/effort feedback; latest manager stopped | Current physical fault status, calibrated tool/payload, fresh pose/wrench feedback, supervised stationary behavior |
| Robotiq | Stable adapter `/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DA95C62Y-if00-port0`; compiled C++ library; historical CRC-valid FC03 response, raw position `225` | Continuous measured feedback interface, exact 85/140 model, width calibration, grasp behavior and fault handling |
| RealSense D405 | SDK serial `218622277827`; RGB `848x480x30`, historical subscriber throughput about 25–27 Hz | Physical base/wrist assignment, sustained timing, exposure timestamps and calibration |
| RealSense D435i | SDK serial `337122075473`; historical throughput improved to about 25 Hz at `640x480x30` | Same qualifications; do not restore the slower 1280x720 profile without measurement |
| SpaceMouse | USB joystick detected; historical ROS publisher/subscriber discovery | Recent deliberate axis/button exercise and qualified end-to-end teleoperation |

The baseline records an initial Robotiq `COMM_TIMEOUT` (`0x09`) followed by nine fault-free reads. These are separate historical samples, not a current fault-free status. The older [RLInf feedback qualification](../experiments/openpi_recap_franka/milestone_1b_1_report.md) recorded raw position `226` and fault `0x09`; it should not override the newer hardware-baseline observations.

The [startup-fix report](../experiments/openpi_recap_franka/fr3_startup_fix/report.md) also records a robot-reported emergency/User Stop abort and a user-reported Desk Generic joint error. Their physical cause and subsequent resolution are not established by the latest idle inventory. The report records 41 historical offline tests, not a validated physical startup. These remain explicit handoff items before any later hardware-control work.

## 3. Upstream reuse and local additions

Upstream already supplies the composition, sensor drivers, teleoperation abstraction, and collection infrastructure. The missing work is primarily transport and lifecycle adaptation to the installed ROS 2 system.

| Component | Direct source evidence | Reuse decision |
| --- | --- | --- |
| Robot composition | [`FrankaRobot.build_arms`, `declare_end_effector`, `FrankaConfig`](../../rlinf/robotics/robots/franka.py) | Reuse separate arm/end-effector/camera ownership and placement. `controller_node_rank` and `camera_node_rank` describe Ray placement, not DDS host addresses |
| Arm contract | [`Arm`, `BaseArm`, `ARM_STATE_FIELDS`](../../rlinf/robotics/parts/arms/base.py), [`FrankaRobotState`](../../rlinf/robotics/parts/arms/franka.py) | Reuse registration and canonical names. Raw unavailable data needs explicit validity; zero-filled dataclass defaults are not measured state |
| ROS arm backend | [`FrankaROSArm._open`, `start_impedance`, `reset_joint`](../../rlinf/robotics/parts/arms/franka_ros.py), [`ROSController`](../../rlinf/robotics/parts/transports/ros/ros_controller.py) | Reference only for this deployment: uses `rospy`, `dynamic_reconfigure`, SERL messages and `roslaunch`; connection launches impedance control and the transport can launch `roscore` |
| Direct FCI alternative | [`FrankyArm`](../../rlinf/robotics/parts/arms/franky.py) | Existing upstream alternative, but not the recommended path: opens its own libfranka connection and tracking sessions instead of reusing the installed NUC controller |
| Robotiq backend | [`RobotiqGripper`](../../rlinf/robotics/parts/end_effectors/grippers/robotiq.py) | Reuse interface conventions/register knowledge. Do not replace the working NUC C++ controller with a second serial owner. `_open()` activates the gripper |
| Camera backend | [`RealSenseCamera`](../../rlinf/robotics/parts/cameras/realsense.py), [`BaseCamera.get_frame_with_timestamp`](../../rlinf/robotics/parts/cameras/base.py) | Reuse for a deliberately selected direct-SDK acquisition mode. Opens pipelines by serial; BGR8 output; frame timestamp is host receipt after driver read |
| Operator input | [`SpaceMouse`, `SpaceMouseExpert`](../../rlinf/robotics/parts/teleop/spacemouse.py) | Reuse six-axis input and button reading where RLInf owns the HID device; retain the existing ROS publisher as the first integration candidate where it already owns input |
| Raw observation helper | [`FrankaObservation`, `state_snapshot`, `RosStateSubscriber`](../../toolkits/realworld_check/franka_observation.py), [ROS 2 subscriber helper](../../toolkits/realworld_check/franka_ros2_observer.py) | Existing local work worth reusing: source headers, measured joint arrays, receipt clocks, explicit unavailable/stale states, separate operator intent. This is not a registered ROS 2 arm backend |
| Task/collection layers | [`FrankaEnv`](../../rlinf/envs/real/franka/base.py), [real-task registry](../../rlinf/envs/real/registry.py), [`CollectEpisode`](../../rlinf/envs/wrappers/collect_episode.py) | Reuse after transport and observation/action contracts are qualified. The current physical env constructor connects hardware and interpolates to reset pose, so construction is an operational action |

The upstream comparison was pinned to [revision `fab7f407`](https://github.com/RLinf/RLinf/tree/fab7f407fbc611085eb163962e3c89c0ab424c1a). The following five files matched exactly: `franka_ros.py`, `ros_controller.py`, `robotiq.py`, `realsense.py`, and `rlinf/envs/real/franka/base.py`. This confirms their described limitations are present upstream, rather than inferred from an old local fork.

The three differences in the comparison matter when planning reuse:

- `robots/franka.py`: only the inspected camera-backend docstring differs; upstream also lists `orbbec`.
- `teleop/spacemouse.py`: this checkout adds explicit `device_path` selection and `open_by_path`, including config forwarding. Preserve this local device-selection capability if updating upstream code.
- `arms/franky.py`: upstream has additional joint-reset rest checks, dynamics scaling and recovery/retry behavior absent locally. Neither version was executed. Do not migrate to that backend merely because its reset code is newer; that would change hardware ownership and lifecycle.

The [ZED + Robotiq collection config](../../examples/embodiment/config/realworld_collect_data_zed_robotiq.yaml) demonstrates composition, SpaceMouse, remote controller placement and LeRobot export. Its ZED serials, sample port, Panda dataset label and task poses are examples, not deployment values for this FR3.

## 4. Existing interfaces and semantic gaps

The installed controller exposes normalized Cartesian intent, whereas RLInf's Franka environment builds absolute TCP pose targets from per-step deltas. These contracts require an explicit conversion design; matching vector lengths is insufficient.

| Interface | Existing contract/evidence | Gap to close |
| --- | --- | --- |
| Arm joints | Baseline `/joint_states`, `sensor_msgs/msg/JointState`; rad, rad/s, effort feedback | Validate seven named FR3 joints, ordering, age and publisher provenance; historical presence does not validate all other state fields |
| Pose and wrench | Local observer defaults to `/franka_robot_state_broadcaster/current_pose` (`PoseStamped`) and `/external_wrench_in_base_frame` (`WrenchStamped`), with `/measured_joint_states` (`JointState`) under the same prefix | Defaults are candidates, not freshly discovered endpoints. Verify actual topic names, frames, source stamps and availability once the approved existing publisher is running |
| Arm command | `/franka_controller/target_cartesian_velocity_percent`, `geometry_msgs/msg/Twist` | Unstamped normalized axes, not SI velocity. Preserve mounting transform and local controller semantics; no direct forwarding of model output |
| RLInf arm command | `FrankaROSArm` accepts `tcp_pose`; `FrankaEnv.step` scales Cartesian deltas then produces xyz + xyzw quaternion targets | Current ROS 2 controller is not an absolute-pose endpoint. A future adapter must explicitly select supported action semantics and reject incompatible requests |
| Robotiq feedback | NUC `RobotiqGripperInterface::get_feedback()` and historical FC03 communication | Missing continuous qualified ROS 2 state source; keep measured raw position, requested-position echo, status bits, fault/current, validity and receipt time distinct |
| Robotiq command | NUC library separates `connect`, `reset`, `activate`, `set_gripper_position`, `open_gripper`, `close_gripper` | Future bridge must expose reviewed command units/lifecycle. Franka Hand actions are incompatible; no bridge is established by baseline evidence |
| Camera images | ROS topics `/camera/d405/color/image_raw`, `/camera/d435i/color/image_raw`, `sensor_msgs/msg/Image`; SDK backend separately supplies BGR arrays | Choose one owner per camera. Existing helper opens SDK pipelines, so it cannot be started casually alongside existing camera nodes. Preserve encoding, serials and timestamp provenance |
| Input | Existing ROS publisher maps `(-y, x, z, -roll, -pitch, -yaw)` and publishes every 0.01 s | One HID owner; preserve `operator_position_front` setting. Input and commands must remain separate from measured state |
| Gripper buttons | Existing publisher sends `Float32` width percentage on `gripper_client/target_gripper_width_percent`, button 0 → 0.0 and button 1 → 1.0 | Its accompanying manager sends Franka Hand actions. Reusing the arm/input publisher does not make that manager compatible with Robotiq |

### Controller source has evolved beyond the older reference notes

The NUC file `/home/aist-franka/Desktop/aist-gamma/franka_controller_ws/src/franka_arm_controllers/src/joint_impedance_ik_controller.cpp` was read and hashed during this audit. Its SHA-256 is:

```text
8d6ebf4ce420586dca3fe39a0befbd0d7bd61a4230153a2c7a5637cdd8bbdc47
```

It matches the saved [source manifest](../experiments/openpi_recap_franka/fr3_startup_fix/source_manifest.json). Inspected source lines show `command_timeout` default `0.25` (line 108), SensorDataQoS keep-last-1 command subscription (194), command/state receipt-age checks (292–293), and normalized offsets `0.007` metres and `0.03` radians (306–307). The [saved incremental patch](../experiments/openpi_recap_franka/fr3_startup_fix/franka_arm_controllers.patch) additionally documents measured-state IK seeding, bounded reference updates and pre-activation rejection.

The older [SpaceMouse reference](../experiments/openpi_recap_franka/spacemouse_ros2_reference.md) says the previously inspected upstream controller lacked an explicit timeout. That statement must not be applied to the current NUC source. Conversely, a matching source hash does not establish that the running/installed library contains these changes. The startup-fix report says normal installed binaries were left unchanged and builds were isolated under `/tmp/fr3_startup_offline/`.

Expiry holds the last impedance reference; it does not deactivate torque control or implement a safety-rated deadman. Because the Twist has no source timestamp, local receipt expiry cannot detect an old HID event that arrives late. The control source's normalized pose offsets also cannot be relabelled as metres/second by multiplying by a publisher frequency.

### Preserve the working Robotiq implementation

Read-only NUC inspection confirmed the existing header at `/home/aist-franka/robotiq/robotiq-gripper-interface-main/include/robotiq/robotiq_gripper_interface.h`, implementation `src/robotiq_gripper_interface.cc`, and serial framing helpers. The public API separates connection, activation and feedback, and the activation documentation explicitly states that it causes motion. Implementation inspection located FC03 response handling in `src/helpers.cc` and feedback reads through `READ_FEEDBACK`.

Prefer wrapping this existing C++ interface over introducing another Modbus implementation or importing RLInf's activation-on-connect backend. Future qualification must inspect the full connection/destruction/failure paths and prove that observation-only lifecycle operations do not activate, reset or move the device. No library method was called in this audit.

RLInf's Python `position` property converts raw position through a default maximum width of `0.085` m and retains a cached position after a failed read. Its `is_open` flag records requested open/close state. Neither is sufficient as calibrated, fresh measured jaw opening or grasp success. Preserve native feedback and report unavailable values on failed/stale reception rather than using cached/default values as fresh measurements.

## 5. Recommended architecture

The recommended split retains real-time and serial ownership on the NUC. RLInf remains on the GPU workstation and consumes explicit state and timing contracts. Existing ROS 2 sensor/input publishers are the first reuse candidates; direct SDK/HID modes remain alternatives with exclusive ownership.

```text
GPU workstation: RLInf workers / future collection or policy task
    |
    +-- observation assembly: source stamps, host receipts, validity, frames
    |       +-- existing Jazzy camera streams (D405 and D435i)
    |       +-- existing SpaceMouse input source (one HID owner)
    |       +-- existing ROS 2 arm feedback via standard messages
    |       +-- future qualified Robotiq state bridge
    |
    +-- future RLInf part adapter and explicit action conversion
                 |
                 +-- future NUC-local command arbitration/validation
                         +-- existing Humble FR3 controller -> FCI
                         +-- narrow Robotiq bridge -> existing C++ library -> RS-485
```

The adapter/bridge/arbitration boxes marked **future** do not exist as a validated integration today. They are responsibilities, not proposed new frameworks or implemented endpoints.

Reuse `Robot`, `Arm`, `EndEffector`, camera and teleop contracts rather than inventing a parallel robot API. A new ROS 2 part adapter should attach to externally managed ROS 2 services, with observation connection separate from command permission. Connecting, closing, timing out or reconnecting must not implicitly launch controllers, recover robot faults, reset a task, activate a gripper, or begin motion.

Keep ROS-native processes in their existing distro/interpreter environments. The local observation helper already demonstrates using system ROS Python in a subprocess while RLInf remains in its virtual environment. Prefer standard ROS message types initially; qualify Jazzy/Humble compatibility and QoS rather than assuming arbitrary custom types interoperate. A ROS transport split does not require adding the NUC as a Ray node. Add remote Ray placement only if a later design actually needs it and its Python/dependency compatibility is established.

Command ownership must be singular on the NUC: operator input and any later policy cannot independently publish competing live commands. A future command gateway needs explicit mode/enable ownership, finite/range/frame checks, sequence/source-age information where supported, expiry and a documented hold/disable policy. This complements the current controller timeout; it does not claim to replace physical stop/enable protections.

Use the FR3 model and mounted Robotiq geometry for TCP, payload and limits. The shared local Franka constants explicitly describe Panda joint limits and are consumed by Franky. The sibling SpaceMouse workspace also contains `load_gripper: "true"` examples. Neither should be copied into a deployment configuration without review; earlier NUC work specifically retained `load_gripper: "false"` and `fr3_link8`.

## 6. Recording and policy boundary

RLInf collection and model transforms are reusable only after the raw interface contract is settled. This is a later implementation concern, included here because it affects which metadata the adapters must preserve.

The current physical Franka observation contains TCP pose 7, TCP velocity 6, force 3, torque 3 and gripper position 1: **20 scalar values** before wrapper transforms. [`RealworldInputs`](../../rlinf/models/embodiment/openpi/policies/realworld_policy.py) requires a **19-element** state and produces two active image slots for PI0/PI05. Its absent second-image fallback can synthesize a blank image; the local [sample validator](../../toolkits/realworld_check/franka_pi05_contract.py) deliberately requires both real camera inputs for this experiment. [`value_dataset.py`](../../rlinf/data/datasets/recap/value_dataset.py) uses `FrankaEEInputs` for its Franka route, so that route is not automatically the same contract.

Do not silently truncate 20 values to 19, relabel normalized Twist as a seven-dimensional policy action, or assume the collection example's dataset metadata establishes compatibility. Record operator intent, accepted commands, measured state, controller health, camera serial/encoding, source clock, host receipt clock and frame/units separately. Base/wrist physical roles and calibrated width remain unresolved. The existing [data-contract audit](../experiments/openpi_recap_franka/data_contract.md) describes these gaps in more detail.

`CollectEpisode` can export LeRobot, but its conversion skips frames with missing required data and can replace actions with intervention actions. A later dataset review must check those behaviors against the chosen command/state alignment and provenance contract, rather than treating a successfully written dataset as interface acceptance.

## 7. Next implementation steps — proposed, not executed

The next work should close observation and ownership gaps before exposing any policy command path. None of these steps was started as Milestone 2.

| Order | Proposed work | Reviewable acceptance evidence |
| --- | --- | --- |
| 1 | Freeze deployment identities: distinguish this checkout from `../RLinf`; identify exact NUC binary/overlay, model, firmware/libfranka, installed controller revision and current Desk status | Version/path manifest with source-versus-binary distinction; explicit record of unresolved physical faults |
| 2 | Specify observation contract and qualify existing publishers when separately available | Bounded fresh seven-joint/pose/wrench samples, publisher/QoS identities, frames and stale/reconnect behavior; no automatic hardware startup |
| 3 | Select camera and HID owners | Per-device ownership table, physical base/wrist confirmation, existing stream versus SDK decision, encoding and timestamp definitions; no duplicate opens |
| 4 | Design an observation-first Robotiq bridge around the existing NUC C++ interface | Full lifecycle review, retained raw measured/requested/status/fault fields, unavailable-on-error behavior, stable port identity; model/calibration required before metric width |
| 5 | Implement the narrow registered ROS 2 part adapter in a separately authorized milestone | CPU tests for contract, joint ordering, partial failure, reconnect and idempotent cleanup; fake vendor/process edges; no launch/activation from observation connection |
| 6 | Specify action conversion and arbitration against the actual controller | Explicit normalized-intent or absolute-target capability, units/frames/rates, owner/mode/expiry rules, rejected unsupported actions and offline loss/invalid-input tests |
| 7 | Resolve physical qualifications through the responsible operator's approved procedure before any control validation | Current fault disposition, TCP/payload/limits, consistent installed build and supervised stationary evidence. Further teleop/motion requires separate authorization |
| 8 | Qualify task/collection/model boundaries after the interfaces pass | Explicit state projection, two-image mapping, action meaning and temporal alignment; LeRobot round-trip and model/value-transform checks on actual data |

For a future upstream change, reuse existing registration points, lazy optional imports and component-level tests; restore CI path coverage if this experiment branch lacks the upstream workflows. A ROS 2 backend name or new config remains a design choice, not an existing supported option. No new backend identifier is asserted by this report.

## 8. Upstream overlap

Read-only open-PR and issue searches were completed with `gh` for `franka OR robotiq OR realsense OR spacemouse`. The broad PR results included adjacent SO-101, R2 and other real-world work; no returned title identified this exact FR3 + existing Humble controller + C++ Robotiq bridge. This is a bounded search result, not proof that no related work exists.

The following open issues were read and are relevant context:

- [#1012 — SpaceMouse collection failure](https://github.com/RLinf/RLinf/issues/1012): reports gripper actuation while arm teleoperation fails; motivates testing composition and the action path rather than only individual devices.
- [#1477 — newer Franka firmware compatibility](https://github.com/RLinf/RLinf/issues/1477): reports constraints of the older SERL/ROS stack. This is a user report, not proof of this FR3's firmware compatibility.
- [#1412 — GELLO with FR3](https://github.com/RLinf/RLinf/issues/1412): raises FR3 support/model assumptions; its GELLO-specific symptoms do not establish SpaceMouse behavior.
- [#974 — UR5e + Robotiq + RealSense](https://github.com/RLinf/RLinf/issues/974): adjacent component-composition request on a different arm.

No upstream issue, PR, comment, commit or push was created or edited.

## 9. Verification and stop record

Verification was restricted to questions that can be answered without opening hardware. Existing source and historical reports were inspected; no new physical health claim is made.

| Check actually performed | Result |
| --- | --- |
| Instruction-pack discovery | Pack absent at audit start; report directory created for this deliverable |
| Local source/config and historical baseline inspection | Completed; path drift, ROS 1/2 mismatch, lifecycle side effects and contract gaps documented |
| Current upstream comparison | Eight files fetched at pinned SHA; five identical, three differences inspected |
| NUC SSH file inspection | Completed; controller source hash matches saved manifest; Robotiq API/source inspected without import or method calls |
| Focused workstation CPU tests | **12 passed in 2.19 s**, Python 3.11.15, pytest 9.1.1; no GPU, ROS or hardware process required |
| Report references and change scope | All 29 local Markdown links resolve; tracked diff whitespace check passed; status shows the new report directory alongside the two pre-existing untracked items |
| Hardware, serial probe, camera/HID acquisition, motion and end-to-end control | Not run under the requested scope |
| Sphinx EN/ZH build and RST checks | Not run: standalone audit Markdown only; published EN/ZH RST trees unchanged |

The exact CPU command was:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit_tests/test_franka_observation.py \
  tests/unit_tests/test_franka_pi05_contract.py
```

These tests exercise source preservation, stale/stopped/missing state handling, duplicate camera-role rejection and candidate model sample validation. They do not test physical feedback, calibrated actions or safety behavior. Initial sandbox network restrictions on GitHub/SSH inspection were resolved through the approval mechanism; they were not hardware failures.

Prepared with an AI assistant. The only intended repository change from this audit is this report. No operational robot software was modified; no controller, camera, HID reader, ROS observer, gripper driver or training process was started. No activation, recovery, serial request or motion command was sent. **Milestone 1 is complete as a system/interface audit; Milestone 2 has not begun.**
