# FR3 + OpenPI π0.5 / RECAP — Roadmap Status

**Updated:** November 7, 2026 (JST). Status reflects completed offline tests and observed hardware work; it does not imply clearance for autonomous robot operation.

| Milestone | Objective | Status | Remaining gate |
| --- | --- | --- | --- |
| 0 — Hardware Audit | Workstation, NUC, FR3, cameras, SpaceMouse, Robotiq | **Complete** | — |
| 1 — System & Interface Audit | RLinf and ROS 2 architecture | **Complete** | — |
| 2A — Robotiq Observation Bridge | Read-only gripper feedback | **Complete** | — |
| 2B — Robotiq Hardware & DDS | Operational Robotiq hardware feedback and ROS 2 communication | **Complete** | — (legacy standalone FC03 qualification remains separate) |
| 2C.1 — Robotiq Commands (Offline) | Command interface and offline tests | **Complete** | — |
| 2C.2 — Robotiq Commands (Physical) | Supervised gripper operation | **Complete** | — |
| 3A — Multimodal Observations (Offline) | Arm, gripper and two cameras | **Complete** | — |
| 3B — Live Observation Qualification | Live observation sampling | **Complete** | Cross-host/exposure sync is a later dataset gate |
| 4 — Teleoperation & Dataset Collection | Record and export training-ready demonstrations | **Partial** | Safety-qualified movement, accepted-action capture, clock alignment, real LeRobot dataset |
| 5 — π0.5 Inference Integration | Policy inference and action adapter | **Not started** | Requires approved action interface and safety gates |
| 6 — End-to-End Validation | Real manipulation task evaluation | **Not started** | Requires Milestone 5 |

**2B scope clarification (November 7):** The **original October 27 standalone vendor FC03 read-only bridge** was only partially qualified (serial-disabled DDS 180/180; no direct FC03 trial). However, the **active Robotiq 2F-85 ROS 2 hardware driver** is running in real-hardware mode on NUC domain 71, with named knuckle-joint feedback and **60/60 live samples** qualified in Milestone 3B on November 6. Thus **2B is complete for the operational FR3/Robotiq integration used by this roadmap**. This does **not** claim the legacy standalone FC03 experiment passed its own physical serial-owner test.

## Milestone 4 — What's done

| Substep | Status |
| --- | --- |
| 4A — Stationary raw recording | **Complete** |
| 4B.1/4B.2 — SpaceMouse + gripper events | **Complete** |
| 4B.3 — Arm controller safety | **Partial** (source audit and stationary checks complete; physical release/fault validation pending) |
| 4C.1 — Raw capture preflight | **Complete** (100 real frames) |
| 4C.2 — Candidate 19D state + Robotiq action semantics | **Partial** (state and gripper semantics verified; six arm actions not yet observable) |
| 4C.2T.1 — Offline telemetry contract | **Complete** (11/11 workstation tests) |
| 4C.2T.2 — Nonblocking queue prototype | **Complete** (3/3 workstation tests, 0.58 s) |
| 4C.2T.3 — Offline controller telemetry adapter | **Complete (offline)** (2/2 workstation tests, 0.61 s) |
| 4C.2T.4 — Separate offline controller reference build | **Complete (offline)** (3/3 workstation tests, 0.68 s) |
| 4C.2T.5 — Isolated actual ROS 2 controller build | **Complete (isolated build)** (NUC: 1 package finished in 0.70 s; BUILD_TESTING=OFF) |
| 4C.2T.6 — Non-RT JSONL recorder (synthetic) | **Complete (offline)** — 1/1 test passed; standalone demo verified: 2 saved, 1 dropped, 2 JSONL lines |
| 4C.2T.7 — Non-RT recorder in isolated ROS controller | **Complete (stationary real-hardware capture)** — recorder-enabled controller loaded; operator activated and deactivated successfully; 4,412 JSONL arm-tick samples (5.0 MB); final controller state inactive |
| 4C.3/4C.4 — LeRobot export + OpenPI/RECAP data path | **Not started** |

**Bottom line:** Milestones **0–3B are operationally complete**; **Milestone 4 is in progress**; **5 and 6 are not started**. The isolated 4C.2T.7 telemetry recorder was deployed and verified with a brief stationary FR3 run; no moving-arm demonstration or policy-ready 7D action dataset exists yet.

**Active-controller pre-deployment check (November 7):** The live ROS domain-70 `joint_impedance_ik_controller` is active in PID `1259561`, loading the **original** `franka_controller_ws/build/franka_arm_controllers/libfranka_arm_controllers.so`. The operator saved a rollback copy at `~/fr3_controller_backup_20261107_055031/libfranka_arm_controllers.so`; original and backup SHA-256 both `ea4ac304e079eb5d8ce2f15917bcabe3823ea060c86a8be82702a560937f30c6`. The rollback directory also contains `controllers.yaml`, `aist_fr3_config.yaml` and `joint_impedance_ik_controller.launch.py` (operator reported `PASS: Controller binary and launch configuration backed up`). Historical pre-deployment snapshot. Subsequently the isolated recorder controller was loaded and activated briefly under operator supervision; see the November 7 stationary capture record below. Physical moving-arm safety qualification and approval remain outstanding.

## Real stationary controller telemetry verified — November 7, 2026

The operator confirmed the isolated recorder-enabled `JointImpedanceIKController` library at `~/fr3_telemetry_isolated_4c2t7/install/franka_arm_controllers/lib/libfranka_arm_controllers.so`, the configured `telemetry_jsonl_path`, zero connected motion-command publishers, and activation/deactivation of the joint controller through ROS 2 Humble. The resulting `~/fr3_telemetry_isolated_4c2t7/recordings/stationary_telemetry.jsonl` contains **4,412 controller-originated JSONL arm_tick records** (approximately 5.0 MB). First sample schema is `fr3_controller_capture_raw_v1`, with measured FR3 q, reference before/after, timestamps and gates. At rest, `intent_axes` were all zero, `ik_target_q_rad=null`, `reference_advanced=false`, and `training_action=null`, `policy_ready=false`. The operator's final `list_controllers` shows `joint_impedance_ik_controller` **inactive**. This proves real stationary recorder functionality, not accepted moving 6D commands, cross-host synchronization, LeRobot compatibility, or moving-arm safety. **Preserve the stationary file**: the recorder opens its configured path using `std::ios::trunc` on activation. Use a fresh per-episode path before another activation. Existing raw camera capture utility is `toolkits/realworld_check/fr3_raw_capture.py`; it does not itself control the robot or produce action labels. Only perform a physically approved supervised moving-arm demo using established lab procedure.
