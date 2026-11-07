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
| 4 — Teleoperation & Dataset Collection Setup | Prepare FR3/SpaceMouse/Robotiq/two-camera recording infrastructure and document the future dataset interface; no real manipulation experiment required | **Complete — setup scope only** | — (experimental synchronization, action labeling and LeRobot export deferred) |
| 5 — π0.5 Inference Integration | Policy inference and action adapter | **Not started** | Requires action-interface review; no physical experiment implied |
| 6 — End-to-End Validation | Real manipulation task evaluation and future robot experiments | **Not started** | Requires Milestone 5 plus lab safety approval |

**2B scope clarification (November 7):** The **original October 27 standalone vendor FC03 read-only bridge** was only partially qualified (serial-disabled DDS 180/180; no direct FC03 trial). However, the **active Robotiq 2F-85 ROS 2 hardware driver** is running in real-hardware mode on NUC domain 71, with named knuckle-joint feedback and **60/60 live samples** qualified in Milestone 3B on November 6. Thus **2B is complete for the operational FR3/Robotiq integration used by this roadmap**. This does **not** claim the legacy standalone FC03 experiment passed its own physical serial-owner test.

## Milestone 4 — What's done

| Substep | Status |
| --- | --- |
| 4A — Stationary raw recording | **Complete** |
| 4B.1/4B.2 — SpaceMouse + gripper events | **Complete** |
| 4B.3 — Arm controller safety | **Setup audit complete for stationary operation** (physical moving-arm safety/fault qualification deferred to the approved experiment stage) |
| 4C.1 — Raw capture preflight | **Complete** (100 real frames) |
| 4C.2 — Candidate 19D state + Robotiq action semantics | **Candidate setup documented** (moving-arm action labels not required in Milestone 4 setup readiness) |
| 4C.2T.1 — Offline telemetry contract | **Complete** (11/11 workstation tests) |
| 4C.2T.2 — Nonblocking queue prototype | **Complete** (3/3 workstation tests, 0.58 s) |
| 4C.2T.3 — Offline controller telemetry adapter | **Complete (offline)** (2/2 workstation tests, 0.61 s) |
| 4C.2T.4 — Separate offline controller reference build | **Complete (offline)** (3/3 workstation tests, 0.68 s) |
| 4C.2T.5 — Isolated actual ROS 2 controller build | **Complete (isolated build)** (NUC: 1 package finished in 0.70 s; BUILD_TESTING=OFF) |
| 4C.2T.6 — Non-RT JSONL recorder (synthetic) | **Complete (offline)** — 1/1 test passed; standalone demo verified: 2 saved, 1 dropped, 2 JSONL lines |
| 4C.2T.7 — Non-RT recorder in isolated ROS controller | **Complete (stationary real-hardware capture)** — recorder-enabled controller loaded; operator activated and deactivated successfully; 4,412 JSONL arm-tick samples (5.0 MB); final controller state inactive |
| 4C.3/4C.4 — LeRobot export + OpenPI/RECAP data path | **Setup interface reviewed** — dual-camera RGB keys, candidate 19D state and `(N,7)` action input documented and validated structurally; actual exporter/action labels **not implemented or claimed ready** (deferred to later experimental pipeline) |

**Bottom line:** Milestones **0–4 are complete within their defined setup/integration scopes**; **5 and 6 are not started**. Milestone 4 does not require a real manipulation experiment or a completed training dataset. The isolated 4C.2T.7 telemetry recorder was deployed and verified with a brief stationary FR3 run. Real moving demonstrations, actual training data collection and end-to-end robot experiments are outside Milestone 4.

**Active-controller pre-deployment check (November 7):** The live ROS domain-70 `joint_impedance_ik_controller` is active in PID `1259561`, loading the **original** `franka_controller_ws/build/franka_arm_controllers/libfranka_arm_controllers.so`. The operator saved a rollback copy at `~/fr3_controller_backup_20261107_055031/libfranka_arm_controllers.so`; original and backup SHA-256 both `ea4ac304e079eb5d8ce2f15917bcabe3823ea060c86a8be82702a560937f30c6`. The rollback directory also contains `controllers.yaml`, `aist_fr3_config.yaml` and `joint_impedance_ik_controller.launch.py` (operator reported `PASS: Controller binary and launch configuration backed up`). Historical pre-deployment snapshot. Subsequently the isolated recorder controller was loaded and activated briefly under operator supervision; see the November 7 stationary capture record below. Physical moving-arm safety qualification and approval remain outstanding.

## Real stationary controller telemetry verified — November 7, 2026

The operator confirmed the isolated recorder-enabled `JointImpedanceIKController` library at `~/fr3_telemetry_isolated_4c2t7/install/franka_arm_controllers/lib/libfranka_arm_controllers.so`, the configured `telemetry_jsonl_path`, zero connected motion-command publishers, and activation/deactivation of the joint controller through ROS 2 Humble. The resulting `~/fr3_telemetry_isolated_4c2t7/recordings/stationary_telemetry.jsonl` contains **4,412 controller-originated JSONL arm_tick records** (approximately 5.0 MB). First sample schema is `fr3_controller_capture_raw_v1`, with measured FR3 q, reference before/after, timestamps and gates. At rest, `intent_axes` were all zero, `ik_target_q_rad=null`, `reference_advanced=false`, and `training_action=null`, `policy_ready=false`. The operator's final `list_controllers` shows `joint_impedance_ik_controller` **inactive**. This proves real stationary recorder functionality, not accepted moving 6D commands, cross-host synchronization, LeRobot compatibility, or moving-arm safety. **Preserve the stationary file**: the recorder opens its configured path using `std::ios::trunc` on activation. Use a fresh per-episode path before another activation. Existing raw camera capture utility is `toolkits/realworld_check/fr3_raw_capture.py`; it does not itself control the robot or produce action labels. Only perform a physically approved supervised moving-arm demo using established lab procedure.

## Scope correction — Milestone 4 setup readiness (November 7, 2026)

The project owner clarified that the **purpose of Milestone 4 is to ensure the teleoperation and dataset-collection environment is set up and ready**, not to run manipulation experiments or collect a training-ready demonstration dataset now. Do not require moving-arm demonstrations, executed-action labeling, human success labels, a full LeRobot export from real data, or moving-arm fault injection merely to close Milestone 4. These are deferred to later approved experimental/inference-validation stages. The already completed stationary FR3 recorder capture (4,412 records), workstation multimodal raw capture (300/300 frames), SpaceMouse publication, Robotiq integration, and isolated controller runtime deployment establish substantial setup readiness. A limited **offline-only** check of planned episode schema, data-path dependencies, action semantics, and exporter entrypoints may still be useful; distinguish verified software capabilities from future work and never imply an unavailable exporter is functional. Physical FR3 controller and SpaceMouse were left active during the November 7 conversation; return SpaceMouse to neutral and safely deactivate the FR3 controller under the lab's procedure before concluding the session. No additional physical experiment is requested.

## Milestone 4 — Setup closure record (November 7, 2026)

**Decision: COMPLETE for the project owner's specified scope: make the teleoperation and dataset-collection setup ready for future experiments.** This is *not* certification of moving-arm safety, a finished training dataset, or inference functionality. No further physical manipulations or repeated diagnostics are required for this milestone.

| Setup area | Evidence and disposition |
| --- | --- |
| Robot/gripper I/O | FR3/NUC ROS 2 domain 70 and Robotiq ROS 2 domain 71 interfaces observed; supervised Robotiq command and live feedback verified in Milestones 2–3. |
| Manual control | Existing SpaceMouse ROS publisher started on workstation, and isolated FR3 ROS controller activated successfully on NUC. Physical moving-arm testing is **not a Milestone 4 deliverable**. |
| Multimodal collection | Existing `toolkits/realworld_check/fr3_raw_capture.py` persisted 300/300 real FR3 + Robotiq + base/wrist camera observation frames in ~30.06 seconds, zero failures and dropped events, at `~/fr3_demo_20261107_084244`. The run had **zero operator events**, so it was **not a moving-task demonstration**. An earlier event pilot captured 1008 command events (1002 twists, 6 gripper requests), with zero drops. |
| Controller-side logging | Recorder-enabled isolated controller loaded and recorded 4,412 real `fr3_controller_capture_raw_v1` arm ticks (~5.0 MB) to `~/fr3_telemetry_isolated_4c2t7/recordings/stationary_telemetry.jsonl`; successfully deactivated. Recorder opens each configured output path with truncation, so future trials must use unique file paths. The newer configured filename `demo_20261107_073155.jsonl` is a **path only**, not evidence that demo data were captured. |
| Raw episode integrity tooling | `fr3_lerobot_preflight.py` audits manifest, states, camera arrays and command events; earlier user-run 100-frame real-episode preflight passed. No LeRobot training set is generated by this command. |
| Planned OpenPI sample input | `franka_pi05_contract.py` validates two RGB images (`observation/image`, `observation/extra_view_image`), a 19D numeric state, an `(N,7)` numeric action tensor, and a prompt. `fr3_policy_contract.py` implements a **candidate**, not an approved production, 19D state preview and describes proposed seven action semantics. The raw dataset does not supply executed 7D actions. |
| RECAP/LeRobot readiness | Offline interface and blocking requirements documented; **no production LeRobot exporter, time-aligned training labels, or verified RECAP two-camera loader integration**. This remains work for the future training/experimental data pipeline, not a false claim of Milestone 4 setup completion. The offline proposed target-telemetry validator's schema (`fr3_controller_target_telemetry_proposal_v1`) differs from the real captured `fr3_controller_capture_raw_v1`; a conversion/validator adapter has not been verified. |
| Safety and operating state | Hardware moving-arm safety acceptance, watchdog fault injection and any experimental motion remain subject to the lab's approval; this milestone is **not** an authorization to move. In the latest conversation, the operator had activated the FR3 controller and started SpaceMouse. **Its final stopped state has not been reconfirmed**, and safe deactivation should follow local procedures before leaving the robot unattended. |

**Handoff:** Milestone 5 can begin with OpenPI π0.5 inference architecture and non-actuating interface integration. Later training/experiment work must address action semantics, camera/controller time alignment, adapter from the raw live telemetry schema, gripper goal provenance, LeRobot episode export, and physical safety approval. Keep these as transparent downstream dependencies; do not reopen Milestone 4 simply because experiments have not yet run.

## Offline synchronization and LeRobot smoke — prepared (November 7, 2026)

New repository code: `toolkits/realworld_check/fr3_dataset_setup_smoke.py` (offline-only) and `tests/unit_tests/test_fr3_dataset_setup_smoke.py`. The script tests deterministic synthetic dual-clock timestamp matching with an explicit known offset and uncertainty, performs a temporary synthetic LeRobot episode write/read roundtrip using the repository's version compatibility adapter, and validates the resulting two-camera + 19D state + 7D action input contract. Optional `--raw` independently audits an already captured real raw episode. It **does not** command the robot, invent real 7D actions, establish actual cross-host clock calibration, or prove the complete RECAP loader. **New script has been committed but not yet executed in the workstation environment; results are pending.** See `milestone_4_dataset_setup_offline_check.md` for one combined command and explicit limitations. A successful smoke confirms offline software interface viability, not the existence of a real training-ready LeRobot dataset. Real simultaneous capture alignment, accepted action labels and RECAP end-to-end remain downstream experiment data-pipeline tasks.
