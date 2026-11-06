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
| 4C.3/4C.4 — LeRobot export + OpenPI/RECAP data path | **Not started** |

**Bottom line:** Milestones **0–3B are operationally complete**; **Milestone 4 is in progress**; **5 and 6 are not started**. The 4C.2T prototypes are offline only and are **not** deployed on the NUC. No policy-ready 7D action dataset exists yet.
