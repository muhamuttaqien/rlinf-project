# FR3 + OpenPI π0.5 / RECAP — Roadmap Status

**Updated:** November 7, 2026 (JST). Status reflects completed offline tests and observed hardware work; it does not imply clearance for autonomous robot operation.

| Milestone | Objective | Status | Remaining gate |
| --- | --- | --- | --- |
| 0 — Hardware Audit | Workstation, NUC, FR3, cameras, SpaceMouse, Robotiq | **Complete** | — |
| 1 — System & Interface Audit | RLinf and ROS 2 architecture | **Complete** | — |
| 2A — Robotiq Observation Bridge | Read-only gripper feedback | **Complete** | — |
| 2B — Robotiq Hardware & DDS | DDS communication and physical FC03 qualification | **Partial** | Formal FC03/serial-owner qualification not signed off |
| 2C.1 — Robotiq Commands (Offline) | Command interface and offline tests | **Complete** | — |
| 2C.2 — Robotiq Commands (Physical) | Supervised gripper operation | **Complete** | — |
| 3A — Multimodal Observations (Offline) | Arm, gripper and two cameras | **Complete** | — |
| 3B — Live Observation Qualification | Live observation sampling | **Complete** | Cross-host/exposure sync is a later dataset gate |
| 4 — Teleoperation & Dataset Collection | Record and export training-ready demonstrations | **Partial** | Safety-qualified movement, accepted-action capture, clock alignment, real LeRobot dataset |
| 5 — π0.5 Inference Integration | Policy inference and action adapter | **Not started** | Requires approved action interface and safety gates |
| 6 — End-to-End Validation | Real manipulation task evaluation | **Not started** | Requires Milestone 5 |

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
| 4C.3/4C.4 — LeRobot export + OpenPI/RECAP data path | **Not started** |

**Bottom line:** Milestones **0–3B** are complete except **2B partial**; **Milestone 4 is in progress**; **5 and 6 are not started**. The 4C.2T prototypes are offline only and are **not** deployed on the NUC. No policy-ready 7D action dataset exists yet.
