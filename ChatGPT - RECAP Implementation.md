# ChatGPT - RECAP Implementation

> **Project continuity / handoff file — last updated 2026-11-18 (Japan time).**
>
> Repository: [muhamuttaqien/rlinf-project](https://github.com/muhamuttaqien/rlinf-project)  
> Working branch: **branch-openpi-recap-franka**  
> Primary current focus: **FR3 real-world π₀.₅ deployment and RECAP data collection**  
> Status: **Controller startup dependency resolved; diagnostic controller is configured but INACTIVE; physical movement cause remains unresolved.**
>
> This is a living summary, not proof of hardware readiness. **In a new ChatGPT/Codex conversation, explicitly ask the assistant to read this file from the branch and check the latest code, logs, and runtime state. ChatGPT does not automatically open a repository document in every conversation.** Update this file whenever a milestone or blocker changes.

## 1. Research objectives and scope

**End goal:** Implement and evaluate RECAP for real-world Franka FR3 peg insertion using an already-trained OpenPI π₀.₅ policy in the RLinf framework.

Practical sequence:

1. Collect and validate real-world operator demonstrations.
2. Supervised fine-tune (SFT) π₀.₅ on the collected demonstration/request dataset.
3. Qualify the actual FR3 command-to-motion interface, ROS topics, coordinate frames, safety controls and gripper bridge.
4. Run **20 genuine human-supervised real π₀.₅ peg-insertion rollout outcomes** (success/failure); aborted or invalid attempts are retained separately and **must not** count as usable outcomes.
5. Use those outcomes for the offline RECAP pipeline: reward/critic/advantage-related evaluation and subsequent policy-improvement experiments, with appropriate baselines.
6. Assess real-world task success, robustness and safety, not just offline prediction metrics.

**Immediate objective as of 2026-11-18:** Diagnose why the FR3 barely responded to movement requests. First resolve **why the instrumented arm controller is inactive**; then obtain one approved, bounded physical test with diagnostic counters. **Do not retrain the policy or start RECAP fine-tuning while this hardware qualification remains blocked.**

A separate simulation track exists in the broader RLinf/RECAP work; it must not be confused with or used as evidence of successful physical FR3 execution. Confirm simulation milestones from their own current logs before making claims.

## 2. Equipment and execution environments

| Component | Known configuration |
| --- | --- |
| Robot | Franka Research 3 (FR3) arm with Robotiq 2F-85 gripper |
| Control PC / NUC | Hostname **aistfranka-NUC13ANK-B**; ROS 2 **Humble**; hardware and custom controller |
| Lab inference workstation | Hostname **pca03395**; ROS 2 **Jazzy**; RLinf/OpenPI Python environment with CUDA GPU |
| Arm ROS domain | **ROS_DOMAIN_ID=70** |
| Robotiq domain | **ROS_DOMAIN_ID=71**, through an existing cross-domain gripper bridge |
| ROS middleware | Previously used **rmw_fastrtps_cpp**, **ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET** |
| Arm command | **/franka_controller/target_cartesian_velocity_percent**, **geometry_msgs/msg/Twist**, six normalized request channels, **not SI velocities** |
| Gripper command | **/gripper_client/target_gripper_width_percent**, **std_msgs/msg/Float32** |
| Pose | **/franka_robot_state_broadcaster/current_pose**; additional measured joints, desired joints, wrench and robot status available from broadcaster |
| Arm controller | **franka_arm_controllers/JointImpedanceIKController**, named **joint_impedance_ik_controller** |
| Cameras for live policy runner | Base and wrist RealSense cameras; script defaults **337122075473** (base), **218622277827** (wrist); verify connected serials before use |
| Inference Python | Project training virtual environment (Python 3.11 in prior runs) |
| Workstation ROS Python | **/usr/bin/python3** with Jazzy / ROS bindings |
| NUC ROS Python | ROS Humble environment |

Do not assume ROS domains, overlays, and middleware are inherited correctly between shells or machines. Do not connect domain 213 test processes to the physical robot: **ROS_DOMAIN_ID=213, ROS_LOCALHOST_ONLY=1** was used for isolated offline C++ tests only.

### Important NUC paths

- Normal arm controller workspace: **~/Desktop/aist-gamma/franka_controller_ws**
- FR3 ROS/hardware workspace: **~/Desktop/aist-gamma/franka_ros2_ws**
- Isolated diagnostic workspace: **~/fr3_controller_diag_build**
- Isolated diagnostic package prefix: **~/fr3_controller_diag_build/install/franka_arm_controllers**
- Runtime Pinocchio 4.1.0 libraries: **/opt/openrobots/lib**
- Diagnostic startup log used during the latest session: **~/fr3_diag_launch.log**
- Existing staging/review directory: **~/fr3_controller_diag_review** (verify its current contents)
- Sourceable helper committed to repo: **toolkits/realworld_check/fr3_controller_env.sh**
- Temporary helper copy used in earlier NUC investigation: **/tmp/fr3_controller_env_investigation.sh** (temporary location; prefer tracked helper after synchronizing the repo)

### Controller configuration and request semantics

- Control update rate configured at **1,000 Hz**; IK timer nominally **50 Hz**.
- Command freshness timeout: **0.25 s**.
- Mounting RPY previously observed: **[0, 0, 0]**; verify actual active configuration.
- Translational controller target: **0.007 × normalized linear request**, in metres, applied to the forward-kinematics pose computed from **measured joint positions**.
- Angular request: controller postmultiplies by RPY rotation of **0.03 × normalized angular request** radians.
- Example direct X request **0.03** implies a **nominal 0.21 mm instantaneous target offset**, NOT 0.21 mm motion per ROS publication and NOT a velocity in m/s.
- Updated pose targets replace prior IK targets; repeated publications refresh intent, not guaranteed accumulated travel.
- Joint reference advances toward fresh valid IK targets with per-cycle velocity limiting; neutral / expired intent holds the last joint reference and is **not a safety-rated stop**.
- Controller uses a joint impedance torque expression with configured proportional gains **[600,600,600,600,250,150,50]** and derivative gains **[30,30,30,30,10,10,5]**. **Do not change gains or bypass limits to compensate for small observed motion.**
- The existing real hardware configuration is **aist_fr3_config.yaml**, with **load_gripper: false** in the investigation. Do not switch to example config with different gripper settings without review.

The controller computes targets using the URDF flange chain **base → fr3_link8**. The broadcaster's **O_T_EE** and the physical Robotiq peg TCP may differ; tool-frame and axis-direction verification remains outstanding.

## 3. Dataset, trained policy and artifacts

### Demonstrations and derived labels

- **100** physical teleoperated peg-insertion episodes collected; **80 train / 20 held-out** source episodes.
- Prior derived dataset had **15,366** complete 7D candidate request labels (**12,288 train; 3,078 held-out**) at approximately **0.1 s observation intervals**.
- Labels represent **time-window-averaged normalized SpaceMouse Twist requests**, not proven controller-accepted commands, physical Cartesian velocities, or measured joint torques.
- Seven output dimensions: **linear_x, linear_y, linear_z, angular_x, angular_y, angular_z, latched_gripper_open_request**.
- Inputs use robot state / camera data; the runner constructs multimodal observations, with arm and gripper readings across ROS domains 70 and 71.
- The SpaceMouse hardware-to-Twist mapping was already applied by the publisher; **do not apply the HID axis transformation a second time** to recorded Twist labels.

### SFT model and relevant workstation paths

- Model: **OpenPI π₀.₅**, approximately **3.35B parameters**, trained in RLinf.
- SFT training: **3,000 steps**, completed; no retraining needed to address current hardware blocker.
- Consolidated checkpoint:

~~~
/data1/Muttaqien/logs/20261111-04:55:26-realworld_sft_openpi_fr3_peg_request_full3000_wandb/fr3_peg_pi05_request_offline_full3000_wandb/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt
~~~

- Train-only normalization statistics directory:

~~~
/data1/Muttaqien/fr3_peg_request_norm_v1
~~~

- Base π₀.₅ RLinf checkpoint:

~~~
/data1/Muttaqien/rlinf_data/checkpoints/torch/pi05_base_rlinf
~~~

- Default intended real-rollout output root (script constant):

~~~
/data1/Muttaqien/fr3_pi05_real_rollouts_20
~~~

- Previously generated offline artifacts include **/data1/Muttaqien/fr3_request_sft_heldout_eval_all.json**, **/data1/Muttaqien/fr3_request_transition_diagnostic.json**, and shadow traces under **/data1/Muttaqien/fr3_shadow_v1/**. Verify the files still exist before using them.
- Selected offline held-out evaluation previously reported **arm 6D MAE ≈ 0.1313**, **RMSE ≈ 0.1882** in normalized request units, and gripper-open classification accuracy **≈ 0.9733** for the **300 nonoverlapping 10-step windows** evaluation. These metrics are **not task success rates** and do not approve hardware execution.

### Core project scripts

- **toolkits/realworld_check/fr3_pi05_supervised_rollout.py** — one operator-confirmed physical policy rollout per invocation, episode artifacts and outcomes.
- **toolkits/realworld_check/fr3_pi05_ros_executor.py** — Jazzy ROS arm/gripper command process, exclusivity, stale-pose and travel guards.
- **toolkits/realworld_check/fr3_pi05_request_adapter.py** — validated normalized 7D action request handling and gripper debounce.
- **toolkits/realworld_check/fr3_supervised_arm_pulse.py** — single low-amplitude X qualification tool; default read-only preflight, physical action requires explicit operator confirmation.
- **toolkits/realworld_check/fr3_controller_diagnostic_stage.py** — creates staged controller instrumentation; does not change active NUC workspace.
- **toolkits/realworld_check/fr3_controller_env.sh** — sourceable ROS Humble/Pinocchio environment fix.
- **docs/experiments/openpi_recap_franka/fr3_controller_investigation.md** — **authoritative detailed technical investigation and rollback**, including source-verified controller semantics and test evidence.
- **docs/experiments/openpi_recap_franka/fr3_startup_fix/controller_motion_tests.patch** — test-only C++ regression cases (apply only to an appropriate isolated copy; original inspected isolated test source already included it).

The policy executor defaults to **legacy_limited** with **ARM_GAIN=0.15**, reducing an example predicted request 0.03 to 0.0045 (nominal target offset 0.0315 mm). This explains additional attenuation in earlier policy rollouts, but **does not explain the direct pulse**, which bypasses that multiplier. **identity_verified** (gain 1.0) is gated behind explicit post-qualification physical mapping approval. Do **not** enable it merely because offline axis names match; 10 Hz inference/playback versus ~100 Hz teleoperation and physical frame/timing still need validation.

The rollout runner contains safety checks and an operator-reviewed outcome workflow; ROS message publication alone is not proof of controller acceptance or task completion. Never count aborted/invalid attempts as success or failure toward the target 20.

## 4. Completed work and investigation results

| Area | Status | Evidence / qualification |
| --- | --- | --- |
| 100 FR3 demonstrations | **Completed** | 80 training + 20 held-out sources |
| π₀.₅ SFT, 3,000 steps | **Completed** | Consolidated checkpoint reload and offline inference |
| Offline held-out request evaluation | **Completed** | Request prediction metrics; not physical peg insertion |
| Live observations and shadow inference | **Previously exercised** | Does not prove physical actuation |
| Early physical π₀.₅ policy rollouts | **Inconclusive / not successful motion qualification** | Maximal physical movement was only a few millimetres in reported limited-gain trials; some aborted due to out-of-range predictions |
| Arm-only direct physical qualification | **Insufficient movement** | 0.1 s / 3 publications: max displacement 0.011 mm; 0.5 s / 11 publications: delta [0.030,-0.003,-0.031] mm, max displacement 0.045 mm |
| Physical direction / six-axis request mapping | **NOT verified** | 0.5 s pulse did not meet the tool's 0.2 mm directional detection threshold |
| Controller diagnostic source instrumentation | **Compiled successfully in isolation** | FR3_DIAG_V1 counters added without changing motion equations |
| Pinocchio controller-startup dependency | **Resolved in tested environment** | Pinocchio 4.1.0 files exist under /opt/openrobots/lib; original failure due to loader path missing |
| NUC hardware/diagnostic library symbol loading | **Passed without hardware initialization** | Corrected LD_LIBRARY_PATH, no required library unresolved |
| Python regression tests | **27 passed** | Codex investigation, prior to latest physical launch |
| C++ controller tests | **8 passed** | Isolated NUC build, localhost domain 213; two added small-request tests |
| Physical diagnostic counters during X pulse | **NOT obtained** | Need active diagnostic controller and operator-approved supervised test |
| 20 real π₀.₅ outcomes | **Pending** | Cannot collect until physical controller qualification |
| RECAP real-world critic/advantage/policy improvement | **Pending** | Requires usable physical outcome data |

**What offline C++ tests did establish:** At a fixed example joint pose, both positive and negative 0.03 X requests generated approximately ±0.21 mm target equilibria; repeated requests with stationary simulated measured joints do not accumulate travel, while feeding ideal measured motion makes targets advance. This rules out a universal small-request software deadband at that fixture pose only. It does **not** prove IK validity or torque execution at the actual physical robot pose.

### Startup failure and resolution

The earlier diagnostic launch crashed before controller initialization:

~~~text
pluginlib::LibraryLoadException
libpinocchio_parsers.so.4.1.0: cannot open shared object file
~~~

Codex confirmed that the correct Pinocchio 4.1.0 shared libraries were already installed under **/opt/openrobots/lib**, but ROS overlays did not include that directory in **LD_LIBRARY_PATH**. It added the committed sourceable helper. Offline **ldd -r** and **ctypes RTLD_NOW** hardware/diagnostic library loads subsequently passed. No ABI swap, system-wide library modification or new dependency install was required.

### Detailed command-path uncertainty

Physical weakness remains **unexplained**. Instrumented evidence must distinguish:

1. ROS delivery/exclusivity (**rx**, **moving_rx**, **invalid_rx**, **last_x**).
2. Current-state sampling and request freshness (**0.25 s** timeout).
3. IK attempt and validity (**ik_attempt**, **ik_valid**, **ik_failed**).
4. Target gate and reference movement (**gate_ticks**, **ref_advanced_ticks**).
5. Torque interface writes (**effort_write_ticks**; a write count does **not** prove hardware torque acceptance).
6. Real hardware response, joint tracking, TCP movement and fault status.

Competing neutral publishers, command gaps, IK failure at the real pose, limited joint tracking, hardware state or fault conditions remain **hypotheses**, not established diagnoses. The log warnings **sequence size exceeds remaining buffer** also remain unexplained; record exact source/timing rather than assuming they cause small motion.

## 5. **Latest live state — 2026-11-18**

**This section overrides the older report's statement that no physical startup had occurred.** A later operator-performed launch was attempted after the runtime path fix.

Latest NUC launch excerpt:

- ROS controller manager initialized with **1,000 Hz** update rate and real-time FIFO scheduling priority 50.
- Hardware initialization/activation and controller loading proceeded past the previous Pinocchio loader failure.
- **joint_state_broadcaster** configured and activated.
- **franka_robot_state_broadcaster** configured and activated.
- **joint_impedance_ik_controller** **loaded and configured**, but **did not activate**.
- Various ROS nodes printed **sequence size exceeds remaining buffer**.
- **No FR3_DIAG_V1 lines** appeared in **~/fr3_diag_launch.log**.

The user ran the sourceable diagnostic helper and verified the package selection:

~~~text
ros2 control list_controllers

joint_state_broadcaster        joint_state_broadcaster/JointStateBroadcaster             active
joint_impedance_ik_controller  franka_arm_controllers/JointImpedanceIKController         inactive
franka_robot_state_broadcaster franka_robot_state_broadcaster/FrankaRobotStateBroadcaster active

ros2 pkg prefix franka_arm_controllers
/home/aist-franka/fr3_controller_diag_build/install/franka_arm_controllers
~~~

A grep of **~/fr3_diag_launch.log** for joint controller activation/errors showed **Configuring controller 'joint_impedance_ik_controller'** and robot name **fr3**, but **no explicit successful activation, activation attempt, or activation failure**. Package-prefix selection confirms the overlay, **not necessarily the exact binary loaded in memory**. **FR3_DIAG_V1 logs are gated on controller active state**, so their absence with an inactive controller is expected, although the binary selection should still be checked.

**Critical interpretation:** The current blocker is **inactive arm controller after the latest launch**. This accounts for missing live diagnostic lines in this session. It does **not** retrospectively explain the earlier weak physical pulses, which occurred in different controller sessions.

## 6. Next steps — resume here, one issue at a time

### **NEXT ACTION: read-only inspection of the actual launch file**

On the NUC, in the shell where the diagnostic overlay is sourced:

~~~bash
LAUNCH="$(ros2 pkg prefix franka_arm_controllers)/share/franka_arm_controllers/launch/joint_impedance_ik_controller.launch.py"

grep -n -C 10 -E \
  'joint_impedance_ik_controller|--inactive|--load-only|--activate' \
  "$LAUNCH"
~~~

Purpose: determine whether the launch intentionally starts the controller in an inactive/configured state or whether activation was attempted and stalled. Share the **actual grep output** before recommending any control operation. If the launch file delegates to another launch or spawner, inspect the referenced file or spawner arguments next. Do **not** blindly call **ros2 control switch_controllers** or attempt manual activation: activating effort control can command real robot torque.

### After the activation issue is understood

1. With an authorized operator physically present, hardware stop accessible, workcell clear, and no SpaceMouse/π₀.₅ command publishers: follow the lab's normal reviewed controller activation procedure. If startup fails, stop and collect relevant errors. Do not bypass FR3 safety.
2. Verify the actual controller is **active**, effort interfaces claimed, diagnostic package/binary selection correct and baseline **FR3_DIAG_V1** counters appearing. Capture at least two baseline log lines. If absent, **do not pulse**.
3. Verify ROS arm domain 70 endpoint counts, exclusive intended publisher, fresh broadcaster pose/joints and no unexpected faults.
4. Only after explicit operator approval, perform **one** bounded low-amplitude X pulse using the existing **fr3_supervised_arm_pulse.py** tool (default read-only preflight first; use the documented explicit interactive execution path only after all preconditions hold). Keep original request magnitude, timeout and safety limits; no automatic repeat or gain change.
5. Compare before/after diagnostic **counter deltas**, raw joint/tcp feedback and hardware state. This identifies whether the bottleneck is ROS reception, freshness, IK, reference gate, or hardware response. Note: a counter staying flat may be normal after the same target has been reached; use the full trace, not a single last-step value.
6. Only then qualify all six axes, frames, timing, gripper bridge and real mapping. Do not remove the existing **0.15** limiter or enable **identity_verified** without verified controller, frame and timing behavior.
7. Once a bounded supervised policy execution is physically verified, collect **20 operator-reviewed real π₀.₅ peg-insertion outcomes**; preserve aborts/faults separately. Then proceed to RECAP offline critic/advantage/optimization and proper experiment evaluation.

**Safety:** Neutral ROS messages and controller timeout are not hardware emergency stops. Never issue robot movement, activate effort controllers, reset faults, change controller gains/speed/scales, install live libraries, or restart active hardware without explicit operator authorization and normal safety procedures. Do not use offline test success to infer physical readiness.

## 7. GitHub history and collaboration

Important verified investigation commits on **branch-openpi-recap-franka**:

- **cfb7dc98** — [Pinocchio runtime dependency fix](https://github.com/muhamuttaqien/rlinf-project/commit/cfb7dc98)
- **f4a1f295** — [Controller pipeline investigation and offline C++ tests](https://github.com/muhamuttaqien/rlinf-project/commit/f4a1f295)

Other prior changes in this effort include the instrumented diagnostic generator and the updated 0.5 s bounded pulse. Before changing files, inspect the current branch contents and current live hardware status rather than assuming earlier commands or state persist.

**Repository policy for future Codex/assistant work:** Use **branch-openpi-recap-franka**; commit and push source, tests, concise findings and documentation to that branch; verify remote commits; avoid unrelated modifications and force pushes; do not commit compiled libraries, large datasets, credentials or proprietary device state; preserve the normal NUC controller source, checkpoint and existing backups. Physical operator actions are never implied by repository-write permission.

The detailed report **docs/experiments/openpi_recap_franka/fr3_controller_investigation.md** contains the precise startup reproduction, verified source semantics, offline tests, diagnostic interpretation and rollback procedure. This root file captures **newer live controller state** and should be updated after each significant hardware session.

## 8. Instructions for a future ChatGPT/Codex conversation

Suggested first message:

> **Read "ChatGPT - RECAP Implementation.md" at the root of the GitHub repository muhamuttaqien/rlinf-project on branch branch-openpi-recap-franka. Then read docs/experiments/openpi_recap_franka/fr3_controller_investigation.md. Use them as project context, verify the latest repository revision and the most recent NUC logs I provide, and resume from "Next steps" in the root handoff file. Our immediate blocker is that joint_impedance_ik_controller is inactive after startup; do not skip to physical pulses or model retraining. Please keep your replies concise and prioritize one actionable step at a time.**

When continuing:

- Distinguish **confirmed physical observations** from **offline/source-only results** and hypotheses.
- Use the most recent live robot/controller outputs to update status; a Markdown record can become stale.
- Do not ask again for established environment facts unless a change or conflict is observed.
- Be focused: first make the controller's activation state understandable, then collect an actual instrumented motion trace, then proceed to π₀.₅/RECAP. Avoid repeated testing and unnecessary infrastructure.
- When a new durable fact is established, update **this file** and commit/push that change.

---
**End of handoff snapshot — 2026-11-18.**
