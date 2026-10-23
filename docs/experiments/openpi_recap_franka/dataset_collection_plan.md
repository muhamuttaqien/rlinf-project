# Dataset collection plan — currently zero episodes

The experiment requires **two distinct camera views** (base and wrist) and **SpaceMouse EE-space teleoperation**, with all peripherals plugged into the GPU Workstation. NUC hosts arm, gripper and safety controller. There is **no dataset yet**, so SFT and RECAP training are blocked.

## Phase 1: capture synchronized raw records (after hardware connection)

- **Base and wrist camera:** record two real image streams, serial IDs, capture timestamps, image dimensions, color format, exposure/crop settings, and dropped-frame indicators. Do not fake missing frames by inserting black images.
- **NUC measured state:** capture joint positions/velocities, EE/TCP pose and twist, gripper width/state, controller state and robot time. Verify TCP and base frame conventions (including quaternion order).
- **Workstation operator intent:** retain six SpaceMouse EE-motion axes and button events as **operator commands** (normalized input, not physical meters/radians); record timestamp, operator mapping and controller mode.
- **NUC executed command:** record accepted/rejected EE target (actual command semantics and units) separately from human intent and measured motion; also record safety clamps and timestamps.
- **Episode metadata:** episode ID, frame index, task text, human intervention, reset, success/failure at episode end, interruption reasons, and sampling rate.

Robot and Workstation clocks need alignment and a documented tolerance. Align observations with the *intended training action time*, preserving timestamps so frame/action skew is auditable. Robot feedback is authoritative for the executed motion.

## Phase 2: choose action and policy state representation

Existing RLinf `RealworldInputs` expects **19D state** and **(N,7) actions** at the transform input. Existing Franka env nested observation expands to **20** scalar values (7 TCP pose + 6 TCP velocity + 3 force + 3 torque + 1 gripper). No projection is approved yet. Determine which state representation is physically and statistically meaningful; explicitly map raw fields into the selected policy vector.

Decide whether policy actions are Cartesian *delta pose*, velocity/setpoint, absolute pose or some other representation; specify 6 components' coordinate frame, units, rotation convention, gripper encoding, action interval/horizon, and how they are measured from actual robot data. **Do not label raw SpaceMouse velocity percentage directly as metric EE action.**

For the candidate 2-camera OpenPI path, map `base` to `observation/image` and `wrist` to `observation/extra_view_image`; confirm RGB/BGR ordering with the full RLinf loader before exporting.

## Phase 3: export and qualify first LeRobot demonstrations

1. Record a small number of operator-driven episodes with known task language and operator-reviewed success labels (no fictional data).
2. Export to LeRobot with the selected schema, consistent sequence indices, `fps`, and exactly two camera observation fields.
3. Run `toolkits/realworld_check/franka_pi05_contract.py` as the candidate *transform-sample* preflight; verify sample path and 19D/7D mapping before treating a passing result as meaningful.
4. Verify one sample end-to-end through **OpenPI SFT** and **RECAP value** transforms, and through RECAP advantage and CFG data loaders; fix the known `RealworldInputs` versus `FrankaEEInputs` mismatch at its source.
5. Only then calculate dataset normalization stats and train an SFT baseline.

## Phase 4: RECAP

RECAP requires real successful demonstrations and separate policy rollouts (including **failures**), with reliable `is_success` labels for rollout episodes; do not manufacture failure examples or returns before they exist. Then reuse the upstream return, value, advantage and CFG stages with consistent dataset/tags. For initial evaluation, compare SFT against the same SFT checkpoint after RECAP using matched task conditions.

**Current status:** Hardware/data pending. This plan is documentation, not a data-collection application.
