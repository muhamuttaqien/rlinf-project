# Milestone 4B.3 — FR3 Arm Controller Safety Validation

**Status: SOURCE-LEVEL AUDIT COMPLETE; PHYSICAL VALIDATION PENDING.**

This is a verification record and a supervised test protocol. No code in
the RLinf raw recorder controls or moves the FR3.

## Verified on the real NUC (November 6, 2026)

- Active arm controller:
  `franka_arm_controllers/JointImpedanceIKController`
  (`joint_impedance_ik_controller`).
- Controller manager PID `1259561`, `ROS_DOMAIN_ID=70`, maps the
  custom shared object
  `/home/aist-franka/Desktop/aist-gamma/franka_controller_ws/build/franka_arm_controllers/libfranka_arm_controllers.so`.
- Separate controller-manager PID `1274171` uses `ROS_DOMAIN_ID=71`;
  it does not map `franka_arm_controllers`.
- Runtime parameters include `command_timeout: 0.25` s and
  `ik_update_rate: 50.0` Hz. Measured FR3 joint-state topic ran at
  approximately 1000 Hz.
- Source code on NUC:
  `~/Desktop/aist-gamma/franka_controller_ws/src/franka_arm_controllers/src/joint_impedance_ik_controller.cpp`
  modified 2026-10-26 19:41:37 JST.
- Loaded library's build file modified 2026-10-26 19:49:27 JST. This
  is **consistent with** a build after the source modification but is
  **not proof** that the loaded binary matches every line of that source.
- The inspected C++ guards reject stale command and sample timestamps
  using `monotonic_ns()`, a 0.25-s age threshold, and activation
  generation checks. Invalid normalized SpaceMouse axes are turned
  into no-motion input; IK rejects invalid/out-of-limit joint solutions.
- When commands become stale or a target is invalid, the real-time
  update loop **does not advance `q_desired_`**. It continues sending
  finite impedance torques toward the last reference. This is a
  **last-reference impedance hold**, not zero torque, an instantaneous
  stop, an E-stop, or proof of independent mechanical safety.
- The source does not independently demonstrate Cartesian workspace
  barriers, collision monitoring, torque saturation, or compliance
  with the workcell's physical safety requirements.

## Additional live ROS diagnostics — November 6, 2026

Read-only commands on NUC after sourcing both ROS Humble and the
`franka_controller_ws` overlay confirmed:

- `ros2 param get /joint_impedance_ik_controller command_timeout`:
  `Double value is: 0.25`.
- Motion command topic `/franka_controller/target_cartesian_velocity_percent`
  has **one RELIABLE publisher** and **one BEST_EFFORT subscriber**,
  identified on the NUC as `joint_impedance_ik_controller`.
  This QoS pairing is compatible; the publisher name was not resolved
  from the NUC query, but had previously been identified as
  `spacemouse_publisher` from workstation discovery.
- `ros2 bag --help` succeeded, confirming bag recording tooling is
  available.
- DDS emitted `sequence size exceeds remaining buffer` warnings.
  Diagnostics returned results, but the warnings' root cause remains
  unknown and should be monitored during telemetry capture.

**Current state:** read-only diagnostics passed; stationary joint-state
baseline recording, on-site safety approval, neutral/release test, and
authorized physical fault-response validation still pending. Do not
interpret successful ROS discovery as a tested physical safe stop.

## Must verify before any moving-arm test

The on-site qualified FR3 operator/safety supervisor is responsible
for confirming:

1. Workcell clear and risk assessment approved; gripper and attached
   tooling are accounted for. No person is within the motion envelope.
2. Working protective stop/E-stop provisions are accessible and
   understood. Do **not** provoke an E-stop by script; follow the
   lab's approved safety check procedure.
3. Only the intended existing SpaceMouse publisher controls the motion
   topic, and only one arm motion controller owns the relevant effort
   command interfaces. Never start another command publisher.
4. The robot starts at rest in a known safe posture with adequate
   joint/workspace clearance; physical speed settings follow the
   lab's low-speed safety procedure.
5. A recording method is available to preserve command and measured
   joint state on the NUC (same host), ideally with `ros2 bag`:
   `/franka_controller/target_cartesian_velocity_percent` and
   `/franka_robot_state_broadcaster/measured_joint_states`.
   ROS bag receipt timestamps are not proof of controller command
   acceptance, so do not equate these to executed 7D policy actions.

## Supervised verification sequence (human-operated)

- **Stationary baseline:** with SpaceMouse neutral, collect measured
  state and command topic data for 5–10 seconds. Confirm no unexpected
  motion, no controller faults, and one command source.
- **Neutral/release behavior:** only after on-site clearance, a trained
  operator may make a deliberately small motion using the **existing**
  physical SpaceMouse and then return it to neutral. Observe and
  record whether the arm settles to its hold reference without
  excessive motion or new faults. **This tests neutral-command hold,
  not the nonzero stale-command fault branch.**
- **True command-loss watchdog validation:** only under an approved
  lab-specific fault-injection procedure, with qualified safety
  personnel and independent protective means. A safe, documented
  setup must explicitly verify the stale-command branch and limit
  post-loss motion. Never abruptly kill or unplug the command source
  while the arm is moving as an ad-hoc test. Do not issue nonzero
  commands through scripts or launch a second publisher.
- Stop immediately on unexpected acceleration, drift, excessive
  force, controller fault, communication instability, or personnel
  entering the motion envelope.

## Read-only diagnostic on the NUC (safe to run while stationary)

```bash
source /opt/ros/humble/setup.bash
source "$HOME/Desktop/aist-gamma/franka_controller_ws/install/setup.bash"
export ROS_DOMAIN_ID=70
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 param get /joint_impedance_ik_controller command_timeout
ros2 topic info -v /franka_controller/target_cartesian_velocity_percent
ros2 topic info -v /franka_robot_state_broadcaster/measured_joint_states
ros2 bag --help | head -n 8
```

If `ros2 bag` is available, on-site supervisor may run a **read-only**
recording of the preapproved low-risk test:

```bash
OUT="$HOME/fr3_watchdog_audit_$(date +%Y%m%d_%H%M%S)"
ros2 bag record -o "$OUT" \
  /franka_controller/target_cartesian_velocity_percent \
  /franka_robot_state_broadcaster/measured_joint_states
# Stop recording with Ctrl+C after the supervised procedure.
```

Recording does not send commands. Keep recordings local and provide
human observations plus bag metadata for review.

## Acceptance gates (all required to close 4B.3)

- [x] Runtime controller/plugin identified and unique domain-70 motion
      command source observed in previous ROS topic diagnostics.
- [x] Source-level timeout, reference-hold fallback, joint limit and
      command validity checks inspected.
- [x] Library file timestamp is later than the source file timestamp
      (limited build-provenance evidence).
- [ ] Deployed artifact build provenance established beyond timestamps,
      or explicitly documented as a residual uncertainty by the local
      maintainer/safety reviewer.
- [ ] On-site safety prerequisites and independent stop provisions
      confirmed by the qualified operator/supervisor.
- [ ] Controlled neutral/release behavior tested with measured
      joint-state data, time-aligned commands and a qualitative
      safety observation log.
- [ ] Genuine nonzero stale-command/fault-injection validation completed
      according to an approved lab safety procedure, with documented
      physical response, or explicitly waived through the site's
      formal safety process (not by this assistant).
- [ ] Formal go/no-go decision signed off by responsible lab personnel.

Until these gates are satisfied, **4B.3 is PARTIAL** and no moving-arm
demonstration recording or policy control should be declared cleared.
