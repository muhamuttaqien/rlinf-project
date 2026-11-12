# FR3 controller diagnostics: staged source patch only

## Purpose

A supervised +X normalized SpaceMouse pulse of 0.03 for 0.1 s produced
3 ROS publishes and only 0.007 mm reported X motion
(0.011 mm maximum TCP displacement). ROS publication does not prove
controller receipt, IK target validity, torque interface writes, or
physical acceptance.

The active NUC workspace was observed at
`~/Desktop/aist-gamma/franka_controller_ws`.
The process loaded the library from that workspace's `build/` tree.
Its source showed normalized Twist -> controller-side 0.007 m linear
and 0.03 rad angular pose-offset coefficients; current mounting orientation
was [0,0,0] and command timeout was 0.25 s.

## Instrumentation scope

`fr3_controller_diagnostic_stage.py` generates **staged copies** and
a unified patch using exact source anchors. It does NOT alter the NUC's
source workspace, build or install the library, restart the controller,
subscribe/publish to ROS topics, or command an FR3.

The proposed patch adds lightweight relaxed atomic counters for:

- Callback: total ROS Twist requests, moving requests, rejected-range
  requests, and latest moving X input
- Non-RT IK timer: solver attempts, valid and invalid results
- RT update: number of ticks passing the reference freshness/validity
  gate and ticks where joint reference actually advanced
- RT effort output: counts of successful calls to the
  `set_value()` command interface; NOT confirmed hardware torque acceptance

A non-RT timer reports these numbers once per second as
`FR3_DIAG_V1` in the controller process logs. It does not log from
the real-time control loop.

## Stage for review only

On the **workstation**:

```bash
git pull --ff-only origin branch-openpi-recap-franka
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit_tests/test_fr3_controller_diagnostic_stage.py

scp toolkits/realworld_check/fr3_controller_diagnostic_stage.py \
  aist-franka@aistfranka-NUC13ANK-B:/tmp/fr3_controller_diagnostic_stage.py
```

On the **NUC**, using a fresh output directory:

```bash
python3 /tmp/fr3_controller_diagnostic_stage.py \
  --workspace ~/Desktop/aist-gamma/franka_controller_ws \
  --output ~/fr3_controller_diag_review
```

Review the resulting
`~/fr3_controller_diag_review/controller_diagnostics_only.patch`.

**STOP HERE for now.** The patch generator only checks that source
anchors match. It is not a C++ compilation, hardware qualification, or
proof of identical source and running binary. Installation requires
an independently reviewed, offline compiled patch and a scheduled safe
maintenance window with the robot disengaged and a rollback plan.
Do not rebuild, overwrite a loaded shared library, reload the controller,
or run further physical pulses in response to the staging script.

## Interpreting future logs (only after separate review/deployment)

When measured over a single supervised pulse:
- `rx` increases, but `moving_rx` does not: request channel invalid
  or neutral, not accepted as moving
- `moving_rx` increases, but `ik_attempt` does not:
  sampled intent/freshness/activation gate may reject it
- `ik_attempt` increases, but `ik_valid` does not:
  IK solution or joint-bound validation fails
- `ik_valid` increases, but `gate_ticks` stays unchanged:
  real-time target freshness/activation matching failed
- `gate_ticks` increases, but `ref_advanced_ticks` does not:
  target and reference are equal, or reference increments are tiny
- `ref_advanced_ticks` increases, and `effort_write_ticks` increases,
  yet measured TCP motion stays near noise: additional controller
  output/hardware-response evidence is required

These cumulative counters do not alone establish per-command causality,
accepted hardware torque, or physical safety. For precise event alignment,
timestamped telemetry is a separate engineering task.
