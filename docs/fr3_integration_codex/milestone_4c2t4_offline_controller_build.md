# Milestone 4C.2T — Separate offline reference-step build

**Status:** Small standalone C++17 controller-reference model built and tested offline. **Not the ROS 2 controller plugin** and **not deployed** on the NUC.

The real FR3 controller remains in the separate NUC workspace. The tested offline model copies the audited reference-update behavior (moving/freshness/activation/IK gates and velocity-limited `q_desired_` update) and calls our existing nonblocking `ControllerTelemetryTap` after the update. The model does **not** reproduce FK/IK, torque control, ROS lifecycle or real robot timing.

Files: `toolkits/realworld_check/fr3_offline_controller_reference.hpp`, `tests/unit_tests/fr3_offline_controller_reference_native.cpp`, and `tests/unit_tests/test_fr3_offline_controller_reference.py`.

**Three synthetic checks:** reference calculation identical with/without telemetry; stale input holds reference with gate reason recorded; full telemetry buffer drops samples without altering the reference.

One test command on the workstation:

```bash
git pull --ff-only origin branch-openpi-recap-franka
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" python -m pytest -q tests/unit_tests/test_fr3_offline_controller_reference.py
```

**Workstation verified November 7, 2026:** 3/3 tests passed in 0.68 s (Python 3.12.3, pytest 9.2.0). The separate offline C++ controller-reference model is verified; no ROS plugin or hardware build was tested.

**Limits:** The deployed ROS controller's real source/binary was not compiled, instrumented or deployed here. The existing `Target` struct did not expose a separate IK-solve timestamp in the inspected code; the prototype deliberately preserves `ik_solve_ns=0` rather than inventing it. The proposed v1 Python telemetry validator requires positive `ik_solve_ns` and is not wired to this stand-in. A future collector must extend the source telemetry and/or version the schema, and must still verify frame/timing semantics and receive lab safety approval before real-robot telemetry is enabled. No 7D training labels are generated.
