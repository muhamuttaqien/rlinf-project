# 4C.2T.6 — Non-realtime telemetry recorder

**Ready to test on workstation; no hardware changes.** The non-RT C++ JSONL drain writes IK joint targets, before/after `q_desired_`, measured joints, input axes, validity, timestamps and drop provenance from the previously built SPSC capture tap. The recorder is **not attached to the actual controller**, and no live telemetry file can exist yet.

Run this one workstation command:

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" python -m pytest -q tests/unit_tests/test_fr3_telemetry_recorder.py
```

To inspect a tangible JSONL file without using ROS:

```bash
g++ -std=c++17 -O2 -I . toolkits/realworld_check/fr3_telemetry_recorder_demo.cpp -o /tmp/fr3_recorder_demo
/tmp/fr3_recorder_demo /tmp/fr3_synthetic_telemetry.jsonl
head -n 1 /tmp/fr3_synthetic_telemetry.jsonl
```

The file uses schema `fr3_controller_capture_raw_v1` to avoid conflating raw snapshots with the unfulfilled training-label contract. Missing `ik_solve_ns` and `sample_ns` are serialized as null; no invented timestamps or actions. The demo's full-queue drop count is reported on stdout. Real integration requires a separate non-RT consumer lifecycle design, lab review and a safety-approved deployment. Do not source, load or restart the isolated controller plugin on the real robot.

## November 7, 2026 workstation verification

The operator ran `tests/unit_tests/test_fr3_telemetry_recorder.py` on the workstation with Python 3.12.3 and pytest 9.2.0: **1 passed in 0.63 s**. This validates the synthetic recorder JSONL test. The separate manually invoked demo command's output has not yet been reported; neither test provides live controller telemetry.

## Standalone demo verified — November 7, 2026

The workstation operator executed `/tmp/fr3_recorder_demo /tmp/fr3_synthetic_telemetry.jsonl` followed by `wc -l`. The observed output was:

```text
SYNTHETIC OFFLINE RECORDER: saved=2 dropped=1 path=/tmp/fr3_synthetic_telemetry.jsonl
2 /tmp/fr3_synthetic_telemetry.jsonl
```

This verifies synthetic JSONL file creation and loss reporting only. It does not mean the isolated ROS 2 controller has a connected consumer or that real robot data have been recorded.
