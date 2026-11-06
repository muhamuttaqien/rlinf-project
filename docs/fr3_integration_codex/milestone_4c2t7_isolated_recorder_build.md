# 4C.2T.7 — Recorder linked to isolated FR3 ROS 2 controller

**Status: implementation committed; NUC isolated build pending.** The NUC synthetic recorder demo from 4C.2T.6 reported `saved=2 dropped=1 path=synthetic.jsonl`.

This script clones the existing isolated 4C.2T.5 controller, adds a **separate non-real-time consumer thread** that drains the SPSC queue into JSONL, and compiles a separate `~/fr3_telemetry_isolated_4c2t7` package. The recorder is *disabled by default* through `telemetry_jsonl_path=""`. It starts only when that parameter is explicitly configured and the plugin is activated. The thread stops and joins at deactivation, cleanup and destruction.

**Workstation action:**

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
tar -czf - toolkits/realworld_check/stage_fr3_recorder_controller.py toolkits/realworld_check/fr3_rt_telemetry_recorder.hpp | ssh nuc-franka 'mkdir -p ~/fr3_recorder_stage_transfer && tar -xzf - -C ~/fr3_recorder_stage_transfer && python3 ~/fr3_recorder_stage_transfer/toolkits/realworld_check/stage_fr3_recorder_controller.py --build'
```

Expected (if NUC compilation succeeds): `1 package finished` and `ISOLATED RECORDER CONTROLLER BUILD PASSED; live FR3 untouched.`

**Build only.** Do not launch or load the isolated controller, set the recording parameter, send robot commands, or source the isolated install into the live ROS workspace. The original controller and the earlier isolated build are unchanged. No live telemetry is being written at this stage.

This is raw telemetry, not validated training actions. Safety, actual RT overhead, recorder lifecycle, timing synchronization and the physical motion gate must be reviewed before any deployment.
