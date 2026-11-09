# FR3 SpaceMouse episode-based raw collection

`toolkits/realworld_check/fr3_data_collect.py` adds **manual start/stop** to the existing read-only FR3/Robotiq/two-camera capture. It **does not** publish motion commands or activate/deactivate ROS controllers. It expects the lab-approved FR3 controller, existing SpaceMouse publisher, Robotiq driver, and ROS 2 state streams to be running already. Ensure the operator is physically present, E-stop accessible, and the workcell safe before moving the arm. The RealSense ROS publishers must be stopped before this direct-SDK camera capture; do not start a second SpaceMouse publisher.

Run from the GPU workstation with the existing Python environment and ROS Jazzy setup:

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=70 ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/unit_tests/test_fr3_data_collect.py \
  tests/unit_tests/test_fr3_raw_capture.py

python -m toolkits.realworld_check.fr3_data_collect \
  --output-root "$HOME/fr3_peg_100" \
  --task "FR3 peg insertion" --target-successes 100 \
  --base-serial 337122075473 --wrist-serial 218622277827 \
  --controller-log "/home/aist-franka/fr3_telemetry_isolated_4c2t7/recordings/stationary_telemetry.jsonl"
```

For each attempt, **press Enter to start**, operate the robot under the approved lab procedure, and **press Enter to stop**. Mark `s` for success, `f` for failure, or `a` for aborted. Press `q` at the *start prompt* to end the session. Recording duration is variable, with a **300-second safety cap** by default (`--max-duration-s` adjusts it). Re-running with the same output root resumes at the next unused episode number and counts only successful, completed raw captures. It never overwrites earlier episode directories. Failures and incomplete attempts are retained, not counted toward the requested 100 successes.

```text
fr3_peg_100/
  session.json
  episode_0001/
    manifest.json
    states.jsonl
    operator_commands.jsonl
    episode_review.json
    frames/base/*.npy
    frames/wrist/*.npy
  episode_0002/...
```

`episode_review.json` stores the operator-entered outcome, workstation-local start/end monotonic and Unix timestamps, capture status, and the **NUC telemetry path only**. The NUC JSONL is **not copied, automatically segmented, or cross-host synchronized**. Preserve it separately before restarting or reactivating the instrumented controller; recorder activation can truncate its configured output. For reliable action/observation alignment, calibrate the clock offset, identify accepted-controller actions and validate the 19D state/7D metric action contract; raw SpaceMouse Twist is not a metric target. An outcome marked `success` is an operator label, not proof of training readiness.

**After the first successful attempt**, validate the raw episode before collecting the remaining 99:

```bash
python -m toolkits.realworld_check.fr3_lerobot_preflight \
  --raw "$HOME/fr3_peg_100/episode_0001"
```

This raw preflight must pass before treating the recording as an intact raw episode. It does **not** certify LeRobot, π0.5 SFT, or RECAP readiness. Disk usage can be substantial because both camera streams are stored as lossless NPY frames; verify available storage before a long collection session.
