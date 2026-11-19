# Milestone 6A: first action for 100 FR3 peg-insertion demos

We have already built and tested the isolated controller-target recorder.
Do not repeat the builds. The missing fact is whether it is actually loaded
on the live NUC and writing accepted-reference snapshots.

The operator observed two ros2_control_node processes, one referencing an
isolated configuration. A config path is **not** proof of plugin activation
or data recording, and two processes need ownership clarification.

Single workstation command:

    cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
    git pull --ff-only origin branch-openpi-recap-franka
    ssh nuc-franka 'python3 -' < toolkits/realworld_check/fr3_live_recorder_status.py

This script uses Python stdlib and reads process metadata only:
- process ROS_DOMAIN_ID and two-controller collision clues;
- the actual mapped FR3 controller library, when readable;
- any currently open recorder JSONL files;
- any launch-file telemetry_jsonl_path setting.

It **never** imports ROS, publishes/activates anything, signals a PID,
changes controller parameters, writes a file, or moves the robot.

If the isolated controller is loaded with open raw JSONL, validate actual
telemetry (not automatically 7D training actions). If not, the existing
offline recorder needs a lab-approved single-controller deployment plan
with real-time logging review and rollback. No unattended reconfiguration,
new ROS control manager, or policy motion. If process info is unreadable,
the conclusion is unknown.

After genuine arm and Robotiq accepted-target telemetry, verified clock
calibration and approved action semantics are available, record **one**
supervised peg-insertion episode, validate the true 19D/7D labels, export to
LeRobot, test the official loader once, then collect the remaining 99.
