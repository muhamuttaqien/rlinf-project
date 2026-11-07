# M5 — Isolated ROS2 policy integration (no physical experiments)

**Implementation:** `toolkits/fr3_policy_ros2/` is a deployable standalone
`ament_python` package. Unlike the earlier mathematical preview, it has real
`rclpy` subscribers, heartbeat handling, a typed ROS 2 `Twist` publisher
(only to a private **shadow** topic), and a smoke-test client. Package source
contains no publisher to the physical FR3 command topic, no Robotiq action
client, and no controller-manager call. It refuses ROS domains 70 and 71 and
rejects publishing a Twist topic that has been remapped from the shadow name.

**Local gate verification:** 27/27 unit tests pass using Python `pytest` in
the assistant's isolated environment. No ROS installation is present there;
**the actual NUC ROS build/message roundtrip is not yet verified**.

## One workstation command: stage and run on isolated NUC

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
tar -C toolkits -cf - fr3_policy_ros2 | ssh nuc-franka \
  'mkdir -p "$HOME/fr3_policy_bridge_m5/src" && tar -xf - -C "$HOME/fr3_policy_bridge_m5/src" && bash "$HOME/fr3_policy_bridge_m5/src/fr3_policy_ros2/run_nuc_shadow_smoke.sh"'
```

Expected final line: `PASS: isolated NUC ROS2 build and synthetic message
roundtrip; no active controller changed`.

This is a complete **non-actuating software handoff** for Milestone 5 only.
It cannot substitute for an approved metric-to-controller mapping, NUC-enforced
exclusive publisher/controller ownership, TCP/rotation and gripper calibration,
physical stop behavior, or a supervised physical robot qualification in M6.
Those physical steps cannot be responsibly simulated away or declared passed.
