# FR3 policy ROS2 **shadow-only** bridge (Milestone 5)

This package is the first **actual ROS 2 transport integration** for FR3
policy messages, using an isolated NUC-local ROS domain. It consumes an
explicitly synthetic 7D JSON request and independent heartbeat, checks owner,
sequence, finite values, time and candidate bounds, and publishes **only to**
`/rlinf/fr3/shadow/normalized_twist_preview` (never the live controller).
The 7th channel is represented as a binary **preview status**, not a real
Robotiq action. It has no robot-control switch or production enable mode.

It refuses ROS domain 70 (FR3) and 71 (Robotiq), and refuses ROS topic
remapping of the Twist preview output. Policy requests must be same-host
monotonic, synthetic-only fixtures; a workstation model cannot bypass this
or send arbitrary real pi0.5 predictions. The original controller runs
unchanged.

**One-command isolated NUC smoke run** (after copying the package to
`~/fr3_policy_bridge_m5/src/fr3_policy_ros2`):

```bash
bash ~/fr3_policy_bridge_m5/src/fr3_policy_ros2/run_nuc_shadow_smoke.sh
```

This script sources ROS Humble, compiles this separate `ament_python` package,
launches the shadow node in `ROS_DOMAIN_ID=220`, sends a synthetic heartbeat
and request, checks the private Twist preview, then stops the node. It never
starts/stops the real controller. If the build or smoke check fails, Milestone
5 ROS transport verification is **not** complete.

**Boundaries:** The source-derived `0.007 m` / `0.03 rad` per normalized axis
are *illustrative* increments, not certified safety limits; rotation mapping
and gripper action authority are not physically established. A separate
NUC-enforced, mutually exclusive, calibrated production controller interface
is needed *before* physically moving with a policy. No user-facing software
result can replace on-site safety approval.
