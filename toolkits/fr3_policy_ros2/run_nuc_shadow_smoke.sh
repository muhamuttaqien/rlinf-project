#!/usr/bin/env bash
# Isolated ROS2 M5 smoke test; does NOT touch the live arm/gripper ROS domains.
set -eo pipefail
# ROS setup scripts can reference unset shell variables; restore nounset afterward.
if [ "$(id -u)" -eq 0 ]; then echo 'Do not run ROS shadow test as root' >&2; exit 2; fi
ROOT="$HOME/fr3_policy_bridge_m5"
PKG="$ROOT/src/fr3_policy_ros2"
if [ ! -f "$PKG/package.xml" ]; then
  echo "Missing staged package at $PKG" >&2; exit 3
fi
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=220
export RLINF_FR3_SHADOW_ONLY=1
# No controller-manager calls, robot topics or active overlay sourced.
colcon build --base-paths "$ROOT/src" --packages-select rlinf_fr3_policy_bridge \
  --build-base "$ROOT/build" --install-base "$ROOT/install" --log-base "$ROOT/log" \
  --event-handlers console_direct+
source "$ROOT/install/local_setup.bash"
set -u
ros2 run rlinf_fr3_policy_bridge fr3_policy_shadow >"$ROOT/shadow_node.log" 2>&1 &
NODE_PID=$!
trap 'kill "$NODE_PID" 2>/dev/null || true; wait "$NODE_PID" 2>/dev/null || true' EXIT
# Client discovers publisher and verifies a synthetic heartbeat+request roundtrip.
ros2 run rlinf_fr3_policy_bridge fr3_policy_shadow_smoke
printf 'PASS: isolated NUC ROS2 build and synthetic message roundtrip; no active controller changed\n'
