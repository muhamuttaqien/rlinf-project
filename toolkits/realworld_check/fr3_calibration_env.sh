#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# Source this file; it starts no ROS context, node, controller or publisher.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '%s\n' 'Source fr3_calibration_env.sh in a fresh Bash shell.' >&2
  exit 1
fi

_fr3_calibration_environment() {
  local overlay="${1:-/tmp/fr3_calibration/jazzy_msgs/install}"
  local repository
  repository="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)" || return
  if [[ -n "${ROS_DISTRO:-}" && "${ROS_DISTRO}" != jazzy ]]; then
    printf '%s\n' 'Expected a fresh Jazzy shell; another ROS distro is sourced.' >&2
    return 1
  fi
  if [[ ! -r "$overlay/local_setup.bash" ]]; then
    printf 'Missing verified Jazzy message overlay: %s\n' "$overlay" >&2
    return 1
  fi
  source /opt/ros/jazzy/setup.bash || return
  source "$overlay/local_setup.bash" || return
  export PYTHONPATH="$repository${PYTHONPATH:+:$PYTHONPATH}"
  export ROS_DOMAIN_ID=70 ROS_LOCALHOST_ONLY=0
  /usr/bin/python3 - <<'PY'
from franka_msgs.msg import FrankaRobotState
from controller_manager_msgs.srv import ListControllers
from rclpy.type_support import check_for_type_support

check_for_type_support(FrankaRobotState)
check_for_type_support(ListControllers)
PY
}

if _fr3_calibration_environment "$@"; then
  unset -f _fr3_calibration_environment
else
  unset -f _fr3_calibration_environment
  return 1
fi
