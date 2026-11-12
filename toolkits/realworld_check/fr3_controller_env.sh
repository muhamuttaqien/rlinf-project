#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# Source on the FR3 NUC in a fresh Bash shell. Optional argument: diagnostic install prefix.
# This changes only the calling shell's environment; it starts no ROS processes.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' 'Source this file to configure the current shell.' >&2
    exit 1
fi

_fr3_controller_env() {
    local root="$HOME/Desktop/aist-gamma"
    local diagnostic="${1:-}"
    local setup
    if [[ -n "${ROS_DISTRO:-}" && "$ROS_DISTRO" != humble ]]; then
        printf '%s\n' 'Expected a fresh shell or ROS Humble; use this file on the NUC.' >&2
        return 1
    fi
    for setup in /opt/openrobots/lib/libpinocchio_{parsers,default}.so.4.1.0 \
        /opt/ros/humble/setup.bash \
        "$root/franka_ros2_ws/install/setup.bash" \
        "$root/franka_controller_ws/install/setup.bash"; do
        if [[ ! -r "$setup" ]]; then
            printf 'Required NUC runtime file is missing: %s\n' "$setup" >&2
            return 1
        fi
    done
    if [[ -n "$diagnostic" && ! -r "$diagnostic/local_setup.bash" ]]; then
        printf 'Expected diagnostic install prefix with local_setup.bash: %s\n' "$diagnostic" >&2
        return 1
    fi
    source /opt/ros/humble/setup.bash || return 1
    source "$root/franka_ros2_ws/install/setup.bash" || return 1
    source "$root/franka_controller_ws/install/setup.bash" || return 1
    if [[ -n "$diagnostic" ]]; then
        source "$diagnostic/local_setup.bash" || return 1
    fi
    # libfranka 0.18 on this NUC needs these exact 4.1 SONAMEs; ROS also supplies 4.0.
    export LD_LIBRARY_PATH="/opt/openrobots/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ROS_DOMAIN_ID=70
}

if _fr3_controller_env "$@"; then
    unset -f _fr3_controller_env
else
    unset -f _fr3_controller_env
    return 1
fi
