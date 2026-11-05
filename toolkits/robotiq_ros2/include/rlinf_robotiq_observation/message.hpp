// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <diagnostic_msgs/msg/diagnostic_array.hpp>

#include "rlinf_robotiq_observation/observation.hpp"

namespace rlinf_robotiq_observation {
diagnostic_msgs::msg::DiagnosticArray
observation_message(const Snapshot &sample, const std::string &port,
                    int64_t published_unix_ns);
diagnostic_msgs::msg::DiagnosticArray
command_message(const CommandResult &result, const std::string &owner,
                int64_t published_unix_ns);
} // namespace rlinf_robotiq_observation
