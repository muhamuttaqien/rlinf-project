// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
#include "rlinf_robotiq_observation/message.hpp"

#include <array>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

namespace rlinf_robotiq_observation {
diagnostic_msgs::msg::DiagnosticArray
observation_message(const Snapshot &sample, const std::string &port,
                    int64_t published_unix_ns) {
  diagnostic_msgs::msg::DiagnosticArray message;
  message.header.stamp.sec =
      static_cast<int32_t>(published_unix_ns / 1000000000);
  message.header.stamp.nanosec =
      static_cast<uint32_t>(published_unix_ns % 1000000000);
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "robotiq/observation";
  status.hardware_id = port;
  status.message =
      sample.state + (sample.error.empty() ? "" : ": " + sample.error);
  status.level = sample.state == "failed" ? status.ERROR : status.STALE;
  auto add = [&](const std::string &key, const std::string &value) {
    diagnostic_msgs::msg::KeyValue entry;
    entry.key = key;
    entry.value = value;
    status.values.push_back(entry);
  };
  auto number = [&](const std::string &key, auto value) {
    std::ostringstream text;
    text << std::setprecision(17) << +value;
    add(key, text.str());
  };
  add("schema_version", "1");
  add("state", sample.state);
  add("communication_valid", sample.feedback ? "true" : "false");
  add("last_attempt_valid", sample.last_attempt_valid ? "true" : "false");
  add("position_units", "normalized_closure_0_open_1_closed");
  add("model", "unconfirmed_2F_variant");
  add("metric_width_available", "false");
  number("sequence", sample.sequence);
  number("last_success_unix_ns", sample.last_success_unix_ns);
  number("last_success_monotonic_ns", sample.last_success_monotonic_ns);
  number("request_started_unix_ns", sample.request_started_unix_ns);
  number("age_s", sample.age_s);
  if (sample.feedback) {
    const auto &f = *sample.feedback;
    status.level = (f.raw_fault & 0x0f) ? status.WARN : status.OK;
    number("position", f.position);
    number("commanded_position", f.commanded_position);
    number("raw_position", f.raw_position);
    number("raw_commanded_position", f.raw_commanded_position);
    number("current_normalized", f.current);
    number("raw_current", f.raw_current);
    number("current_ampere", f.raw_current * 0.01);
    number("activation_status", f.status.gact);
    number("action_status", f.status.ggto);
    number("finger_status", f.status.gsta);
    number("object_status", f.status.gobj);
    number("raw_fault", f.raw_fault);
    number("fault_code", f.raw_fault & 0x0f);
    const std::array<const char *, 16> faults = {
        "NONE",
        "UNKNOWN",
        "UNKNOWN",
        "UNKNOWN",
        "UNKNOWN",
        "ACTION_DELAYED",
        "UNKNOWN",
        "ACTIVATION_NEEDED",
        "MAX_TEMP_EXCEEDED",
        "COMM_TIMEOUT",
        "UNDER_VOLTAGE",
        "AUTOMATIC_RELEASE_IN_PROGRESS",
        "INTERNAL_FAULT",
        "ACTIVATION_FAULT",
        "OVERCURRENT",
        "AUTOMATIC_RELEASE_COMPLETED"};
    add("fault_status", faults[f.raw_fault & 0x0f]);
  }
  message.status.push_back(status);
  return message;
}
diagnostic_msgs::msg::DiagnosticArray
command_message(const CommandResult &result, const std::string &owner,
                int64_t stamp) {
  diagnostic_msgs::msg::DiagnosticArray message;
  message.header.stamp.sec = static_cast<int32_t>(stamp / 1000000000);
  message.header.stamp.nanosec = static_cast<uint32_t>(stamp % 1000000000);
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "robotiq/command";
  status.hardware_id = owner;
  status.message = result.stage + ": " + result.reason;
  status.level = result.stage == "failed" ? status.ERROR : status.OK;
  for (const auto &pair : std::vector<std::pair<std::string, std::string>>{
           {"schema_version", "1"},
           {"stage", result.stage},
           {"sequence", std::to_string(result.sequence)},
           {"owner", owner},
           {"grasp_outcome", "unknown"}}) {
    diagnostic_msgs::msg::KeyValue value;
    value.key = pair.first;
    value.value = pair.second;
    status.values.push_back(value);
  }
  message.status.push_back(status);
  return message;
}
} // namespace rlinf_robotiq_observation
