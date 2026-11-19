// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
// Non-realtime JSONL drain. Never call from a controller update callback.
#pragma once

#include "toolkits/realworld_check/fr3_rt_telemetry_capture.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <stdexcept>

namespace fr3_rt_telemetry {

inline void json_num(std::ostream& out, double x) {
  if (!std::isfinite(x)) throw std::runtime_error("Nonfinite telemetry value");
  out << x;
}

template <std::size_t N>
void json_array(std::ostream& out, const std::array<double, N>& values) {
  out << '[';
  for (std::size_t i = 0; i < N; ++i) {
    if (i) out << ',';
    json_num(out, values[i]);
  }
  out << ']';
}

inline void json_stamp(std::ostream& out, std::uint64_t stamp) {
  if (stamp) out << stamp;
  else out << "null"; // Do not invent timestamps missing from the source.
}

inline void write_snapshot_jsonl(std::ostream& out, const ArmTickSnapshot& s) {
  // The schema intentionally differs from the action-ready validator schema.
  // These are raw controller references, not 7D training labels.
  out << std::setprecision(17)
      << R"({"schema":"fr3_controller_capture_raw_v1","event":"arm_tick","seq":)"
      << s.seq
      << R"(,"controller_tick_seq":)" << s.controller_tick_seq
      << R"(,"tick_ns":)";
  json_stamp(out, s.tick_ns);
  out << R"(,"activation_ns":)";
  json_stamp(out, s.activation_ns);
  out << R"(,"intent_recv_ns":)";
  json_stamp(out, s.intent_recv_ns);
  out << R"(,"target_state_ns":)";
  json_stamp(out, s.target_state_ns);
  out << R"(,"target_command_ns":)";
  json_stamp(out, s.target_command_ns);
  out << R"(,"target_activation_ns":)";
  json_stamp(out, s.target_activation_ns);
  out << R"(,"sample_ns":)";
  json_stamp(out, s.sample_ns);
  out << R"(,"ik_solve_ns":)";
  json_stamp(out, s.ik_solve_ns);
  out << R"(,"intent_axes":)";
  json_array(out, s.intent_axes);
  out << R"(,"measured_q_rad":)";
  json_array(out, s.measured_q_rad);
  out << R"(,"ik_target_q_rad":)";
  if (s.gate_bits & kTargetValid) json_array(out, s.ik_target_q_rad);
  else out << "null";
  out << R"(,"reference_before_rad":)";
  json_array(out, s.reference_before_rad);
  out << R"(,"reference_after_rad":)";
  json_array(out, s.reference_after_rad);
  out << R"(,"velocity_limits_rad_s":)";
  json_array(out, s.velocity_limits_rad_s);
  out << R"(,"dt_s":)";
  json_num(out, s.dt_s);
  out << R"(,"command_timeout_s":)";
  json_num(out, s.command_timeout_s);
  out << R"(,"gate_bits":)" << static_cast<unsigned>(s.gate_bits)
      << R"(,"reference_advanced":)" << (s.reference_advanced ? "true" : "false")
      << R"(,"telemetry_drops_total_before_this_try":)"
      << s.telemetry_drops_total_before_this_try
      << R"(,"training_action":null,"policy_ready":false})" << '\n';
  if (!out) throw std::runtime_error("Telemetry recorder write failed");
}

// Must run exclusively on a non-RT consumer thread/process.
// Caller owns stream/session lifecycle and must save final queue drop total.
template <std::size_t Capacity>
std::size_t drain_jsonl(ControllerTelemetryTap<Capacity>& tap, std::ostream& out) {
  std::size_t saved = 0;
  ArmTickSnapshot row{};
  while (tap.drain_one(row)) {
    write_snapshot_jsonl(out, row);
    ++saved;
  }
  return saved;
}
}  // namespace fr3_rt_telemetry
