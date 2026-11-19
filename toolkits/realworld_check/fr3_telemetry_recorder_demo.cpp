// Standalone synthetic example: saves raw controller-reference records to JSONL.
// No ROS, no device access, no controller loading.
#include "toolkits/realworld_check/fr3_rt_telemetry_recorder.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: fr3_recorder_demo /path/to/output.jsonl\n";
    return 2;
  }
  using namespace fr3_rt_telemetry;
  ControllerTelemetryTap<2> tap;
  ArmTickSnapshot s{};
  s.tick_ns = 1'000'000'000ULL;
  s.activation_ns = 900'000'000ULL;
  s.intent_recv_ns = 999'900'000ULL;
  s.target_state_ns = 999'900'000ULL;
  s.target_command_ns = 999'900'000ULL;
  s.target_activation_ns = 900'000'000ULL;
  s.intent_axes[0] = 0.5;
  s.ik_target_q_rad.fill(0.2);
  s.velocity_limits_rad_s.fill(1.0);
  s.dt_s = .001;
  s.command_timeout_s = .25;
  s.gate_bits = kMoving | kIntentFresh | kTargetValid |
                kActivationMatches | kStateFresh | kCommandFresh;
  s.reference_advanced = true;

  s.controller_tick_seq = 1;
  s.reference_before_rad.fill(0.0);
  s.reference_after_rad.fill(0.001);
  if (!tap.capture_after_update(s)) return 3;

  s.controller_tick_seq = 2;
  s.tick_ns += 1'000'000;
  s.intent_axes[0] = 0.0;
  s.gate_bits = kIntentFresh | kTargetValid |
                kActivationMatches | kStateFresh | kCommandFresh;
  s.reference_advanced = false;
  s.reference_before_rad.fill(0.001);
  s.reference_after_rad.fill(0.001);
  if (!tap.capture_after_update(s)) return 3;

  // Force one loss to demonstrate attempted sequence/drop provenance.
  s.controller_tick_seq = 3;
  s.tick_ns += 1'000'000;
  if (tap.capture_after_update(s) || tap.dropped_total() != 1) return 3;

  std::ofstream output(argv[1], std::ios::out | std::ios::trunc);
  if (!output) {
    std::cerr << "Cannot open output file\n";
    return 4;
  }
  const std::size_t saved = drain_jsonl(tap, output);
  output.close();
  if (!output) return 5;
  std::cout << "SYNTHETIC OFFLINE RECORDER: saved=" << saved
            << " dropped=" << tap.dropped_total()
            << " path=" << argv[1] << '\n';
  return saved == 2 ? 0 : 6;
}
