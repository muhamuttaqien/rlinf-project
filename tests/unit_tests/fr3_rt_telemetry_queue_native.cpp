// Standalone, synthetic-only C++ test executable for the offline SPSC prototype.
#include "toolkits/realworld_check/fr3_rt_telemetry_queue.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>

using fr3_rt_telemetry::ArmTickSnapshot;
using fr3_rt_telemetry::SpscQueue;

void fifo_overflow() {
  SpscQueue<int, 4> q;
  int out = -1;
  assert(!q.try_pop(out));
  for (int i = 0; i < 4; ++i) assert(q.try_push(i));
  assert(!q.try_push(4));
  assert(!q.try_push(5));
  assert(q.dropped_total() == 2);
  for (int i = 0; i < 4; ++i) {
    assert(q.try_pop(out));
    assert(out == i);
  }
  assert(!q.try_pop(out));
  assert(q.try_push(6));
  assert(q.try_pop(out) && out == 6);
  assert(q.dropped_total() == 2);
}

void snapshot_fidelity() {
  SpscQueue<ArmTickSnapshot, 2> q;
  ArmTickSnapshot input{};
  input.seq = 42;
  input.controller_tick_seq = 1234;
  input.tick_ns = 4'000'000'000ULL;
  input.intent_axes = {0.5, -0.4, 0, 0, 0, 0};
  input.ik_target_q_rad[6] = 1.23;
  input.reference_before_rad[6] = 0.2;
  input.reference_after_rad[6] = 0.201;
  input.gate_bits = fr3_rt_telemetry::kMoving |
                    fr3_rt_telemetry::kIntentFresh;
  input.reference_advanced = false;
  assert(q.try_push(input));
  ArmTickSnapshot out{};
  assert(q.try_pop(out));
  assert(out.seq == 42 && out.controller_tick_seq == 1234);
  assert(out.ik_target_q_rad[6] == 1.23);
  assert(out.reference_before_rad[6] == 0.2);
  assert(out.reference_after_rad[6] == 0.201);
  assert(out.intent_axes[1] == -0.4);
  assert(out.gate_bits == input.gate_bits);
  assert(!out.reference_advanced);
}

void concurrent_spsc() {
  constexpr std::uint64_t kAttempts = 200'000;
  SpscQueue<ArmTickSnapshot, 64> q;
  std::atomic<bool> done{false};
  std::uint64_t seen = 0;
  std::uint64_t prev = 0;
  std::thread consumer([&] {
    ArmTickSnapshot r{};
    while (true) {
      if (q.try_pop(r)) {
        if (seen) assert(r.seq > prev);
        prev = r.seq;
        ++seen;
      } else if (done.load(std::memory_order_acquire)) {
        break;
      } else {
        std::this_thread::yield();
      }
    }
  });
  for (std::uint64_t i = 0; i < kAttempts; ++i) {
    ArmTickSnapshot s{};
    s.seq = i;
    s.tick_ns = i + 1;
    q.try_push(s);
  }
  done.store(true, std::memory_order_release);
  consumer.join();
  assert(seen + q.dropped_total() == kAttempts);
  std::cout << "accepted=" << seen << " dropped=" << q.dropped_total() << "\n";
}

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string mode = argv[1];
  if (mode == "fifo") fifo_overflow();
  else if (mode == "snapshot") snapshot_fidelity();
  else if (mode == "concurrent") concurrent_spsc();
  else return 2;
  std::cout << "PASS " << mode << "\n";
}
