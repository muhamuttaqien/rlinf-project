// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
// Milestone 4C.2T.2: OFFLINE design prototype, NOT a deployed ROS controller.
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace fr3_rt_telemetry {

// These bits mirror the audited controller gates; the producer must set them
// from the actual loop decisions, never recompute them in the recorder.
enum Gate : std::uint8_t {
  kMoving = 1u << 0,
  kIntentFresh = 1u << 1,
  kTargetValid = 1u << 2,
  kActivationMatches = 1u << 3,
  kStateFresh = 1u << 4,
  kCommandFresh = 1u << 5,
};

// A fixed-layout snapshot copied *after* the controller reference update.
// All timestamps are NUC monotonic nanoseconds; zero means not available.
// Frame IDs / FK / tool offsets belong in separate non-RT session metadata.
// Sequence numbers count attempted samples, including dropped samples.
// No ownership of memory outside this struct, no ROS types, no pointers.
struct ArmTickSnapshot final {
  std::uint64_t seq{};
  std::uint64_t controller_tick_seq{};
  std::uint64_t telemetry_drops_total_before_this_try{};
  std::uint64_t tick_ns{};
  std::uint64_t activation_ns{};
  std::uint64_t intent_recv_ns{};
  std::uint64_t target_state_ns{};
  std::uint64_t target_command_ns{};
  std::uint64_t target_activation_ns{};
  std::uint64_t sample_ns{};
  std::uint64_t ik_solve_ns{};
  std::array<double, 6> intent_axes{};
  std::array<double, 7> measured_q_rad{};
  std::array<double, 7> ik_target_q_rad{};
  std::array<double, 7> reference_before_rad{};
  std::array<double, 7> reference_after_rad{};
  std::array<double, 7> velocity_limits_rad_s{};
  double dt_s{};
  double command_timeout_s{};
  std::uint8_t gate_bits{};
  bool reference_advanced{};
};

static_assert(std::is_trivially_copyable<ArmTickSnapshot>::value,
              "RT snapshots must remain trivially copyable");
// This offline prototype targets the x86_64 NUC; do not use on a machine
// where std::atomic<uint64_t> would require hidden locking.
static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "64-bit atomics must be always lock-free");

// One and ONLY one producer (the controller loop) and one consumer
// (a dedicated *non-realtime* drain thread). Drop newest on overflow.
// Do not replace this with a multi-producer queue without re-design.
// The object's lifetime must outlive both threads; never reset it in use.
template <typename T, std::size_t Capacity>
class SpscQueue final {
 public:
  static_assert(std::is_trivially_copyable<T>::value,
                "Only fixed-size trivially-copyable records are supported");
  static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0,
                "Capacity must be a power of two and at least 2");

  SpscQueue() = default;
  SpscQueue(const SpscQueue&) = delete;
  SpscQueue& operator=(const SpscQueue&) = delete;

  bool try_push(const T& sample) noexcept {
    const auto head = head_.load(std::memory_order_relaxed);
    const auto tail = tail_.load(std::memory_order_acquire);
    if (head - tail == Capacity) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    slots_[head & (Capacity - 1)] = sample;
    head_.store(head + 1, std::memory_order_release);
    return true;
  }

  bool try_pop(T& output) noexcept {
    const auto tail = tail_.load(std::memory_order_relaxed);
    const auto head = head_.load(std::memory_order_acquire);
    if (tail == head) return false;
    output = slots_[tail & (Capacity - 1)];
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }

  std::uint64_t dropped_total() const noexcept {
    return dropped_.load(std::memory_order_relaxed);
  }

 private:
  alignas(64) std::atomic<std::uint64_t> head_{0};
  alignas(64) std::atomic<std::uint64_t> tail_{0};
  alignas(64) std::atomic<std::uint64_t> dropped_{0};
  std::array<T, Capacity> slots_{};
};

// Enqueue only a precomputed POD snapshot: no I/O, heap allocation, mutex,
// ROS call, timing call, or dynamic serialization on the producer path.
// Atomic primitives being lock-free is not a measured worst-case latency
// guarantee; this prototype MUST be benchmarked before any RT deployment.

}  // namespace fr3_rt_telemetry
