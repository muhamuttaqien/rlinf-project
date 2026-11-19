// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
// Offline integration adapter. Not wired into any robot controller.
#pragma once

#include "toolkits/realworld_check/fr3_rt_telemetry_queue.hpp"

namespace fr3_rt_telemetry {

// Construct ArmTickSnapshot from controller values AFTER its update.
// This adapter only assigns sequence/drop metadata and enqueues a copy.
// Exactly one producer and one consumer; never call capture() from ROS callbacks.
template <std::size_t Capacity = 1024>
class ControllerTelemetryTap final {
 public:
  ControllerTelemetryTap() = default;
  ControllerTelemetryTap(const ControllerTelemetryTap&) = delete;
  ControllerTelemetryTap& operator=(const ControllerTelemetryTap&) = delete;

  bool capture_after_update(const ArmTickSnapshot& observed) noexcept {
    ArmTickSnapshot sample = observed;
    sample.seq = attempted_++;
    sample.telemetry_drops_total_before_this_try = queue_.dropped_total();
    return queue_.try_push(sample);
  }

  // Only the non-RT single consumer may call this.
  bool drain_one(ArmTickSnapshot& sample) noexcept {
    return queue_.try_pop(sample);
  }

  std::uint64_t dropped_total() const noexcept {
    return queue_.dropped_total();
  }

 private:
  std::uint64_t attempted_{0};  // producer thread only
  SpscQueue<ArmTickSnapshot, Capacity> queue_{};
};

}  // namespace fr3_rt_telemetry
