// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <robotiq/robotiq_gripper_interface.h>

#include "rlinf_robotiq_observation/command.hpp"

namespace rlinf_robotiq_observation {

// A process-edge claim: the alternate proc root supports simulated OS tests.
class PortClaim {
public:
  explicit PortClaim(std::string port, std::string proc_root = "/proc");
  ~PortClaim();
  PortClaim(const PortClaim &) = delete;
  PortClaim &operator=(const PortClaim &) = delete;
  void acquire();
  void release() noexcept;

private:
  std::string port_, proc_root_;
  int lock_fd_{-1};
};

// One serial owner for observations and explicitly authorized position
// requests.
class SerialObserver {
public:
  SerialObserver(std::string port, unsigned timeout_ms,
                 std::string proc_root = "/proc",
                 std::shared_ptr<CommandGate> commands = nullptr);
  ~SerialObserver();
  void connect();
  robotiq::GripperFeedback read();
  std::optional<CommandResult> execute_pending();
  void disconnect() noexcept;

private:
  std::string port_;
  unsigned timeout_ms_;
  PortClaim claim_;
  std::unique_ptr<robotiq::RobotiqGripperInterface> device_;
  std::shared_ptr<CommandGate> commands_;
};

struct Snapshot {
  std::string state{"missing"};
  std::string error;
  bool last_attempt_valid{false};
  uint64_t sequence{0};
  int64_t last_success_unix_ns{0};
  int64_t last_success_monotonic_ns{0};
  int64_t request_started_unix_ns{0};
  double age_s{-1};
  std::optional<robotiq::GripperFeedback> feedback;
};

// Failed attempts immediately invalidate measurements without erasing history.
class ObservationCache {
public:
  void received(const robotiq::GripperFeedback &feedback, int64_t unix_ns,
                int64_t monotonic_ns, int64_t request_started_unix_ns);
  void failed(const std::string &error);
  Snapshot snapshot(int64_t now_monotonic_ns, int64_t max_age_ns) const;

private:
  mutable std::mutex mutex_;
  Snapshot latest_;
  std::optional<robotiq::GripperFeedback> feedback_;
};

int64_t monotonic_ns();
int64_t unix_ns();

} // namespace rlinf_robotiq_observation
