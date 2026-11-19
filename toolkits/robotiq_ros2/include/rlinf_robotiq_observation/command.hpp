// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <mutex>
#include <optional>
#include <string>

#include <robotiq/types.h>

namespace rlinf_robotiq_observation {
struct CommandPolicy {
  bool enabled{false};
  std::string owner;
  double max_speed{0.1}, max_force{0.1}, max_step{0.1};
  int64_t feedback_max_age_ns{500000000};
};

struct PositionRequest {
  std::string owner, authorization_id;
  uint64_t sequence{0};
  int64_t issued_unix_ns{0};
  uint32_t valid_for_ms{0};
  double position{0}, speed{0}, force{0};
};

struct Authorization {
  bool success{false};
  std::string id, reason;
};

struct CommandResult {
  bool accepted{false};
  uint64_t sequence{0};
  std::string stage{"idle"}, reason;
};

// One lease and one pending request, bound to the current serial connection.
class CommandGate {
public:
  explicit CommandGate(CommandPolicy policy);
  void connected();
  void disconnected();
  void observed(const robotiq::GripperFeedback &feedback, int64_t now_ns);
  void failed();
  void shutdown();
  Authorization authorize(const std::string &owner, uint32_t lease_ms,
                          int64_t now_ns);
  Authorization revoke(const std::string &owner, const std::string &id);
  CommandResult submit(const PositionRequest &request, int64_t unix_ns,
                       int64_t monotonic_ns);
  std::optional<CommandResult>
  dispatch(const std::function<bool(const PositionRequest &)> &writer,
           int64_t unix_ns, int64_t monotonic_ns);
  CommandResult result() const;

private:
  void invalidate(const std::string &reason);
  std::string ready(int64_t now_ns) const;
  std::string validate(const PositionRequest &request, int64_t wall,
                       int64_t steady) const;
  CommandPolicy policy_;
  mutable std::mutex mutex_;
  bool connected_{false}, stopped_{false}, awaiting_echo_{false};
  uint8_t expected_echo_{0};
  std::optional<robotiq::GripperFeedback> feedback_;
  int64_t received_ns_{0}, lease_started_ns_{0}, lease_duration_ns_{0};
  std::string authorization_id_;
  uint64_t last_sequence_{0};
  struct Pending {
    PositionRequest request;
    int64_t accepted_ns, remaining_ns;
  };
  std::optional<Pending> pending_;
  CommandResult latest_;
};
} // namespace rlinf_robotiq_observation
