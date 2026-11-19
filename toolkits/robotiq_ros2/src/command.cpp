// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
#include "rlinf_robotiq_observation/command.hpp"

#include <sys/random.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <stdexcept>

namespace rlinf_robotiq_observation {
namespace {
bool fraction(double value) {
  return std::isfinite(value) && value >= 0 && value <= 1;
}
std::string new_authorization() {
  std::array<unsigned char, 16> bytes;
  size_t used = 0;
  while (used < bytes.size()) {
    const auto n =
        ::getrandom(bytes.data() + used, bytes.size() - used, GRND_NONBLOCK);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      throw std::runtime_error("Authorization entropy unavailable");
    used += static_cast<size_t>(n);
  }
  const char *digits = "0123456789abcdef";
  std::string result;
  for (auto value : bytes) {
    result += digits[value >> 4];
    result += digits[value & 15];
  }
  return result;
}
} // namespace

CommandGate::CommandGate(CommandPolicy policy) : policy_(std::move(policy)) {
  if ((policy_.enabled && policy_.owner.empty()) || policy_.owner.size() > 64 ||
      !fraction(policy_.max_speed) || !fraction(policy_.max_force) ||
      !fraction(policy_.max_step) || policy_.max_step == 0 ||
      policy_.feedback_max_age_ns <= 0 ||
      policy_.feedback_max_age_ns > 500000000)
    throw std::invalid_argument(
        "Expected owner, normalized command limits and positive feedback age");
}

void CommandGate::invalidate(const std::string &reason) {
  if (pending_)
    latest_.sequence = pending_->request.sequence;
  if (pending_ || latest_.stage != "failed") {
    latest_.accepted = false;
    latest_.stage = pending_ ? "dropped" : "revoked";
    latest_.reason = reason;
  }
  pending_.reset();
  authorization_id_.clear();
}
void CommandGate::connected() {
  std::lock_guard<std::mutex> lock(mutex_);
  invalidate("New connection requires authorization");
  feedback_.reset();
  awaiting_echo_ = false;
  connected_ = !stopped_;
}
void CommandGate::disconnected() {
  std::lock_guard<std::mutex> lock(mutex_);
  invalidate("Serial session disconnected");
  connected_ = false;
  feedback_.reset();
  awaiting_echo_ = false;
}
void CommandGate::failed() {
  std::lock_guard<std::mutex> lock(mutex_);
  invalidate("Communication failed; explicit reauthorization required");
  feedback_.reset();
}
void CommandGate::shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  stopped_ = true;
  connected_ = false;
  feedback_.reset();
  invalidate("Command gate shut down");
}
void CommandGate::observed(const robotiq::GripperFeedback &feedback,
                           int64_t now) {
  std::lock_guard<std::mutex> lock(mutex_);
  feedback_ = feedback;
  received_ns_ = now;
  if (!feedback.valid || feedback.raw_fault != 0 ||
      feedback.status.gflt != robotiq::NONE ||
      feedback.status.gact != robotiq::ACTIVATED ||
      feedback.status.gsta != robotiq::ACTIVATION_COMPLETE ||
      !fraction(feedback.position) ||
      (feedback.status.ggto != robotiq::STOPPED &&
       feedback.status.ggto != robotiq::GOTO_POSITION) ||
      static_cast<int>(feedback.status.gobj) < 0 ||
      static_cast<int>(feedback.status.gobj) > 3) {
    invalidate(
        "Device not activated/healthy; no activation or recovery attempted");
    return;
  }
  if (awaiting_echo_ && feedback.raw_commanded_position == expected_echo_)
    awaiting_echo_ = false;
}

std::string CommandGate::ready(int64_t now) const {
  if (!policy_.enabled)
    return "Commands disabled";
  if (stopped_ || !connected_)
    return "No command-capable serial session";
  if (!feedback_ || !feedback_->valid || now < received_ns_ ||
      now - received_ns_ > policy_.feedback_max_age_ns)
    return "Feedback missing or stale";
  const auto &f = *feedback_;
  if (f.raw_fault || f.status.gflt != robotiq::NONE ||
      f.status.gact != robotiq::ACTIVATED ||
      f.status.gsta != robotiq::ACTIVATION_COMPLETE || !fraction(f.position))
    return "Device not activated/healthy";
  if ((f.status.ggto != robotiq::STOPPED &&
       f.status.ggto != robotiq::GOTO_POSITION) ||
      static_cast<int>(f.status.gobj) < 0 ||
      static_cast<int>(f.status.gobj) > 3)
    return "Device status invalid";
  if (f.status.ggto == robotiq::GOTO_POSITION &&
      f.status.gobj == robotiq::IN_MOTION)
    return "Device reports motion in progress";
  if (awaiting_echo_)
    return "Previous request echo not yet observed";
  return {};
}

Authorization CommandGate::authorize(const std::string &owner, uint32_t ms,
                                     int64_t now) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto error = ready(now);
  if (!error.empty())
    return {false, {}, error};
  if (owner != policy_.owner || owner.empty())
    return {false, {}, "Owner not configured"};
  if (ms == 0 || ms > 1000)
    return {false, {}, "Lease must be in [1, 1000] ms"};
  if (!authorization_id_.empty()) {
    if (now >= lease_started_ns_ &&
        now - lease_started_ns_ < lease_duration_ns_)
      return {false, {}, "An owner lease is already active"};
    invalidate("Previous authorization expired");
  }
  try {
    authorization_id_ = new_authorization();
  } catch (const std::exception &e) {
    return {false, {}, e.what()};
  }
  lease_started_ns_ = now;
  lease_duration_ns_ = static_cast<int64_t>(ms) * 1000000;
  last_sequence_ = 0;
  return {true, authorization_id_, "Authorized current connection only"};
}
Authorization CommandGate::revoke(const std::string &owner,
                                  const std::string &id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (owner != policy_.owner || id.empty() || id != authorization_id_)
    return {false, {}, "Authorization does not own this lease"};
  invalidate("Explicitly revoked");
  return {true, {}, "Revoked; no stop/motion command sent"};
}

std::string CommandGate::validate(const PositionRequest &r, int64_t wall,
                                  int64_t steady) const {
  const auto error = ready(steady);
  if (!error.empty())
    return error;
  if (r.owner != policy_.owner || r.authorization_id.empty() ||
      r.authorization_id != authorization_id_)
    return "Command does not own an authorized lease";
  if (steady < lease_started_ns_ ||
      steady - lease_started_ns_ >= lease_duration_ns_)
    return "Authorization expired";
  if (r.valid_for_ms == 0 || r.valid_for_ms > 500 || r.issued_unix_ns <= 0 ||
      wall < r.issued_unix_ns ||
      wall - r.issued_unix_ns >= static_cast<int64_t>(r.valid_for_ms) * 1000000)
    return "Command timestamp/validity invalid or expired";
  if (!fraction(r.position) || !fraction(r.speed) || !fraction(r.force) ||
      r.speed > policy_.max_speed || r.force > policy_.max_force ||
      std::abs(r.position - feedback_->position) > policy_.max_step ||
      std::abs(std::floor(r.position * 255.0) / 255.0 - feedback_->position) >
          policy_.max_step)
    return "Command exceeds finite normalized range or configured limits";
  return {};
}
CommandResult CommandGate::submit(const PositionRequest &r, int64_t wall,
                                  int64_t steady) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto error = validate(r, wall, steady);
  if (error.empty() && (r.sequence == 0 || r.sequence <= last_sequence_))
    error = "Sequence replay/out of order";
  if (error.empty() && pending_)
    error = "One request is already pending";
  if (!error.empty())
    return {false, r.sequence, "rejected", error};
  pending_ = Pending{r, steady,
                     static_cast<int64_t>(r.valid_for_ms) * 1000000 -
                         (wall - r.issued_unix_ns)};
  last_sequence_ = r.sequence;
  latest_ = {true, r.sequence, "accepted",
             "Queued only; no device acknowledgment or grasp confirmation"};
  return latest_;
}
std::optional<CommandResult> CommandGate::dispatch(
    const std::function<bool(const PositionRequest &)> &writer, int64_t wall,
    int64_t steady) {
  // Revocation and dispatch are serialized through the point of wire
  // acknowledgment.
  std::lock_guard<std::mutex> lock(mutex_);
  if (!pending_)
    return std::nullopt;
  const auto pending = *pending_;
  pending_.reset();
  auto error = validate(pending.request, wall, steady);
  if (error.empty() && (steady < pending.accepted_ns ||
                        steady - pending.accepted_ns >= pending.remaining_ns))
    error = "Queued request expired in monotonic time";
  if (!error.empty()) {
    latest_ = {false, pending.request.sequence, "dropped", error};
    return latest_;
  }
  bool acknowledged = false;
  try {
    acknowledged = writer(pending.request);
  } catch (const std::exception &e) {
    error = e.what();
  }
  if (!acknowledged) {
    invalidate("Write/acknowledgment failed; outcome unknown, no retry");
    feedback_.reset();
    latest_ = {false, pending.request.sequence, "failed",
               error.empty() ? "No valid FC16 acknowledgment; outcome unknown"
                             : error};
  } else {
    expected_echo_ = static_cast<uint8_t>(pending.request.position * 255.0);
    awaiting_echo_ = true;
    latest_ = {true, pending.request.sequence, "acknowledged",
               "Register write acknowledged; motion and grasp outcome unknown"};
  }
  return latest_;
}
CommandResult CommandGate::result() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return latest_;
}
} // namespace rlinf_robotiq_observation
