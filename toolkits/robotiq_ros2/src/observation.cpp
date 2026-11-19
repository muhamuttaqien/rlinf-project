// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
#include "rlinf_robotiq_observation/observation.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <system_error>

namespace rlinf_robotiq_observation {
namespace fs = std::filesystem;

PortClaim::PortClaim(std::string port, std::string proc_root)
    : port_(std::move(port)), proc_root_(std::move(proc_root)) {}
PortClaim::~PortClaim() { release(); }

void PortClaim::acquire() {
  if (lock_fd_ >= 0)
    return;
  struct stat target {};
  if (::stat(port_.c_str(), &target) || !S_ISCHR(target.st_mode))
    throw std::runtime_error("Expected an existing serial character device: " +
                             port_);
  const auto lock = "/tmp/rlinf-robotiq-" +
                    std::to_string(major(target.st_rdev)) + "-" +
                    std::to_string(minor(target.st_rdev)) + ".lock";
  lock_fd_ =
      ::open(lock.c_str(), O_CREAT | O_RDONLY | O_CLOEXEC | O_NOFOLLOW, 0666);
  struct stat info {};
  if (lock_fd_ < 0 || ::fstat(lock_fd_, &info) || !S_ISREG(info.st_mode) ||
      info.st_nlink != 1 || ::flock(lock_fd_, LOCK_EX | LOCK_NB)) {
    release();
    throw std::runtime_error("Serial claim unavailable: " + lock);
  }
  try {
    // TIOCEXCL prevents later opens; this check rejects pre-existing owners.
    for (const auto &process : fs::directory_iterator(proc_root_)) {
      const auto pid = process.path().filename().string();
      if (pid.empty() ||
          !std::all_of(pid.begin(), pid.end(),
                       [](unsigned char c) { return std::isdigit(c); }))
        continue;
      std::error_code error;
      fs::directory_iterator handles(process.path() / "fd", error);
      if (error == std::errc::no_such_file_or_directory)
        continue;
      if (error)
        throw std::runtime_error("Cannot inspect serial owners: " +
                                 error.message());
      for (const auto &handle : handles) {
        struct stat opened {};
        if (::stat(handle.path().c_str(), &opened)) {
          if (errno == ENOENT)
            continue;
          throw std::runtime_error("Cannot inspect serial descriptor: " +
                                   handle.path().string());
        }
        if (S_ISCHR(opened.st_mode) && opened.st_rdev == target.st_rdev)
          throw std::runtime_error("Serial device already open in process " +
                                   pid);
      }
    }
  } catch (...) {
    release();
    throw;
  }
}

void PortClaim::release() noexcept {
  if (lock_fd_ >= 0) {
    ::close(lock_fd_);
    lock_fd_ = -1;
  }
}

SerialObserver::SerialObserver(std::string port, unsigned timeout_ms,
                               std::string proc_root,
                               std::shared_ptr<CommandGate> commands)
    : port_(std::move(port)), timeout_ms_(timeout_ms),
      claim_(port_, std::move(proc_root)), commands_(std::move(commands)) {
  if (timeout_ms == 0 || timeout_ms > 1000)
    throw std::invalid_argument("Expected serial timeout in [1, 1000] ms");
}
SerialObserver::~SerialObserver() { disconnect(); }

void SerialObserver::connect() {
  if (device_)
    return;
  claim_.acquire();
  try {
    auto device = std::make_unique<robotiq::RobotiqGripperInterface>();
    device->set_timeout(timeout_ms_);
    if (!device->connect(port_, 115200, 1.0, 0.0, true))
      throw std::runtime_error("Robotiq serial connection failed");
    device_ = std::move(device);
    if (commands_)
      commands_->connected();
  } catch (...) {
    device_.reset();
    claim_.release();
    throw;
  }
}

robotiq::GripperFeedback SerialObserver::read() {
  if (!device_)
    throw std::runtime_error("Expected connected observation session");
  try {
    auto feedback = device_->get_feedback();
    if (!feedback.valid)
      throw std::runtime_error("No valid FC03 feedback before deadline");
    if (commands_)
      commands_->observed(feedback, monotonic_ns());
    return feedback;
  } catch (...) {
    if (commands_)
      commands_->failed();
    throw;
  }
}

std::optional<CommandResult> SerialObserver::execute_pending() {
  if (!commands_)
    return std::nullopt;
  const auto result = commands_->dispatch(
      [this](const PositionRequest &request) {
        return device_ && device_->request_position(
                              request.position,
                              static_cast<uint8_t>(request.speed * 255.0),
                              static_cast<uint8_t>(request.force * 255.0));
      },
      unix_ns(), monotonic_ns());
  if (result && result->stage == "failed")
    throw std::runtime_error(result->reason);
  return result;
}

void SerialObserver::disconnect() noexcept {
  if (commands_)
    commands_->disconnected();
  device_.reset();
  claim_.release();
}

void ObservationCache::received(const robotiq::GripperFeedback &feedback,
                                int64_t wall, int64_t steady, int64_t started) {
  if (!feedback.valid)
    throw std::invalid_argument("Expected validated library feedback");
  std::lock_guard<std::mutex> lock(mutex_);
  latest_.error.clear();
  latest_.last_attempt_valid = true;
  ++latest_.sequence;
  latest_.last_success_unix_ns = wall;
  latest_.last_success_monotonic_ns = steady;
  latest_.request_started_unix_ns = started;
  feedback_ = feedback;
}

void ObservationCache::failed(const std::string &error) {
  std::lock_guard<std::mutex> lock(mutex_);
  latest_.last_attempt_valid = false;
  latest_.error = error;
}

Snapshot ObservationCache::snapshot(int64_t now, int64_t max_age) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto result = latest_;
  if (feedback_)
    result.age_s = (now - result.last_success_monotonic_ns) / 1e9;
  if (!result.error.empty())
    result.state = "failed";
  else if (!feedback_)
    result.state = "missing";
  else if (now < result.last_success_monotonic_ns ||
           now - result.last_success_monotonic_ns > max_age)
    result.state = "stale";
  else {
    result.state = "fresh";
    result.feedback = feedback_;
  }
  return result;
}

int64_t monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
int64_t unix_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
} // namespace rlinf_robotiq_observation
