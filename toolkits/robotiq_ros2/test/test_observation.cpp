// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
#include <fcntl.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <pty.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <future>
#include <map>
#include <thread>

#include "rlinf_robotiq_observation/observation.hpp"
#include "src/helpers.h"
#ifdef ROBOTIQ_TEST_ROS
#include "rlinf_robotiq_observation/message.hpp"
#endif

namespace obs = rlinf_robotiq_observation;
using namespace std::chrono_literals;

// All serial I/O in this suite is confined to a freshly allocated PTY pair.
class ObservationTest : public ::testing::Test {
protected:
  void SetUp() override {
    int slave;
    char name[128];
    ASSERT_EQ(::openpty(&master_, &slave, name, nullptr, nullptr), 0);
    port_ = name;
    termios mode{};
    ASSERT_EQ(::tcgetattr(slave, &mode), 0);
    ::cfmakeraw(&mode);
    ASSERT_EQ(::tcsetattr(slave, TCSANOW, &mode), 0);
    ::close(slave);
    char temporary[] = "/tmp/robotiq-observation-test-XXXXXX";
    auto root = ::mkdtemp(temporary);
    ASSERT_NE(root, nullptr);
    proc_ = root;
  }
  void TearDown() override {
    ::close(master_);
    std::filesystem::remove_all(proc_);
  }
  std::string request() {
    std::string result;
    for (int i = 0; i < 8; ++i) {
      pollfd event{master_, POLLIN, 0};
      if (::poll(&event, 1, 1000) != 1)
        throw std::runtime_error("Missing simulated request");
      char byte;
      if (::read(master_, &byte, 1) != 1)
        throw std::runtime_error("PTY read failed");
      result.push_back(byte);
    }
    EXPECT_EQ(robotiq::bin_to_hex(result), "090307D00003040E");
    return result;
  }
  std::string response(std::string body = "090306F900A912E107") {
    return robotiq::hex_to_bin(body + robotiq::crc16_modbus(body));
  }
  void send(const std::string &bytes) {
    ASSERT_EQ(::write(master_, bytes.data(), bytes.size()),
              static_cast<ssize_t>(bytes.size()));
  }
  void expect_no_request() {
    pollfd event{master_, POLLIN, 0};
    EXPECT_EQ(::poll(&event, 1, 10), 0);
  }
  int master_{-1};
  std::string port_, proc_;
};

TEST_F(ObservationTest, MapsMeasuredAndRequestedFeedbackWithEchoAndFault) {
  obs::SerialObserver device(port_, 100, proc_);
  device.connect();
  device.connect();
  expect_no_request();
  auto read = std::async(std::launch::async, [&] { return device.read(); });
  const auto echo = request();
  send(echo + response());
  const auto f = read.get();
  EXPECT_TRUE(f.valid);
  EXPECT_EQ(f.raw_commanded_position, 18);
  EXPECT_EQ(f.raw_position, 225);
  EXPECT_DOUBLE_EQ(f.commanded_position, 18.0 / 255);
  EXPECT_DOUBLE_EQ(f.position, 225.0 / 255);
  EXPECT_EQ(f.raw_current, 7);
  EXPECT_DOUBLE_EQ(f.current, 7.0 / 255);
  EXPECT_EQ(f.raw_fault, 0xA9);
  EXPECT_EQ(f.status.gflt, robotiq::COMM_TIMEOUT);
  EXPECT_EQ(f.status.gact, 1);
  EXPECT_EQ(f.status.ggto, 1);
  EXPECT_EQ(static_cast<int>(f.status.gsta), 3);
  EXPECT_EQ(f.status.gobj, 3);
  device.disconnect();
  device.disconnect();
}

TEST_F(ObservationTest, RejectsCrcSlaveFunctionCountAndExceptionResponses) {
  const std::string wrong_crc = robotiq::hex_to_bin("090306F9000012E1070000");
  for (const auto &bytes :
       {wrong_crc, response("080306F9000012E107"),
        response("090406F9000012E107"), response("090302F900"),
        response("098302"), std::string{}}) {
    obs::SerialObserver device(port_, 40, proc_);
    device.connect();
    auto read = std::async(std::launch::async, [&] { return device.read(); });
    request();
    if (!bytes.empty())
      send(bytes);
    EXPECT_THROW(read.get(), std::runtime_error);
  }
}

TEST_F(ObservationTest, ResynchronizesAfterNoiseAndCorruptedFrame) {
  obs::SerialObserver device(port_, 100, proc_);
  device.connect();
  auto read = std::async(std::launch::async, [&] { return device.read(); });
  request();
  send(std::string("garbage") + robotiq::hex_to_bin("090306F9000012E1070000") +
       response());
  EXPECT_TRUE(read.get().valid);
}

TEST_F(ObservationTest, DeadlineBoundsSlowPartialReplyAndAllowsReconnect) {
  obs::SerialObserver device(port_, 50, proc_);
  device.connect();
  auto start = std::chrono::steady_clock::now();
  auto read = std::async(std::launch::async, [&] { return device.read(); });
  request();
  for (char byte : std::string("\x09\x03\x06", 3)) {
    send(std::string(1, byte));
    std::this_thread::sleep_for(20ms);
  }
  EXPECT_THROW(read.get(), std::runtime_error);
  EXPECT_LT(std::chrono::steady_clock::now() - start, 250ms);
  device.disconnect();
  device.connect();
  expect_no_request();
  auto next = std::async(std::launch::async, [&] { return device.read(); });
  request();
  send(response("090306310000000000"));
  const auto recovered = next.get();
  EXPECT_TRUE(recovered.valid);
  EXPECT_EQ(recovered.raw_fault, 0);
  EXPECT_EQ(recovered.status.gsta, robotiq::ACTIVATION_COMPLETE);
}

TEST_F(ObservationTest,
       ClaimRejectsAliasesExistingOwnersAndMissingProcVisibility) {
  obs::PortClaim first(port_, proc_);
  first.acquire();
  first.acquire();
  const auto alias = proc_ + "/alias";
  std::filesystem::create_symlink(port_, alias);
  obs::PortClaim second(alias, proc_);
  EXPECT_THROW(second.acquire(), std::runtime_error);
  first.release();
  first.release();
  std::filesystem::create_directories(proc_ + "/123/fd");
  std::filesystem::create_symlink(port_, proc_ + "/123/fd/4");
  EXPECT_THROW(second.acquire(), std::runtime_error);
  std::filesystem::remove(proc_ + "/123/fd/4");
  second.acquire();
  second.release();
  obs::PortClaim invisible(port_, proc_ + "/absent");
  EXPECT_THROW(invisible.acquire(), std::filesystem::filesystem_error);
}

TEST_F(ObservationTest,
       DestructionAndFailedConnectReleaseClaimsWithoutCommands) {
  {
    obs::SerialObserver device(port_, 50, proc_);
    device.connect();
    expect_no_request();
  }
  obs::SerialObserver next(port_, 50, proc_);
  next.connect();
  expect_no_request();
  next.disconnect();
  obs::SerialObserver absent(proc_ + "/missing", 50, proc_);
  EXPECT_THROW(absent.connect(), std::runtime_error);
  EXPECT_THROW(absent.connect(), std::runtime_error);
  absent.disconnect();
  EXPECT_THROW(absent.read(), std::runtime_error);
}

TEST_F(ObservationTest, KernelExclusiveAccessAndFailedOpenReleaseClaim) {
  const int other = ::open(port_.c_str(), O_RDWR | O_NOCTTY);
  ASSERT_GE(other, 0);
  ASSERT_EQ(::ioctl(other, TIOCEXCL), 0);
  obs::SerialObserver device(port_, 50, proc_);
  EXPECT_THROW(device.connect(), std::runtime_error);
  EXPECT_EQ(::ioctl(other, TIOCNXCL), 0);
  ::close(other);
  device.connect();
  const int conflicting = ::open(port_.c_str(), O_RDWR | O_NOCTTY);
  EXPECT_EQ(conflicting, -1);
  if (conflicting >= 0)
    ::close(conflicting);
  expect_no_request();
  device.disconnect();
  const int reopened = ::open(port_.c_str(), O_RDWR | O_NOCTTY);
  EXPECT_GE(reopened, 0);
  if (reopened >= 0)
    ::close(reopened);
}

TEST_F(ObservationTest,
       PermissionDeniedOwnerInspectionReleasesClaimBeforeOpen) {
  if (::geteuid() == 0)
    GTEST_SKIP()
        << "Permission-denial qualification requires an ordinary account";
  const auto restricted = std::filesystem::path(proc_) / "123/fd";
  std::filesystem::create_directories(restricted);
  struct RestorePermissions {
    std::filesystem::path path;
    ~RestorePermissions() {
      std::error_code ignored;
      std::filesystem::permissions(path, std::filesystem::perms::owner_all,
                                   ignored);
    }
  } restore{restricted};
  std::filesystem::permissions(restricted, std::filesystem::perms::none);
  obs::SerialObserver device(port_, 50, proc_);
  try {
    device.connect();
    FAIL() << "Expected inaccessible ownership view to reject connection";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("Cannot inspect serial owners"),
              std::string::npos);
  }
  pollfd event{master_, POLLIN, 0};
  ASSERT_GE(::poll(&event, 1, 10), 0);
  EXPECT_EQ(event.revents & POLLIN, 0);
  std::filesystem::permissions(restricted, std::filesystem::perms::owner_all);
  obs::PortClaim replacement(port_, proc_);
  EXPECT_NO_THROW(replacement.acquire());
  replacement.release();
  device.disconnect();
  device.connect();
  device.connect();
  expect_no_request();
  device.disconnect();
  device.disconnect();
}

TEST_F(ObservationTest, KernelExclusiveModeDoesNotRevokePreexistingDescriptor) {
  const int existing = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_CLOEXEC);
  ASSERT_GE(existing, 0);
  struct CloseDescriptor {
    int fd;
    ~CloseDescriptor() { ::close(fd); }
  } holder{existing};
  // Deliberately omit the holder from the simulated process view.
  obs::SerialObserver device(port_, 50, proc_);
  device.connect();
  expect_no_request();
  int exclusive = 0;
  ASSERT_EQ(::ioctl(existing, TIOCGEXCL, &exclusive), 0);
  EXPECT_NE(exclusive, 0);
  ASSERT_EQ(::write(existing, "owner", 5), 5);
  pollfd event{master_, POLLIN, 0};
  ASSERT_EQ(::poll(&event, 1, 100), 1);
  char received[5];
  ASSERT_EQ(::read(master_, received, sizeof(received)), 5);
  EXPECT_EQ(std::string(received, sizeof(received)), "owner");
  device.disconnect();
  EXPECT_GE(::fcntl(existing, F_GETFD), 0);
}

TEST(ObservationCache, MissingFreshStaleFailedAndRecoveryKeepExplicitClocks) {
  obs::ObservationCache cache;
  EXPECT_EQ(cache.snapshot(1000, 100).state, "missing");
  robotiq::GripperFeedback f{};
  EXPECT_THROW(cache.received(f, 10000, 1000, 9990), std::invalid_argument);
  f.valid = true;
  cache.received(f, 10000, 1000, 9990);
  auto sample = cache.snapshot(1100, 100);
  EXPECT_EQ(sample.state, "fresh");
  EXPECT_TRUE(sample.feedback.has_value());
  EXPECT_EQ(sample.last_success_unix_ns, 10000);
  EXPECT_EQ(sample.request_started_unix_ns, 9990);
  EXPECT_EQ(sample.sequence, 1u);
  EXPECT_EQ(cache.snapshot(1101, 100).state, "stale");
  EXPECT_FALSE(cache.snapshot(1101, 100).feedback);
  EXPECT_EQ(cache.snapshot(999, 100).state, "stale");
  cache.failed("timeout");
  sample = cache.snapshot(1100, 100);
  EXPECT_EQ(sample.state, "failed");
  EXPECT_FALSE(sample.last_attempt_valid);
  EXPECT_FALSE(sample.feedback);
  EXPECT_EQ(sample.last_success_unix_ns, 10000);
  cache.received(f, 20000, 2000, 19990);
  sample = cache.snapshot(2001, 100);
  EXPECT_EQ(sample.state, "fresh");
  EXPECT_TRUE(sample.last_attempt_valid);
  EXPECT_TRUE(sample.error.empty());
  EXPECT_EQ(sample.sequence, 2u);
}

#ifdef ROBOTIQ_TEST_ROS
TEST(ObservationMessage, PreservesSchemaAndSuppressesUnavailableMeasurements) {
  obs::ObservationCache cache;
  auto message =
      obs::observation_message(cache.snapshot(1000, 100), "port", 12345678901);
  EXPECT_EQ(message.header.stamp.sec, 12);
  EXPECT_EQ(message.header.stamp.nanosec, 345678901u);
  EXPECT_EQ(message.status.at(0).level,
            diagnostic_msgs::msg::DiagnosticStatus::STALE);
  auto values = [](const auto &message) {
    std::map<std::string, std::string> result;
    for (const auto &kv : message.status.at(0).values)
      result[kv.key] = kv.value;
    return result;
  };
  EXPECT_EQ(values(message)["communication_valid"], "false");
  EXPECT_EQ(values(message).count("position"), 0u);
  robotiq::GripperFeedback feedback{};
  feedback.valid = true;
  feedback.raw_fault = 9;
  feedback.raw_position = 225;
  feedback.raw_commanded_position = 18;
  feedback.raw_current = 7;
  cache.received(feedback, 10000, 1000, 9990);
  message =
      obs::observation_message(cache.snapshot(1001, 100), "port", 12345678901);
  auto data = values(message);
  EXPECT_EQ(message.status.at(0).level,
            diagnostic_msgs::msg::DiagnosticStatus::WARN);
  EXPECT_EQ(data["communication_valid"], "true");
  EXPECT_EQ(data["fault_status"], "COMM_TIMEOUT");
  EXPECT_EQ(data["raw_position"], "225");
  EXPECT_EQ(data["raw_commanded_position"], "18");
  EXPECT_NEAR(std::stod(data["current_ampere"]), 0.07, 1e-12);
  EXPECT_EQ(data["metric_width_available"], "false");
  cache.failed("crc or timeout");
  message =
      obs::observation_message(cache.snapshot(1001, 100), "port", 12345678901);
  EXPECT_EQ(message.status.at(0).level,
            diagnostic_msgs::msg::DiagnosticStatus::ERROR);
  EXPECT_EQ(values(message).count("raw_position"), 0u);
  EXPECT_EQ(values(message)["last_success_unix_ns"], "10000");
}
#endif
