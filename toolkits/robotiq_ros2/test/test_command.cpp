// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <poll.h>
#include <pty.h>
#include <termios.h>
#include <unistd.h>

#include <filesystem>
#include <future>
#include <limits>
#include <map>

#include "rlinf_robotiq_observation/observation.hpp"
#include "src/helpers.h"
#ifdef ROBOTIQ_TEST_ROS
#include "rlinf_robotiq_observation/message.hpp"
#include "rlinf_robotiq_observation/srv/authorize_commands.hpp"
#include "rlinf_robotiq_observation/srv/set_normalized_position.hpp"
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#endif

namespace obs = rlinf_robotiq_observation;
constexpr int64_t steady = 1000000000, wall = 10000000000;

obs::CommandPolicy enabled_policy() {
  return {true, "operator", 1, 1, 1, 500000000};
}
robotiq::GripperFeedback healthy() {
  robotiq::GripperFeedback f{};
  f.valid = true;
  f.raw_position = f.raw_commanded_position = 128;
  f.position = f.commanded_position = 128.0 / 255;
  f.status = {robotiq::ACTIVATED, robotiq::GOTO_POSITION,
              robotiq::ACTIVATION_COMPLETE, robotiq::AT_REQUESTED_POSITION,
              robotiq::NONE};
  return f;
}
std::string prepare(obs::CommandGate &gate, int64_t now = steady) {
  gate.connected();
  gate.observed(healthy(), now);
  auto lease = gate.authorize("operator", 1000, now);
  EXPECT_TRUE(lease.success) << lease.reason;
  EXPECT_EQ(lease.id.size(), 32u);
  return lease.id;
}
obs::PositionRequest request(const std::string &id) {
  return {"operator", id, 1, wall, 500, 0.5, 0.1, 0.1};
}

TEST(CommandGate,
     RequiresEnableHealthyConnectionAndExplicitExclusiveAuthorization) {
  obs::CommandGate disabled({});
  disabled.connected();
  disabled.observed(healthy(), steady);
  EXPECT_FALSE(disabled.authorize("operator", 1000, steady).success);
  obs::CommandGate gate(enabled_policy());
  EXPECT_FALSE(gate.authorize("operator", 1000, steady).success);
  gate.connected();
  EXPECT_FALSE(gate.authorize("operator", 1000, steady).success);
  gate.observed(healthy(), steady);
  EXPECT_FALSE(gate.authorize("other", 1000, steady).success);
  EXPECT_FALSE(gate.authorize("operator", 0, steady).success);
  EXPECT_FALSE(gate.authorize("operator", 1001, steady).success);
  auto lease = gate.authorize("operator", 1000, steady);
  ASSERT_TRUE(lease.success);
  EXPECT_FALSE(gate.authorize("operator", 1000, steady).success);
  EXPECT_FALSE(gate.revoke("other", lease.id).success);
  EXPECT_FALSE(gate.revoke("operator", "forged").success);
  EXPECT_TRUE(gate.revoke("operator", lease.id).success);
  EXPECT_FALSE(gate.submit(request(lease.id), wall, steady).accepted);
  EXPECT_THROW(obs::CommandGate((obs::CommandPolicy{true, ""})),
               std::invalid_argument);
}

TEST(CommandGate,
     RejectsRangesNonfiniteTimestampsLimitsAndForeignOwnerWithoutWrites) {
  obs::CommandPolicy policy{true, "operator"};
  obs::CommandGate gate(policy);
  auto id = prepare(gate);
  const auto base = request(id);
  for (double invalid : {-0.01, 1.01, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    auto r = base;
    r.position = invalid;
    EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
    r = base;
    r.speed = invalid;
    EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
    r = base;
    r.force = invalid;
    EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  }
  auto r = base;
  r.speed = 0.101;
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  r = base;
  r.force = 0.101;
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  r = base;
  r.position = 0.9;
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  r = base;
  r.position = healthy().position - 0.1;
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  r = base;
  r.owner = "other";
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  r = base;
  r.authorization_id = "forged";
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  r = base;
  r.issued_unix_ns = wall + 1;
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  r = base;
  r.issued_unix_ns = 0;
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  r = base;
  r.valid_for_ms = 0;
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  r = base;
  r.valid_for_ms = 501;
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  EXPECT_FALSE(gate.submit(base, wall + 500000000, steady).accepted);
  int writes = 0;
  EXPECT_FALSE(gate.dispatch(
      [&](auto &) {
        ++writes;
        return true;
      },
      wall, steady));
  EXPECT_EQ(writes, 0);
  EXPECT_TRUE(gate.submit(base, wall, steady).accepted);
}

TEST(CommandGate, OnePendingRequestNoReplayAndNoNewRequestBeforeEcho) {
  obs::CommandGate gate(enabled_policy());
  auto id = prepare(gate);
  auto r = request(id);
  EXPECT_TRUE(gate.submit(r, wall, steady).accepted);
  r.sequence = 2;
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  int writes = 0;
  auto result = gate.dispatch(
      [&](auto &) {
        ++writes;
        return true;
      },
      wall, steady);
  ASSERT_TRUE(result);
  EXPECT_EQ(result->stage, "acknowledged");
  EXPECT_EQ(writes, 1);
  EXPECT_FALSE(gate.dispatch(
      [&](auto &) {
        ++writes;
        return true;
      },
      wall, steady));
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  auto f = healthy();
  f.raw_commanded_position = 127;
  gate.observed(f, steady);
  r.sequence = 1;
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  r.sequence = 0;
  EXPECT_FALSE(gate.submit(r, wall, steady).accepted);
  r.sequence = 2;
  EXPECT_TRUE(gate.submit(r, wall, steady).accepted);
}

TEST(CommandGate, RevocationCannotBypassOutstandingRequestEcho) {
  obs::CommandGate gate(enabled_policy());
  const auto id = prepare(gate);
  ASSERT_TRUE(gate.submit(request(id), wall, steady).accepted);
  ASSERT_TRUE(gate.dispatch([](auto &) { return true; }, wall, steady));
  EXPECT_TRUE(gate.revoke("operator", id).success);
  EXPECT_FALSE(gate.authorize("operator", 1000, steady).success);
  auto f = healthy();
  f.raw_commanded_position = 127;
  gate.observed(f, steady);
  EXPECT_TRUE(gate.authorize("operator", 1000, steady).success);
}

TEST(CommandGate, RevalidatesLeaseQueueWallAndMonotonicFreshnessAtDispatch) {
  for (int scenario = 0; scenario < 5; ++scenario) {
    obs::CommandGate gate(enabled_policy());
    auto id = prepare(gate);
    ASSERT_TRUE(gate.submit(request(id), wall, steady).accepted);
    int64_t now_wall = wall, now_steady = steady;
    if (scenario == 0)
      now_wall += 500000000;
    if (scenario == 1)
      now_steady += 500000000;
    if (scenario == 2)
      now_wall -= 1;
    if (scenario == 3)
      now_steady -= 1;
    if (scenario == 4) {
      now_steady += 1000000000;
      gate.observed(healthy(), now_steady);
    }
    int writes = 0;
    auto outcome = gate.dispatch(
        [&](auto &) {
          ++writes;
          return true;
        },
        now_wall, now_steady);
    ASSERT_TRUE(outcome);
    EXPECT_EQ(outcome->stage, "dropped");
    EXPECT_EQ(writes, 0);
  }
  obs::CommandGate gate(enabled_policy());
  prepare(gate);
  EXPECT_FALSE(gate.authorize("operator", 1000, steady + 500000001).success);
}

TEST(CommandGate, FaultsAndCommunicationFailuresInvalidateQueuedAuthorization) {
  for (int scenario = 0; scenario < 6; ++scenario) {
    obs::CommandGate gate(enabled_policy());
    auto id = prepare(gate);
    ASSERT_TRUE(gate.submit(request(id), wall, steady).accepted);
    auto f = healthy();
    if (scenario == 0)
      f.valid = false;
    if (scenario == 1)
      f.raw_fault = 9;
    if (scenario == 2)
      f.status.gact = robotiq::NOT_ACTIVATED;
    if (scenario == 3)
      f.status.gsta = robotiq::ACTIVATION_IN_PROGRESS;
    if (scenario == 4)
      f.status.gflt = robotiq::COMM_TIMEOUT;
    if (scenario == 5) {
      gate.failed();
    } else
      gate.observed(f, steady);
    int writes = 0;
    EXPECT_FALSE(gate.dispatch(
        [&](auto &) {
          ++writes;
          return true;
        },
        wall, steady));
    gate.observed(healthy(), steady);
    EXPECT_FALSE(gate.submit(request(id), wall, steady).accepted);
    EXPECT_EQ(writes, 0);
  }
  obs::CommandGate moving(enabled_policy());
  auto id = prepare(moving);
  auto f = healthy();
  f.status.gobj = robotiq::IN_MOTION;
  moving.observed(f, steady);
  EXPECT_FALSE(moving.submit(request(id), wall, steady).accepted);
}

TEST(CommandGate, ReconnectRevocationAndShutdownNeverReplayQueuedCommand) {
  for (int scenario = 0; scenario < 3; ++scenario) {
    obs::CommandGate gate(enabled_policy());
    auto id = prepare(gate);
    ASSERT_TRUE(gate.submit(request(id), wall, steady).accepted);
    if (scenario == 0) {
      gate.disconnected();
      gate.connected();
    }
    if (scenario == 1)
      EXPECT_TRUE(gate.revoke("operator", id).success);
    if (scenario == 2)
      gate.shutdown();
    gate.observed(healthy(), steady);
    int writes = 0;
    EXPECT_FALSE(gate.dispatch(
        [&](auto &) {
          ++writes;
          return true;
        },
        wall, steady));
    EXPECT_FALSE(gate.submit(request(id), wall, steady).accepted);
    EXPECT_EQ(writes, 0);
    if (scenario == 2) {
      gate.connected();
      EXPECT_FALSE(gate.authorize("operator", 1000, steady).success);
    } else {
      auto next = gate.authorize("operator", 1000, steady);
      EXPECT_TRUE(next.success);
      EXPECT_NE(id, next.id);
    }
  }
}

class CommandSerial : public ::testing::Test {
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
    char root[] = "/tmp/robotiq-command-XXXXXX";
    const auto created = ::mkdtemp(root);
    ASSERT_NE(created, nullptr);
    proc_ = created;
    gate_ = std::make_shared<obs::CommandGate>(enabled_policy());
    device_ = std::make_unique<obs::SerialObserver>(port_, 60, proc_, gate_);
    device_->connect();
  }
  void TearDown() override {
    device_.reset();
    ::close(master_);
    std::filesystem::remove_all(proc_);
  }
  std::string take(size_t size) {
    std::string result;
    for (size_t i = 0; i < size; ++i) {
      pollfd fd{master_, POLLIN, 0};
      if (::poll(&fd, 1, 1000) != 1)
        throw std::runtime_error("Simulated request missing");
      char c;
      if (::read(master_, &c, 1) != 1)
        throw std::runtime_error("PTY read failed");
      result += c;
    }
    return result;
  }
  void send(const std::string &bytes) {
    ASSERT_EQ(::write(master_, bytes.data(), bytes.size()),
              static_cast<ssize_t>(bytes.size()));
  }
  std::string packet(const std::string &body) {
    return robotiq::hex_to_bin(body + robotiq::crc16_modbus(body));
  }
  void quiet() {
    pollfd fd{master_, POLLIN, 0};
    EXPECT_EQ(::poll(&fd, 1, 10), 0);
  }
  robotiq::GripperFeedback feedback(std::string echo = "80",
                                    std::string fault = "00") {
    auto future =
        std::async(std::launch::async, [&] { return device_->read(); });
    EXPECT_EQ(robotiq::bin_to_hex(take(8)), "090307D00003040E");
    send(packet("090306F900" + fault + echo + "8000"));
    return future.get();
  }
  std::string grant() {
    feedback();
    auto a = gate_->authorize("operator", 1000, obs::monotonic_ns());
    EXPECT_TRUE(a.success);
    return a.id;
  }
  obs::PositionRequest real_request(const std::string &id) {
    auto r = request(id);
    r.issued_unix_ns = obs::unix_ns();
    return r;
  }
  int master_{-1};
  std::string port_, proc_;
  std::shared_ptr<obs::CommandGate> gate_;
  std::unique_ptr<obs::SerialObserver> device_;
};

TEST_F(CommandSerial, ExactVendorEncodingBoundedAckAndIndependentFeedback) {
  quiet();
  auto id = grant();
  auto r = real_request(id);
  r.position = 0;
  r.speed = r.force = 1;
  ASSERT_TRUE(gate_->submit(r, obs::unix_ns(), obs::monotonic_ns()).accepted);
  auto future = std::async(std::launch::async,
                           [&] { return device_->execute_pending(); });
  const auto frame = take(15);
  EXPECT_EQ(robotiq::bin_to_hex(frame), "091003E800030609000000FFFF7219");
  send(frame + robotiq::hex_to_bin("091003E800030130"));
  auto result = future.get();
  ASSERT_TRUE(result);
  EXPECT_EQ(result->stage, "acknowledged");
  EXPECT_NE(result->reason.find("unknown"), std::string::npos);
  quiet();
  auto f = feedback("00");
  EXPECT_EQ(f.raw_commanded_position, 0);
  EXPECT_EQ(f.raw_position, 128);
  EXPECT_NE(f.position, f.commanded_position);
  EXPECT_FALSE(device_->execute_pending());
  quiet();
}

TEST_F(CommandSerial,
       FractionsEncodeIndependentSpeedForceAndDoNotWaitForMotion) {
  auto id = grant();
  auto r = real_request(id);
  r.speed = 0.1;
  r.force = 0.25;
  ASSERT_TRUE(gate_->submit(r, obs::unix_ns(), obs::monotonic_ns()).accepted);
  auto future = std::async(std::launch::async,
                           [&] { return device_->execute_pending(); });
  const auto frame = take(15);
  EXPECT_EQ(frame, packet("091003E80003060900007F193F"));
  send(robotiq::hex_to_bin("091003E800030130"));
  ASSERT_TRUE(future.get());
  quiet();
}

TEST_F(CommandSerial, AckFailureNoRetryRevokesAndNeverSurvivesReconnect) {
  for (const auto &bad :
       {robotiq::hex_to_bin("091003E800030000"), packet("091003E90003"),
        packet("099002"), std::string{}}) {
    auto id = grant();
    auto r = real_request(id);
    ASSERT_TRUE(gate_->submit(r, obs::unix_ns(), obs::monotonic_ns()).accepted);
    auto future = std::async(std::launch::async,
                             [&] { return device_->execute_pending(); });
    take(15);
    if (!bad.empty())
      send(bad);
    EXPECT_THROW(future.get(), std::runtime_error);
    EXPECT_EQ(gate_->result().stage, "failed");
    quiet();
    device_->disconnect();
    device_->connect();
    quiet();
    feedback();
    EXPECT_FALSE(
        gate_->submit(real_request(id), obs::unix_ns(), obs::monotonic_ns())
            .accepted);
    EXPECT_FALSE(device_->execute_pending());
    quiet();
  }
}

TEST_F(CommandSerial, FaultBeforeDispatchDropsRequestAndLifecycleSendsNothing) {
  auto id = grant();
  auto r = real_request(id);
  ASSERT_TRUE(gate_->submit(r, obs::unix_ns(), obs::monotonic_ns()).accepted);
  auto f = feedback("80", "09");
  EXPECT_EQ(f.status.gflt, robotiq::COMM_TIMEOUT);
  EXPECT_FALSE(device_->execute_pending());
  quiet();
  EXPECT_FALSE(gate_->authorize("operator", 1000, obs::monotonic_ns()).success);
  device_->disconnect();
  device_->disconnect();
  device_->connect();
  device_->connect();
  quiet();
  EXPECT_FALSE(device_->execute_pending());
}

#ifdef ROBOTIQ_TEST_ROS
TEST(CommandInterfaces, OfflineServiceSerializationAndOutcomeSeparation) {
  using Request = obs::srv::SetNormalizedPosition::Request;
  Request input;
  input.owner = "operator";
  input.authorization_id = "opaque-lease";
  input.sequence = 42;
  input.issued_at.sec = 123;
  input.issued_at.nanosec = 456;
  input.valid_for_ms = 200;
  input.position = 0.5;
  input.speed = 0.1;
  input.force = 0;
  rclcpp::Serialization<Request> serializer;
  rclcpp::SerializedMessage bytes;
  serializer.serialize_message(&input, &bytes);
  Request output;
  serializer.deserialize_message(&bytes, &output);
  EXPECT_EQ(input, output);
  using Grant = obs::srv::AuthorizeCommands::Request;
  Grant grant;
  grant.enable = true;
  grant.owner = "operator";
  grant.lease_ms = 1000;
  rclcpp::Serialization<Grant> grants;
  rclcpp::SerializedMessage encoded;
  grants.serialize_message(&grant, &encoded);
  Grant restored;
  grants.deserialize_message(&encoded, &restored);
  EXPECT_EQ(grant, restored);
  auto message =
      obs::command_message({true, 42, "acknowledged", "Register write only"},
                           "operator", 1000000000);
  std::map<std::string, std::string> values;
  for (const auto &kv : message.status.at(0).values)
    values[kv.key] = kv.value;
  EXPECT_EQ(values["grasp_outcome"], "unknown");
  EXPECT_EQ(values["sequence"], "42");
  EXPECT_EQ(values.count("raw_position"), 0u);
  EXPECT_EQ(values.count("authorization_id"), 0u);
}
#endif
