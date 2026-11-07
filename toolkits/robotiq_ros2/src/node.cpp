// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <thread>

#include "rlinf_robotiq_observation/message.hpp"
#include "rlinf_robotiq_observation/srv/authorize_commands.hpp"
#include "rlinf_robotiq_observation/srv/set_normalized_position.hpp"

namespace rlinf_robotiq_observation {
class ObservationNode : public rclcpp::Node {
public:
  ObservationNode() : Node("robotiq_observation") {
    rcl_interfaces::msg::ParameterDescriptor fixed;
    fixed.read_only = true;
    enabled_ = declare_parameter("enabled", false, fixed);
    port_ = declare_parameter(
        "serial_port",
        std::string(
            "/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DA95C62Y-if00-port0"),
        fixed);
    const auto timeout = declare_parameter("serial_timeout_ms", 100, fixed);
    const auto poll_hz = declare_parameter("poll_hz", 5.0, fixed);
    const auto max_age = declare_parameter("max_age_s", 0.5, fixed);
    const auto reconnect = declare_parameter("reconnect_s", 1.0, fixed);
    if (timeout < 1 || timeout > 1000 || !std::isfinite(poll_hz) ||
        poll_hz <= 0 || poll_hz > 50 || !std::isfinite(max_age) ||
        max_age <= 0 || max_age > 60 || !std::isfinite(reconnect) ||
        reconnect < 0.1 || reconnect > 60)
      throw std::invalid_argument(
          "Expected timeout [1,1000] ms, poll (0,50] Hz, "
          "age (0,60] s and reconnect [0.1,60] s");
    max_age_ns_ = static_cast<int64_t>(max_age * 1e9);
    CommandPolicy policy;
    policy.enabled = declare_parameter("commands_enabled", false, fixed);
    policy.owner = declare_parameter("command_owner", std::string{}, fixed);
    policy.max_speed = declare_parameter("command_max_speed", 0.1, fixed);
    policy.max_force = declare_parameter("command_max_force", 0.1, fixed);
    policy.max_step = declare_parameter("command_max_step", 0.1, fixed);
    policy.feedback_max_age_ns = std::min<int64_t>(max_age_ns_, 500000000);
    if (policy.enabled && !enabled_)
      throw std::invalid_argument(
          "Command enable requires explicit serial enable");
    commands_ = std::make_shared<CommandGate>(policy);
    rclcpp::on_shutdown([gate = std::weak_ptr<CommandGate>(commands_)] {
      if (auto locked = gate.lock())
        locked->shutdown();
    });
    if (policy.enabled) {
      command_publisher_ =
          create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
              "~/command_status",
              rclcpp::QoS(1).reliable().durability_volatile());
      authorize_service_ = create_service<srv::AuthorizeCommands>(
          "~/authorize_commands",
          [this](const std::shared_ptr<srv::AuthorizeCommands::Request> request,
                 std::shared_ptr<srv::AuthorizeCommands::Response> response) {
            Authorization authorization;
            if (request->enable && !request->authorization_id.empty())
              authorization = {
                  false, {}, "Grant requires an empty authorization_id"};
            else if (request->enable)
              authorization = commands_->authorize(
                  request->owner, request->lease_ms, monotonic_ns());
            else
              authorization =
                  commands_->revoke(request->owner, request->authorization_id);
            response->success = authorization.success;
            response->authorization_id = authorization.id;
            response->reason = authorization.reason;
          });
      position_service_ = create_service<srv::SetNormalizedPosition>(
          "~/set_normalized_position",
          [this](
              const std::shared_ptr<srv::SetNormalizedPosition::Request>
                  request,
              std::shared_ptr<srv::SetNormalizedPosition::Response> response) {
            const auto stamp = request->issued_at;
            PositionRequest position{
                request->owner,
                request->authorization_id,
                request->sequence,
                stamp.sec >= 0 && stamp.nanosec < 1000000000
                    ? static_cast<int64_t>(stamp.sec) * 1000000000 +
                          stamp.nanosec
                    : 0,
                request->valid_for_ms,
                request->position,
                request->speed,
                request->force};
            const auto result =
                commands_->submit(position, unix_ns(), monotonic_ns());
            response->accepted = result.accepted;
            response->sequence = result.sequence;
            response->stage = result.stage;
            response->reason = result.reason;
          });
      command_owner_ = policy.owner;
    }
    publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "~/state", rclcpp::QoS(1).reliable().durability_volatile());
    timer_ = create_wall_timer(std::chrono::milliseconds(100), [this] {
      publisher_->publish(observation_message(
          cache_.snapshot(monotonic_ns(), max_age_ns_), port_, unix_ns()));
      if (command_publisher_)
        command_publisher_->publish(
            command_message(commands_->result(), command_owner_, unix_ns()));
    });
    if (enabled_) {
      worker_ = std::thread([this, timeout, poll_hz, reconnect] {
        SerialObserver observer(port_, static_cast<unsigned>(timeout), "/proc",
                                commands_);
        while (!stopping()) {
          double delay = 1.0 / poll_hz;
          try {
            observer.connect();
            const auto started = unix_ns();
            const auto feedback = observer.read();
            cache_.received(feedback, unix_ns(), monotonic_ns(), started);
            observer.execute_pending();
          } catch (const std::exception &error) {
            cache_.failed(error.what());
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s",
                                 error.what());
            observer.disconnect();
            delay = reconnect;
          }
          std::unique_lock<std::mutex> lock(stop_mutex_);
          stop_condition_.wait_for(lock, std::chrono::duration<double>(delay),
                                   [this] { return stop_; });
        }
      });
    } else {
      RCLCPP_INFO(get_logger(),
                  "Serial observation disabled; publishing missing state");
    }
  }

  ~ObservationNode() override {
    commands_->shutdown();
    {
      std::lock_guard<std::mutex> lock(stop_mutex_);
      stop_ = true;
    }
    stop_condition_.notify_all();
    if (worker_.joinable())
      worker_.join();
  }

private:
  bool stopping() {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    return stop_;
  }
  bool enabled_{false}, stop_{false};
  std::string port_;
  int64_t max_age_ns_{0};
  std::mutex stop_mutex_;
  std::condition_variable stop_condition_;
  std::thread worker_;
  ObservationCache cache_;
  std::shared_ptr<CommandGate> commands_;
  std::string command_owner_;
  rclcpp::Service<srv::AuthorizeCommands>::SharedPtr authorize_service_;
  rclcpp::Service<srv::SetNormalizedPosition>::SharedPtr position_service_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      command_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};
} // namespace rlinf_robotiq_observation

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(
        std::make_shared<rlinf_robotiq_observation::ObservationNode>());
  } catch (const std::exception &error) {
    RCLCPP_ERROR(rclcpp::get_logger("robotiq_observation"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
