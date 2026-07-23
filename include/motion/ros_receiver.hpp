#pragma once

#include "core/config.hpp"
#include "core/types.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include <atomic>
#include <memory>
#include <thread>

namespace igris_c_gmt_public {

class MotionFrameBuffer;

class RosMotionReceiver {
public:
  explicit RosMotionReceiver(std::shared_ptr<MotionFrameBuffer> buffer);
  ~RosMotionReceiver();

  bool start(const RosMotionConfig &config);
  void stop();
  std::shared_ptr<const MotionFrame> readLatest() const;
  bool useRecordedReference() const;

private:
  void modeCallback(const std_msgs::msg::Bool &msg);
  void callback(const std_msgs::msg::String &msg);

private:
  std::shared_ptr<MotionFrameBuffer> buffer_;
  RosMotionConfig config_;
  std::atomic<uint64_t> seq_{0};

  rclcpp::Context::SharedPtr context_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mode_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spin_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> use_recorded_reference_{false};
};

} // namespace igris_c_gmt_public
