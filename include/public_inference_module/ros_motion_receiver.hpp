#pragma once

#include "public_inference_module/config.hpp"
#include "public_inference_module/motion_data.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include <atomic>
#include <memory>
#include <thread>

namespace public_inference_module {

class RosMotionReceiver {
  public:
    explicit RosMotionReceiver(std::shared_ptr<MotionDataBuffer> buffer);
    ~RosMotionReceiver();

    bool start(const RosMotionConfig &config);
    void stop();
    std::shared_ptr<const MotionDataSample> readLatest() const;
    bool useRecordedReference() const;

  private:
    void modeCallback(const std_msgs::msg::Bool &msg);
    void callback(const std_msgs::msg::String &msg);
    static uint64_t nowSteadyNs();

  private:
    std::shared_ptr<MotionDataBuffer> buffer_;
    RosMotionConfig config_;
    std::atomic<uint64_t> seq_{0};

    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mode_subscription_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
    std::thread spin_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> use_recorded_reference_{false};
};

}  // namespace public_inference_module
