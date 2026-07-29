#include "modules/motion_handler.h"

#include "motion/frame_buffer.hpp"
#include "motion/frame_source.hpp"
#include "motion/ros_receiver.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace igris_c_gmt_public {
namespace {

constexpr double kPi = 3.14159265358979323846;

template <std::size_t N>
void LowPassInPlace(std::array<double, N> &value,
                    std::array<double, N> &state, double sample_hz,
                    double cutoff_hz) {
  if (cutoff_hz <= 0.0 || sample_hz <= 0.0) {
    state = value;
    return;
  }
  const double dt = 1.0 / sample_hz;
  const double rc = 1.0 / (2.0 * kPi * cutoff_hz);
  const double alpha = dt / (rc + dt);
  for (std::size_t i = 0; i < N; ++i) {
    state[i] += alpha * (value[i] - state[i]);
    value[i] = state[i];
  }
}

} // namespace

MotionHandler::MotionHandler() = default;

MotionHandler::~MotionHandler() { stop(); }

int MotionHandler::configure(const InferenceConfig &config) {
  try {
    stop();
    config_ = config;
    reset();

    if (config_.motion_source.type == "ros2") {
      if (config_.motion_source.ros_config_path.empty()) {
        throw std::runtime_error("motion_source.ros_config_path must be set "
                                 "when motion_source.type=ros2");
      }

      const RosMotionConfig ros_config =
          LoadRosMotionConfig(config_.motion_source.ros_config_path);
      ros_motion_buffer_ = std::make_shared<MotionFrameBuffer>();
      ros_receiver_ = std::make_unique<RosMotionReceiver>(ros_motion_buffer_);

      std::cerr << "Starting ROS motion receiver\n";
      if (!ros_receiver_->start(ros_config)) {
        throw std::runtime_error("Failed to start ROS 2 motion receiver");
      }
      std::cerr << "ROS motion receiver started\n";

      if (!config_.motion_source.onnx_path.empty()) {
        InferenceConfig recorded_config = config_;
        recorded_config.motion_source.type = "onnx_replay";
        recorded_source_ = CreateMotionFrameSource(recorded_config);
        std::cerr << "Recorded reference source loaded\n";
      }
    } else {
      replay_source_ = CreateMotionFrameSource(config_);
    }
  } catch (const std::exception &exception) {
    std::cerr << "MotionHandler failed to configure: " << exception.what()
              << "\n";
    stop();
    return -1;
  }

  return 0;
}

void MotionHandler::reset() {
  active_mode_ = "waiting";
  motion_step_ = 0;
  live_motion_seen_ = false;
  last_live_motion_seq_ = 0;
  last_live_motion_wall_time_ = std::chrono::steady_clock::now();
  resetReferenceVelocityFilters();

  if (replay_source_) {
    replay_source_->reset();
  }
  if (recorded_source_) {
    recorded_source_->reset();
  }
}

void MotionHandler::stop() {
  if (ros_receiver_) {
    ros_receiver_->stop();
  }
  ros_receiver_.reset();
  ros_motion_buffer_.reset();
  replay_source_.reset();
  recorded_source_.reset();
  reset();
}

bool MotionHandler::read(MotionHandlerOutput &output,
                         std::chrono::steady_clock::time_point now) {
  output = MotionHandlerOutput{};
  output.step = motion_step_;

  bool fresh = false;
  bool mode_changed = false;
  std::string motion_mode = "waiting";

  if (ros_receiver_) {
    if (recorded_source_ && ros_receiver_->useRecordedReference()) {
      if (active_mode_ != "recorded") {
        motion_step_ = 0;
        recorded_source_->reset();
        mode_changed |= transitionTo("recorded");
      }
      fresh = recorded_source_->getFrameStackAtStep(
          static_cast<std::size_t>(motion_step_), output.frames,
          static_cast<std::size_t>(kMotionFrameStackLength));
      motion_mode = fresh ? "recorded" : "waiting";
    } else {
      const std::shared_ptr<const MotionFrame> motion_frame =
          ros_receiver_->readLatest();
      if (motion_frame && motion_frame->valid) {
        if (!live_motion_seen_ || motion_frame->seq != last_live_motion_seq_) {
          live_motion_seen_ = true;
          last_live_motion_seq_ = motion_frame->seq;
          last_live_motion_wall_time_ = now;
        }

        const auto live_age_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_live_motion_wall_time_)
                .count();
        fresh = static_cast<double>(live_age_ms) <=
                config_.motion_source.frame_timeout_ms;
        if (fresh) {
          output.frames.assign(
              static_cast<std::size_t>(kMotionFrameStackLength), *motion_frame);
        }
        motion_mode = fresh ? "live" : "waiting";
      }
    }
  } else if (replay_source_) {
    if (active_mode_ != config_.motion_source.type) {
      motion_step_ = 0;
      replay_source_->reset();
      mode_changed |= transitionTo(config_.motion_source.type);
    }
    fresh = replay_source_->getFrameStackAtStep(
        static_cast<std::size_t>(motion_step_), output.frames,
        static_cast<std::size_t>(kMotionFrameStackLength));
    motion_mode = fresh ? config_.motion_source.type : "waiting";
  }

  if (!fresh) {
    mode_changed |= transitionTo("waiting");
    output.available = false;
    output.mode_changed = mode_changed;
    output.step = motion_step_;
    output.mode = active_mode_;
    return false;
  }

  mode_changed |= transitionTo(motion_mode);
  updateReferenceVelocityFilters(output.frames);
  output.available = true;
  output.mode_changed = mode_changed;
  output.step = motion_step_;
  output.mode = active_mode_;
  return true;
}

void MotionHandler::advance() { ++motion_step_; }

bool MotionHandler::transitionTo(const std::string &mode) {
  if (mode == active_mode_) {
    return false;
  }
  active_mode_ = mode;
  resetReferenceVelocityFilters();
  return true;
}

void MotionHandler::resetReferenceVelocityFilters() {
  reference_velocity_lpf_initialized_ = false;
  reference_joint_velocity_lpf_.fill(0.0);
  reference_anchor_linear_velocity_lpf_.fill(0.0);
  reference_anchor_angular_velocity_lpf_.fill(0.0);
}

void MotionHandler::updateReferenceVelocityFilters(
    std::vector<MotionFrame> &frames) {
  if (frames.empty()) {
    return;
  }

  if (!reference_velocity_lpf_initialized_) {
    reference_joint_velocity_lpf_ = frames.front().joint_velocity;
    reference_anchor_linear_velocity_lpf_ =
        frames.front().anchor_linear_velocity_b;
    reference_anchor_angular_velocity_lpf_ =
        frames.front().anchor_angular_velocity_b;
    reference_velocity_lpf_initialized_ = true;
  }

  for (MotionFrame &frame : frames) {
    LowPassInPlace(frame.joint_velocity, reference_joint_velocity_lpf_,
                   static_cast<double>(config_.policy_hz),
                   config_.reference_joint_velocity_lpf_cutoff_hz);
    LowPassInPlace(frame.anchor_linear_velocity_b,
                   reference_anchor_linear_velocity_lpf_,
                   static_cast<double>(config_.policy_hz),
                   config_.reference_anchor_linear_velocity_lpf_cutoff_hz);
    LowPassInPlace(frame.anchor_angular_velocity_b,
                   reference_anchor_angular_velocity_lpf_,
                   static_cast<double>(config_.policy_hz),
                   config_.reference_anchor_angular_velocity_lpf_cutoff_hz);
  }
}

} // namespace igris_c_gmt_public
