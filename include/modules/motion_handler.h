#pragma once

#include "core/config.hpp"
#include "core/types.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace igris_c_gmt_public {

class MotionFrameBuffer;
class MotionFrameSource;
class RosMotionReceiver;

struct MotionHandlerOutput {
  bool available = false;
  bool mode_changed = false;
  uint64_t step = 0;
  std::string mode = "waiting";
  std::vector<MotionFrame> frames;
};

class MotionHandler {
public:
  MotionHandler();
  ~MotionHandler();

  int configure(const InferenceConfig &config);
  void reset();
  void stop();

  bool read(MotionHandlerOutput &output,
            std::chrono::steady_clock::time_point now =
                std::chrono::steady_clock::now());
  void advance();

private:
  bool transitionTo(const std::string &mode);

private:
  InferenceConfig config_;
  std::unique_ptr<MotionFrameSource> replay_source_;
  std::unique_ptr<MotionFrameSource> recorded_source_;
  std::unique_ptr<RosMotionReceiver> ros_receiver_;
  std::shared_ptr<MotionFrameBuffer> ros_motion_buffer_;

  std::string active_mode_ = "waiting";
  uint64_t motion_step_ = 0;

  bool live_motion_seen_ = false;
  uint64_t last_live_motion_seq_ = 0;
  std::chrono::steady_clock::time_point last_live_motion_wall_time_ =
      std::chrono::steady_clock::now();
};

} // namespace igris_c_gmt_public
