#pragma once

#include "core/config.hpp"
#include "core/pipeline_types.hpp"

#include <mutex>

namespace igris_c_gmt_public {

class StateHandler {
public:
  void configure(const InferenceConfig &config);
  void reset();
  bool update(const igris_c::msg::dds::LowState &state, uint64_t sequence);
  bool readLatest(ProcessedState &state) const;

private:
  void updateVelocityFilters(const igris_c::msg::dds::LowState &state);

private:
  InferenceConfig config_;
  mutable std::mutex mutex_;
  ProcessedState latest_state_;
  bool has_state_ = false;

  VectorQd joint_velocity_lpf_ = VectorQd::Zero();
  Eigen::Vector3d imu_angular_velocity_lpf_ = Eigen::Vector3d::Zero();
  bool velocity_lpf_initialized_ = false;
};

} // namespace igris_c_gmt_public
