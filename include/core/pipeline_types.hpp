#pragma once

#include "core/types.hpp"

#include "igris_c_sdk/igris_c_msgs.hpp"

#include <Eigen/Dense>

#include <map>
#include <string>
#include <vector>

namespace igris_c_gmt_public {

struct ProcessedState {
  igris_c::msg::dds::LowState raw_state;
  uint64_t sequence = 0;
  bool valid = false;

  VectorQd joint_position = VectorQd::Zero();
  Vector23d rl_joint_position = Vector23d::Zero();
  Vector23d rl_joint_velocity = Vector23d::Zero();
  Eigen::Vector3d base_angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d projected_gravity = Eigen::Vector3d::Zero();
};

struct ObservationInput {
  ProcessedState state;
  std::vector<MotionFrame> motion_frames;
  Vector23d q_default = Vector23d::Zero();
  Vector23d last_actions = Vector23d::Zero();
};

struct ObservationResult {
  std::map<std::string, Eigen::VectorXd> groups;
  Vector23d motion_joint_position = Vector23d::Zero();
};

} // namespace igris_c_gmt_public
