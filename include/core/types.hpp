#pragma once

#include "igris_c_sdk/igris_c_msgs.hpp"

#include <Eigen/Dense>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace igris_c_gmt_public {

inline constexpr int kModelDof = static_cast<int>(igris_c::msg::dds::N_JOINTS);
inline constexpr int kParallelGroupCount = 5;

enum JointIndex {
  _WY = 0,
  _WR = 1,
  _WP = 2,
  _LHP = 3,
  _LHR = 4,
  _LHY = 5,
  _LKP = 6,
  _LAP = 7,
  _LAR = 8,
  _RHP = 9,
  _RHR = 10,
  _RHY = 11,
  _RKP = 12,
  _RAP = 13,
  _RAR = 14,
  _LSP = 15,
  _LSR = 16,
  _LSY = 17,
  _LEP = 18,
  _RSP = 22,
  _RSR = 23,
  _RSY = 24,
  _REP = 25,
};

inline constexpr int kRlNumJointActions = 23;
inline constexpr int kMotionFrameStackLength = 1;

using VectorQd = Eigen::Matrix<double, kModelDof, 1>;
using Vector23d = Eigen::Matrix<double, kRlNumJointActions, 1>;

inline constexpr std::array<int, kRlNumJointActions> kObsToSystemJointMapping =
    {
        _LSP, _RSP, _WY,  _LSR, _RSR, _WR,  _LSY, _RSY, _WP,  _LEP, _REP, _LHP,
        _RHP, _LHR, _RHR, _LHY, _RHY, _LKP, _RKP, _LAP, _RAP, _LAR, _RAR,
};

inline constexpr std::array<int, kRlNumJointActions>
    kActionsToSystemJointMapping = {
        _LSP, _RSP, _WY,  _LSR, _RSR, _WR,  _LSY, _RSY, _WP,  _LEP, _REP, _LHP,
        _RHP, _LHR, _RHR, _LHY, _RHY, _LKP, _RKP, _LAP, _RAP, _LAR, _RAR,
};

struct MotionFrame {
  // Monotonic sequence assigned by the receiver or replay source.
  uint64_t seq = 0;
  // True only after all required source fields have been parsed and normalized.
  bool valid = false;
  // Policy-order reference joint position and velocity for the 23 controlled
  // joints.
  std::array<double, kRlNumJointActions> joint_position{};
  std::array<double, kRlNumJointActions> joint_velocity{};
  // Names for each packed body entry below.
  std::vector<std::string> body_names;
  // XYZ-packed body positions in the motion anchor frame.
  std::vector<double> body_position;
  // WXYZ-packed body orientations in the motion anchor frame.
  std::vector<double> body_orientation;
  // XYZ-packed body linear and angular velocities in the motion anchor frame,
  // when provided.
  std::vector<double> body_linear_velocity;
  std::vector<double> body_angular_velocity;
  // Anchor/root position in the world frame.
  std::array<double, 3> anchor_position{};
  // Anchor/root linear and angular velocities in the world frame.
  std::array<double, 3> anchor_linear_velocity{};
  std::array<double, 3> anchor_angular_velocity{};
  // Anchor/root linear and angular velocities expressed in the anchor/body
  // frame.
  std::array<double, 3> anchor_linear_velocity_b{};
  std::array<double, 3> anchor_angular_velocity_b{};
  // Anchor/root orientation in world frame, stored as wxyz.
  std::array<double, 4> anchor_quaternion_wxyz{1.0, 0.0, 0.0, 0.0};
  bool anchor_quaternion_valid = false;
};

struct InferenceCommand {
  std::array<igris_c::msg::dds::KinematicMode, kParallelGroupCount>
      kinematic_modes{};
  VectorQd q = VectorQd::Zero();
  VectorQd q_dot = VectorQd::Zero();
  VectorQd tau = VectorQd::Zero();
  VectorQd kp = VectorQd::Zero();
  VectorQd kd = VectorQd::Zero();

  InferenceCommand() {
    kinematic_modes.fill(igris_c::msg::dds::KinematicMode::MS);
  }
};

} // namespace igris_c_gmt_public
