#pragma once

#include "core/pipeline_types.hpp"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace igris_c_gmt_public::obs_functions {

using ObsFunction = std::function<Eigen::VectorXd(const ObservationInput &)>;

inline constexpr std::size_t kMotionRootStateObsSize = 6;
inline constexpr std::size_t kGeneralMotionAnchorOrientationObsSize = 6;

// Edit this list when the policy should observe a different body-position
// subset.
inline constexpr std::array kMotionBodyNames = {
    "base_link",
    "Left_Hand",
    "Right_Hand",
    "Link_Ankle_Roll_Left",
    "Link_Ankle_Roll_Right",
    "Link_Elbow_Pitch_Left",
    "Link_Elbow_Pitch_Right",
    "Link_Shoulder_Pitch_Left",
    "Link_Shoulder_Pitch_Right",
    "Link_Knee_Pitch_Left",
    "Link_Knee_Pitch_Right",
    "Link_Hip_Pitch_Left",
    "Link_Hip_Pitch_Right",
};
inline constexpr std::size_t kMotionBodyCount = kMotionBodyNames.size();
inline constexpr std::size_t kMotionBodyPositionObsSize = kMotionBodyCount * 3;
inline constexpr std::size_t kMotionBodyOrientationObsSize =
    kMotionBodyCount * 4;
inline constexpr std::size_t kMotionBodyVelocityObsSize = kMotionBodyCount * 3;

template <typename Derived>
inline Eigen::VectorXd ToVectorXd(const Eigen::MatrixBase<Derived> &value) {
  Eigen::VectorXd result(value.size());
  result = value;
  return result;
}

template <std::size_t N>
inline Eigen::VectorXd ToVectorXd(const std::array<double, N> &value) {
  Eigen::VectorXd result(static_cast<Eigen::Index>(N));
  for (std::size_t i = 0; i < N; ++i) {
    result(static_cast<Eigen::Index>(i)) = value[i];
  }
  return result;
}

inline Eigen::VectorXd ToVectorXd(const std::vector<double> &value) {
  Eigen::VectorXd result(static_cast<Eigen::Index>(value.size()));
  for (std::size_t i = 0; i < value.size(); ++i) {
    result(static_cast<Eigen::Index>(i)) = value[i];
  }
  return result;
}

inline const MotionFrame &LatestMotionFrame(const ObservationInput &input) {
  if (input.motion_frames.empty() || !input.motion_frames.front().valid) {
    throw std::runtime_error("observation term requires a valid motion frame");
  }
  return input.motion_frames.front();
}

inline Eigen::Matrix3d RotateWithY(double angle_rad) {
  const double c = std::cos(angle_rad);
  const double s = std::sin(angle_rad);
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  rotation(0, 0) = c;
  rotation(0, 2) = s;
  rotation(2, 0) = -s;
  rotation(2, 2) = c;
  return rotation;
}

template <typename Scalar>
inline Eigen::Matrix3d
RotationMatrixFromWxyz(const std::array<Scalar, 4> &quat_wxyz) {
  Eigen::Quaterniond quaternion(
      static_cast<double>(quat_wxyz[0]), static_cast<double>(quat_wxyz[1]),
      static_cast<double>(quat_wxyz[2]), static_cast<double>(quat_wxyz[3]));
  quaternion.normalize();
  return quaternion.toRotationMatrix();
}

inline Eigen::Matrix3d
ComputeBaseLinkRotm(const igris_c::msg::dds::LowState &state) {
  constexpr double kPi = 3.14159265358979323846;
  return RotationMatrixFromWxyz(state.imu_state().quaternion()) *
         RotateWithY(-kPi / 2.0);
}

inline Eigen::Quaterniond
ExtractYawQuaternion(const Eigen::Quaterniond &quat_wxyz) {
  const Eigen::Quaterniond quaternion = quat_wxyz.normalized();
  const double yaw = std::atan2(
      2.0 * (quaternion.w() * quaternion.z() + quaternion.x() * quaternion.y()),
      1.0 - 2.0 * (quaternion.y() * quaternion.y() +
                   quaternion.z() * quaternion.z()));
  return Eigen::Quaterniond(std::cos(0.5 * yaw), 0.0, 0.0, std::sin(0.5 * yaw))
      .normalized();
}

inline Eigen::Quaterniond
RemoveYawFromQuaternion(const Eigen::Quaterniond &quat_wxyz) {
  const Eigen::Quaterniond quaternion = quat_wxyz.normalized();
  return (ExtractYawQuaternion(quaternion).conjugate() * quaternion)
      .normalized();
}

template <std::size_t BodyCount, std::size_t ValuesPerBody, std::size_t ObsSize>
inline Eigen::VectorXd
SelectBodyValues(const std::vector<double> &values,
                 const std::vector<std::string> &body_names,
                 const std::array<const char *, BodyCount> &target_names,
                 const std::string &term_name) {
  static_assert(ObsSize == BodyCount * ValuesPerBody);
  if (values.empty() || (values.size() % ValuesPerBody) != 0) {
    throw std::runtime_error("motion " + term_name +
                             " are missing or not body-packed");
  }

  const std::size_t source_body_count = values.size() / ValuesPerBody;
  if (body_names.empty()) {
    throw std::runtime_error("motion body names are missing; " + term_name +
                             " must be selected by name");
  }
  if (body_names.size() != source_body_count) {
    throw std::runtime_error("motion body_names count does not match " +
                             term_name + " count");
  }

  std::unordered_map<std::string, std::size_t> source_index_by_name;
  for (std::size_t i = 0; i < body_names.size(); ++i) {
    const auto insert_result = source_index_by_name.emplace(body_names[i], i);
    if (!insert_result.second) {
      throw std::runtime_error("motion body_names contains duplicate body: " +
                               body_names[i]);
    }
  }

  Eigen::Matrix<double, ObsSize, 1> selected =
      Eigen::Matrix<double, ObsSize, 1>::Zero();
  std::unordered_set<std::string> requested_names;
  for (std::size_t i = 0; i < target_names.size(); ++i) {
    const std::string target_name =
        target_names[i] ? std::string(target_names[i]) : std::string();
    if (target_name.empty()) {
      throw std::runtime_error("motion body target name must not be empty");
    }
    const auto request_insert = requested_names.emplace(target_name);
    if (!request_insert.second) {
      throw std::runtime_error(
          "motion body target list contains duplicate body: " + target_name);
    }
    const auto it = source_index_by_name.find(target_name);
    if (it == source_index_by_name.end()) {
      throw std::runtime_error("motion " + term_name +
                               " is missing required body: " + target_name);
    }
    const std::size_t source_offset = it->second * ValuesPerBody;
    const Eigen::Index target_offset =
        static_cast<Eigen::Index>(i * ValuesPerBody);
    for (std::size_t value_index = 0; value_index < ValuesPerBody;
         ++value_index) {
      selected(target_offset + static_cast<Eigen::Index>(value_index)) =
          values[source_offset + value_index];
    }
  }

  return ToVectorXd(selected);
}

template <std::size_t BodyCount, std::size_t ObsSize>
inline Eigen::VectorXd
SelectBodyPositions(const MotionFrame &frame,
                    const std::array<const char *, BodyCount> &target_names) {
  return SelectBodyValues<BodyCount, 3, ObsSize>(
      frame.body_position, frame.body_names, target_names, "body positions");
}

template <std::size_t BodyCount, std::size_t ObsSize>
inline Eigen::VectorXd SelectBodyOrientations(
    const MotionFrame &frame,
    const std::array<const char *, BodyCount> &target_names) {
  return SelectBodyValues<BodyCount, 4, ObsSize>(frame.body_orientation,
                                                 frame.body_names, target_names,
                                                 "body orientations");
}

template <std::size_t BodyCount, std::size_t ObsSize>
inline Eigen::VectorXd SelectBodyLinearVelocities(
    const MotionFrame &frame,
    const std::array<const char *, BodyCount> &target_names) {
  return SelectBodyValues<BodyCount, 3, ObsSize>(frame.body_linear_velocity,
                                                 frame.body_names, target_names,
                                                 "body linear velocities");
}

template <std::size_t BodyCount, std::size_t ObsSize>
inline Eigen::VectorXd SelectBodyAngularVelocities(
    const MotionFrame &frame,
    const std::array<const char *, BodyCount> &target_names) {
  return SelectBodyValues<BodyCount, 3, ObsSize>(frame.body_angular_velocity,
                                                 frame.body_names, target_names,
                                                 "body angular velocities");
}

// Live robot joint positions in policy joint order.
inline Eigen::VectorXd JointPos(const ObservationInput &input) {
  return ToVectorXd(input.state.rl_joint_position);
}

// Live robot joint positions relative to q_default from obs.yaml.
inline Eigen::VectorXd JointPosRel(const ObservationInput &input) {
  return ToVectorXd(input.state.rl_joint_position - input.q_default);
}

// Live robot joint velocities in policy joint order.
inline Eigen::VectorXd JointVel(const ObservationInput &input) {
  return ToVectorXd(input.state.rl_joint_velocity);
}

// Live robot base angular velocity in the base frame.
inline Eigen::VectorXd BaseAngVel(const ObservationInput &input) {
  return ToVectorXd(input.state.base_angular_velocity);
}

// Gravity vector projected into the live robot base frame.
inline Eigen::VectorXd ProjectedGravity(const ObservationInput &input) {
  return ToVectorXd(input.state.projected_gravity);
}

// Last policy action after action-buffer update.
inline Eigen::VectorXd LastActions(const ObservationInput &input) {
  return ToVectorXd(input.last_actions);
}

// Reference motion joint positions in policy joint order.
inline Eigen::VectorXd MotionJointPos(const ObservationInput &input) {
  return ToVectorXd(LatestMotionFrame(input).joint_position);
}

// Reference motion joint velocities in policy joint order.
inline Eigen::VectorXd MotionJointVel(const ObservationInput &input) {
  return ToVectorXd(LatestMotionFrame(input).joint_velocity);
}

// Selected reference body positions in the motion anchor frame.
inline Eigen::VectorXd MotionBodyPos(const ObservationInput &input) {
  return SelectBodyPositions<kMotionBodyCount, kMotionBodyPositionObsSize>(
      LatestMotionFrame(input), kMotionBodyNames);
}

// Selected reference body orientations in the motion anchor frame, wxyz-packed.
inline Eigen::VectorXd MotionBodyOrientation(const ObservationInput &input) {
  return SelectBodyOrientations<kMotionBodyCount,
                                kMotionBodyOrientationObsSize>(
      LatestMotionFrame(input), kMotionBodyNames);
}

// Selected reference body linear velocities in the motion anchor frame.
inline Eigen::VectorXd MotionBodyLinVel(const ObservationInput &input) {
  return SelectBodyLinearVelocities<kMotionBodyCount,
                                    kMotionBodyVelocityObsSize>(
      LatestMotionFrame(input), kMotionBodyNames);
}

// Selected reference body angular velocities in the motion anchor frame.
inline Eigen::VectorXd MotionBodyAngVel(const ObservationInput &input) {
  return SelectBodyAngularVelocities<kMotionBodyCount,
                                     kMotionBodyVelocityObsSize>(
      LatestMotionFrame(input), kMotionBodyNames);
}

// Reference anchor/root linear velocity in the world frame.
inline Eigen::VectorXd MotionAnchorLinVel(const ObservationInput &input) {
  return ToVectorXd(LatestMotionFrame(input).anchor_linear_velocity);
}

// Reference anchor/root angular velocity in the world frame.
inline Eigen::VectorXd MotionAnchorAngVel(const ObservationInput &input) {
  return ToVectorXd(LatestMotionFrame(input).anchor_angular_velocity);
}

// Reference anchor/root linear velocity in the anchor/body frame.
inline Eigen::VectorXd MotionAnchorLinVelB(const ObservationInput &input) {
  return ToVectorXd(LatestMotionFrame(input).anchor_linear_velocity_b);
}

// Reference anchor/root angular velocity in the anchor/body frame.
inline Eigen::VectorXd MotionAnchorAngVelB(const ObservationInput &input) {
  return ToVectorXd(LatestMotionFrame(input).anchor_angular_velocity_b);
}

// Reference anchor/root orientation relative to the live robot anchor,
// represented by the first two rotation matrix columns.
inline Eigen::VectorXd MotionAnchorOrientation(const ObservationInput &input) {
  const MotionFrame &frame = LatestMotionFrame(input);
  if (!frame.anchor_quaternion_valid) {
    throw std::runtime_error(
        "motion_anchor_orientation requires anchor_quaternion_wxyz");
  }
  const Eigen::Quaterniond robot_anchor_world(
      ComputeBaseLinkRotm(input.state.raw_state));
  Eigen::Quaterniond reference_anchor_world(
      frame.anchor_quaternion_wxyz[0], frame.anchor_quaternion_wxyz[1],
      frame.anchor_quaternion_wxyz[2], frame.anchor_quaternion_wxyz[3]);
  reference_anchor_world.normalize();
  const Eigen::Matrix3d root_state =
      (robot_anchor_world.conjugate() * reference_anchor_world)
          .toRotationMatrix();
  const std::array<double, kGeneralMotionAnchorOrientationObsSize> value = {
      root_state(0, 0), root_state(0, 1), root_state(1, 0),
      root_state(1, 1), root_state(2, 0), root_state(2, 1)};
  return ToVectorXd(value);
}

// Reference anchor/root world height, extracted from anchor_position.z.
inline Eigen::VectorXd MotionAnchorHeight(const ObservationInput &input) {
  Eigen::VectorXd value(1);
  value(0) = LatestMotionFrame(input).anchor_position[2];
  return value;
}

// Reference root state with yaw removed from robot and reference anchors.
inline Eigen::VectorXd MotionRootState(const ObservationInput &input) {
  const MotionFrame &frame = LatestMotionFrame(input);
  if (!frame.anchor_quaternion_valid) {
    throw std::runtime_error(
        "motion_root_state requires anchor_quaternion_wxyz");
  }
  const Eigen::Quaterniond robot_anchor_world(
      ComputeBaseLinkRotm(input.state.raw_state));
  const Eigen::Quaterniond reference_anchor_world(
      frame.anchor_quaternion_wxyz[0], frame.anchor_quaternion_wxyz[1],
      frame.anchor_quaternion_wxyz[2], frame.anchor_quaternion_wxyz[3]);
  const Eigen::Quaterniond robot_anchor_no_yaw =
      RemoveYawFromQuaternion(robot_anchor_world);
  const Eigen::Quaterniond reference_no_yaw =
      RemoveYawFromQuaternion(reference_anchor_world);
  const Eigen::Matrix3d root_state =
      (robot_anchor_no_yaw.conjugate() * reference_no_yaw).toRotationMatrix();
  const std::array<double, kMotionRootStateObsSize> value = {
      root_state(0, 0), root_state(0, 1), root_state(1, 0),
      root_state(1, 1), root_state(2, 0), root_state(2, 1)};
  return ToVectorXd(value);
}

inline const std::unordered_map<std::string, ObsFunction> &Registry() {
  static const std::unordered_map<std::string, ObsFunction> registry = {
      {"joint_pos", JointPos},
      {"joint_pos_rel", JointPosRel},
      {"joint_vel", JointVel},
      {"base_ang_vel", BaseAngVel},
      {"projected_gravity", ProjectedGravity},
      {"last_actions", LastActions},
      {"motion_joint_pos", MotionJointPos},
      {"motion_joint_vel", MotionJointVel},
      {"motion_body_pos", MotionBodyPos},
      {"motion_body_orientation", MotionBodyOrientation},
      {"motion_body_lin_vel", MotionBodyLinVel},
      {"motion_body_ang_vel", MotionBodyAngVel},
      {"motion_anchor_lin_vel", MotionAnchorLinVel},
      {"motion_anchor_ang_vel", MotionAnchorAngVel},
      {"motion_anchor_lin_vel_b", MotionAnchorLinVelB},
      {"motion_anchor_ang_vel_b", MotionAnchorAngVelB},
      {"motion_anchor_orientation", MotionAnchorOrientation},
      {"motion_anchor_height", MotionAnchorHeight},
      {"motion_root_state", MotionRootState},
  };
  return registry;
}

} // namespace igris_c_gmt_public::obs_functions
