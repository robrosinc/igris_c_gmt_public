#include "modules/state_handler.h"

#include <algorithm>
#include <cmath>

namespace igris_c_gmt_public {
namespace {

constexpr double kPi = 3.14159265358979323846;

Eigen::Matrix3d RotateWithY(double angle_rad) {
  const double c = std::cos(angle_rad);
  const double s = std::sin(angle_rad);
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  rotation(0, 0) = c;
  rotation(0, 2) = s;
  rotation(2, 0) = -s;
  rotation(2, 2) = c;
  return rotation;
}

Eigen::Matrix3d RotationMatrixFromWxyz(const std::array<float, 4> &quat_wxyz) {
  Eigen::Quaterniond quaternion(
      static_cast<double>(quat_wxyz[0]), static_cast<double>(quat_wxyz[1]),
      static_cast<double>(quat_wxyz[2]), static_cast<double>(quat_wxyz[3]));
  quaternion.normalize();
  return quaternion.toRotationMatrix();
}

Eigen::Matrix3d
ComputeImuRotationMatrix(const igris_c::msg::dds::LowState &state) {
  return RotationMatrixFromWxyz(state.imu_state().quaternion());
}

Eigen::Matrix3d ComputeBaseLinkRotm(const igris_c::msg::dds::LowState &state) {
  return ComputeImuRotationMatrix(state) * RotateWithY(-kPi / 2.0);
}

Eigen::Vector3d
ComputeProjectedGravity(const igris_c::msg::dds::LowState &state) {
  static const Eigen::Vector3d kWorldGravity(0.0, 0.0, -1.0);
  return ComputeBaseLinkRotm(state).transpose() * kWorldGravity;
}

Eigen::Vector3d
ComputeBaseLinkAngularVelocity(const igris_c::msg::dds::LowState &state,
                               const Eigen::Vector3d &filtered_imu_gyro) {
  const Eigen::Matrix3d imu_rotation = ComputeImuRotationMatrix(state);
  const Eigen::Matrix3d base_link_rotm = ComputeBaseLinkRotm(state);
  const Eigen::Vector3d base_link_angular_velocity_global =
      imu_rotation * filtered_imu_gyro;
  return base_link_rotm.transpose() * base_link_angular_velocity_global;
}

template <int N>
Eigen::Matrix<double, N, 1> LowPass(const Eigen::Matrix<double, N, 1> &input,
                                    const Eigen::Matrix<double, N, 1> &prev,
                                    double sample_hz, double cutoff_hz) {
  if (cutoff_hz <= 0.0 || sample_hz <= 0.0) {
    return input;
  }
  const double dt = 1.0 / sample_hz;
  const double rc = 1.0 / (2.0 * kPi * cutoff_hz);
  const double alpha = dt / (rc + dt);
  return prev + alpha * (input - prev);
}

} // namespace

void StateHandler::configure(const InferenceConfig &config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
  latest_state_ = ProcessedState{};
  has_state_ = false;
  joint_velocity_lpf_.setZero();
  imu_angular_velocity_lpf_.setZero();
  velocity_lpf_initialized_ = false;
}

void StateHandler::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  latest_state_ = ProcessedState{};
  has_state_ = false;
  joint_velocity_lpf_.setZero();
  imu_angular_velocity_lpf_.setZero();
  velocity_lpf_initialized_ = false;
}

bool StateHandler::update(const igris_c::msg::dds::LowState &state,
                          uint64_t sequence) {
  ProcessedState processed;
  processed.raw_state = state;
  processed.sequence = sequence;
  processed.valid = true;

  updateVelocityFilters(state);

  for (int i = 0; i < kModelDof; ++i) {
    processed.joint_position(i) = static_cast<double>(
        state.joint_state()[static_cast<std::size_t>(i)].q());
  }
  for (std::size_t i = 0; i < kObsToSystemJointMapping.size(); ++i) {
    const int joint_index = kObsToSystemJointMapping[i];
    processed.rl_joint_position(static_cast<Eigen::Index>(i)) =
        processed.joint_position(joint_index);
    processed.rl_joint_velocity(static_cast<Eigen::Index>(i)) =
        joint_velocity_lpf_(joint_index);
  }
  processed.base_angular_velocity =
      ComputeBaseLinkAngularVelocity(state, imu_angular_velocity_lpf_);
  processed.projected_gravity = ComputeProjectedGravity(state);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_state_ = processed;
    has_state_ = true;
  }
  return true;
}

bool StateHandler::readLatest(ProcessedState &state) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!has_state_) {
    return false;
  }
  state = latest_state_;
  return true;
}

void StateHandler::updateVelocityFilters(
    const igris_c::msg::dds::LowState &state) {
  VectorQd joint_velocity_raw = VectorQd::Zero();
  for (int i = 0; i < kModelDof; ++i) {
    joint_velocity_raw(i) = static_cast<double>(
        state.joint_state()[static_cast<std::size_t>(i)].dq());
  }

  Eigen::Vector3d imu_gyro_raw;
  for (int i = 0; i < 3; ++i) {
    imu_gyro_raw(i) = static_cast<double>(
        state.imu_state().gyroscope()[static_cast<std::size_t>(i)]);
  }

  if (!velocity_lpf_initialized_) {
    joint_velocity_lpf_ = joint_velocity_raw;
    imu_angular_velocity_lpf_ = imu_gyro_raw;
    velocity_lpf_initialized_ = true;
    return;
  }

  joint_velocity_lpf_ =
      config_.q_dot_lpf_cutoff_hz > 0.0
          ? LowPass<kModelDof>(joint_velocity_raw, joint_velocity_lpf_,
                               static_cast<double>(config_.control_hz),
                               config_.q_dot_lpf_cutoff_hz)
          : joint_velocity_raw;
  imu_angular_velocity_lpf_ =
      config_.imu_ang_vel_lpf_cutoff_hz > 0.0
          ? LowPass<3>(imu_gyro_raw, imu_angular_velocity_lpf_,
                       static_cast<double>(config_.control_hz),
                       config_.imu_ang_vel_lpf_cutoff_hz)
          : imu_gyro_raw;
}

} // namespace igris_c_gmt_public
