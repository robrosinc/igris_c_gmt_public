#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace igris_c_gmt_public {

struct MotionSourceConfig {
  std::string type = "null";
  std::string csv_path;
  std::string onnx_path;
  std::string ros_config_path;
  double fps = 50.0;
  double frame_timeout_ms = 100.0;
  std::vector<int> frame_stack_offsets{0};
  bool loop = true;
};

struct LoggingConfig {
  bool enabled = false;
  std::string csv_path;
  bool only_low_command_mode = true;
};

struct RosMotionConfig {
  std::size_t domain_id = 0;
  std::string node_name = "igris_c_gmt_public_motion_subscriber";
  std::string topic_name = "/gmr/teleop/retarget_frame";
  std::string recorded_reference_topic_name =
      "/gmr/teleop/use_recorded_reference";
  std::size_t qos_depth = 1;
  bool best_effort = true;
};

struct InferenceConfig {
  int domain_id = 0;
  std::string robot_namespace;
  std::string cyclonedds_xml_path;
  int control_hz = 300;
  int policy_hz = 50;
  std::string obs_config_path;
  std::string action_config_path;
  MotionSourceConfig motion_source;
  std::string policy_onnx_path;
  double q_dot_lpf_cutoff_hz = 16.0;
  double imu_ang_vel_lpf_cutoff_hz = 50.0;
  double reference_joint_velocity_lpf_cutoff_hz = 0.0;
  double reference_anchor_linear_velocity_lpf_cutoff_hz = 0.0;
  double reference_anchor_angular_velocity_lpf_cutoff_hz = 0.0;
  LoggingConfig logging;
};

InferenceConfig LoadInferenceConfig(const std::string &config_path);
RosMotionConfig LoadRosMotionConfig(const std::string &config_path);

} // namespace igris_c_gmt_public
