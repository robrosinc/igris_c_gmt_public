#include "core/config.hpp"

#include "yaml-cpp/yaml.h"

#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace igris_c_gmt_public {
namespace {

namespace fs = std::filesystem;

std::string ResolvePath(const fs::path &base_dir, const std::string &value) {
  if (value.empty()) {
    return value;
  }
  const fs::path path(value);
  if (path.is_absolute()) {
    return path.lexically_normal().string();
  }
  return (base_dir / path).lexically_normal().string();
}

YAML::Node GetRoot(const YAML::Node &root) {
  const YAML::Node config = root["igris_c_gmt_public"];
  if (!config) {
    throw std::runtime_error(
        "params.yaml must contain igris_c_gmt_public root");
  }
  return config;
}

template <typename T>
T GetOr(const YAML::Node &node, const char *key, const T &default_value) {
  if (!node || !node[key]) {
    return default_value;
  }
  return node[key].as<T>();
}

std::vector<int> MotionFrameOffsetsFromTerm(const YAML::Node &term_node) {
  const std::string function_name = GetOr<std::string>(term_node, "function", "");
  if (function_name != "motion_frame_stack") {
    return {0};
  }

  const YAML::Node params = term_node["params"];
  const int past_frame_count = GetOr<int>(params, "past_frame_count", 0);
  const int future_frame_count = GetOr<int>(params, "future_frame_count", 0);
  const int stride = GetOr<int>(params, "stride", 1);
  const bool include_current = GetOr<bool>(params, "include_current", true);
  if (past_frame_count < 0 || future_frame_count < 0 || stride <= 0) {
    throw std::runtime_error(
        "motion_frame_stack requires non-negative frame counts and positive stride");
  }

  std::vector<int> offsets;
  for (int offset = -past_frame_count * stride; offset < 0; offset += stride) {
    offsets.push_back(offset);
  }
  if (include_current) {
    offsets.push_back(0);
  }
  for (int i = 1; i <= future_frame_count; ++i) {
    offsets.push_back(i * stride);
  }
  if (offsets.empty()) {
    throw std::runtime_error(
        "motion_frame_stack requires at least one frame offset");
  }
  return offsets;
}

std::vector<int> CollectMotionFrameOffsets(const std::string &obs_config_path) {
  const YAML::Node root = YAML::LoadFile(obs_config_path);
  const YAML::Node obs = root["igris_c_gmt_public_observation"];
  if (!obs) {
    throw std::runtime_error(
        "obs.yaml must contain igris_c_gmt_public_observation root");
  }

  std::set<int> offsets;
  offsets.insert(0);
  const YAML::Node groups = obs["groups"];
  if (!groups || !groups.IsSequence()) {
    return {0};
  }
  for (const YAML::Node &group_node : groups) {
    const YAML::Node terms = group_node["terms"];
    if (!terms || !terms.IsSequence()) {
      continue;
    }
    for (const YAML::Node &term_node : terms) {
      for (int offset : MotionFrameOffsetsFromTerm(term_node)) {
        offsets.insert(offset);
      }
    }
  }
  return std::vector<int>(offsets.begin(), offsets.end());
}

} // namespace

InferenceConfig LoadInferenceConfig(const std::string &config_path) {
  const YAML::Node root = YAML::LoadFile(config_path);
  const YAML::Node cfg = GetRoot(root);
  const fs::path base_dir = fs::path(config_path).parent_path();

  InferenceConfig config;
  const YAML::Node robot_cfg = cfg["robot"];
  config.domain_id = GetOr<int>(robot_cfg, "domain_id", config.domain_id);
  config.robot_namespace =
      GetOr<std::string>(robot_cfg, "namespace", config.robot_namespace);
  config.cyclonedds_xml_path = ResolvePath(
      base_dir, GetOr<std::string>(robot_cfg, "cyclonedds_xml_path", ""));

  const YAML::Node loop_cfg = cfg["loop"];
  config.control_hz = GetOr<int>(loop_cfg, "control_hz", config.control_hz);
  config.policy_hz = GetOr<int>(loop_cfg, "policy_hz", config.policy_hz);
  if (config.control_hz <= 0 || config.policy_hz <= 0 ||
      (config.control_hz % config.policy_hz) != 0) {
    throw std::runtime_error(
        "loop.control_hz must be positive and divisible by loop.policy_hz");
  }

  const YAML::Node observation_cfg = cfg["observation"];
  config.obs_config_path = ResolvePath(
      base_dir, GetOr<std::string>(observation_cfg, "config_path", "obs.yaml"));
  config.motion_source.frame_stack_offsets =
      CollectMotionFrameOffsets(config.obs_config_path);

  const YAML::Node action_cfg = cfg["action"];
  config.action_config_path = ResolvePath(
      base_dir, GetOr<std::string>(action_cfg, "config_path", "action.yaml"));

  const YAML::Node motion_source_cfg = cfg["motion_source"];
  config.motion_source.type =
      GetOr<std::string>(motion_source_cfg, "type", config.motion_source.type);
  config.motion_source.csv_path = ResolvePath(
      base_dir, GetOr<std::string>(motion_source_cfg, "csv_path", ""));
  config.motion_source.onnx_path = ResolvePath(
      base_dir, GetOr<std::string>(motion_source_cfg, "onnx_path", ""));
  config.motion_source.ros_config_path = ResolvePath(
      base_dir, GetOr<std::string>(motion_source_cfg, "ros_config_path", ""));
  config.motion_source.fps =
      GetOr<double>(motion_source_cfg, "fps", config.motion_source.fps);
  config.motion_source.frame_timeout_ms =
      GetOr<double>(motion_source_cfg, "frame_timeout_ms",
                    config.motion_source.frame_timeout_ms);
  config.motion_source.loop =
      GetOr<bool>(motion_source_cfg, "loop", config.motion_source.loop);

  const YAML::Node policy_cfg = cfg["policy"];
  config.policy_onnx_path =
      ResolvePath(base_dir, GetOr<std::string>(policy_cfg, "onnx_path", ""));

  const YAML::Node filter_cfg = cfg["filters"];
  config.q_dot_lpf_cutoff_hz = GetOr<double>(filter_cfg, "q_dot_lpf_cutoff_hz",
                                             config.q_dot_lpf_cutoff_hz);
  config.imu_ang_vel_lpf_cutoff_hz =
      GetOr<double>(filter_cfg, "imu_ang_vel_lpf_cutoff_hz",
                    config.imu_ang_vel_lpf_cutoff_hz);
  config.reference_joint_velocity_lpf_cutoff_hz =
      GetOr<double>(filter_cfg, "reference_joint_velocity_lpf_cutoff_hz",
                    config.reference_joint_velocity_lpf_cutoff_hz);
  config.reference_anchor_linear_velocity_lpf_cutoff_hz =
      GetOr<double>(filter_cfg, "reference_anchor_linear_velocity_lpf_cutoff_hz",
                    config.reference_anchor_linear_velocity_lpf_cutoff_hz);
  config.reference_anchor_angular_velocity_lpf_cutoff_hz =
      GetOr<double>(filter_cfg,
                    "reference_anchor_angular_velocity_lpf_cutoff_hz",
                    config.reference_anchor_angular_velocity_lpf_cutoff_hz);

  const YAML::Node logging_cfg = cfg["logging"];
  config.logging.enabled =
      GetOr<bool>(logging_cfg, "enabled", config.logging.enabled);
  config.logging.csv_path = ResolvePath(
      base_dir, GetOr<std::string>(logging_cfg, "csv_path",
                                   config.logging.csv_path));
  config.logging.only_low_command_mode =
      GetOr<bool>(logging_cfg, "only_low_command_mode",
                  config.logging.only_low_command_mode);
  if (config.logging.enabled && config.logging.csv_path.empty()) {
    throw std::runtime_error("logging.csv_path is required when logging.enabled is true");
  }

  return config;
}

RosMotionConfig LoadRosMotionConfig(const std::string &config_path) {
  const YAML::Node root = YAML::LoadFile(config_path);
  const YAML::Node cfg = root["igris_c_gmt_public_ros"];
  if (!cfg) {
    throw std::runtime_error(
        "ros.yaml must contain igris_c_gmt_public_ros root");
  }
  const YAML::Node ros = cfg["ros"];
  if (!ros) {
    throw std::runtime_error(
        "ros.yaml must contain igris_c_gmt_public_ros.ros");
  }

  RosMotionConfig config;
  config.domain_id = static_cast<std::size_t>(
      GetOr<int>(ros, "domain_id", static_cast<int>(config.domain_id)));
  config.node_name = GetOr<std::string>(ros, "node_name", config.node_name);
  config.topic_name = GetOr<std::string>(ros, "topic_name", config.topic_name);
  config.recorded_reference_topic_name =
      GetOr<std::string>(ros, "recorded_reference_topic_name",
                         config.recorded_reference_topic_name);
  config.qos_depth = static_cast<std::size_t>(
      GetOr<int>(ros, "qos_depth", static_cast<int>(config.qos_depth)));
  config.best_effort = GetOr<bool>(ros, "best_effort", config.best_effort);
  return config;
}

} // namespace igris_c_gmt_public
