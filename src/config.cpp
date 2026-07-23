#include "public_inference_module/config.hpp"

#include "yaml-cpp/yaml.h"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace public_inference_module {
namespace {

namespace fs = std::filesystem;

const Vector23d &GetDefaultQDefault() {
    static const Vector23d kValue = [] {
        Vector23d value;
        value << 0.13, 0.13, 0.0, 0.13, -0.13, 0.0, 0.0, 0.0, 0.0, -0.3, -0.3, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, -0.1, -0.1, 0.0,
            0.0;
        return value;
    }();
    return kValue;
}

const Vector23d &GetDefaultJointPosLimitHigh() {
    static const Vector23d kValue = [] {
        Vector23d value;
        value << 0.99818407346, 0.99818407346, 1.53938040046, 3.10843672746, 0.14084407346, 0.402094, 1.53938040046, 1.53938040046, 0.290046,
            -0.021, -0.021, 0.47457964775, 0.47457964775, 2.3296325451, 0.2734380551, 1.53938040046, 1.53938040046, 1.98, 1.98, 0.620957,
            0.620957, 0.369274, 0.338874;
        return value;
    }();
    return kValue;
}

const Vector23d &GetDefaultJointPosLimitLow() {
    static const Vector23d kValue = [] {
        Vector23d value;
        value << -3.09977672746, -3.09977672746, -1.53938040046, -0.14084407346, -3.10843672746, -0.402094, -1.53938040046, -1.53938040046,
            -0.734446, -2.079, -2.079, -2.01661487275, -2.01661487275, -0.2734380551, -2.3296325451, -1.53938040046, -1.53938040046, 0.02,
            0.02, -0.686657, -0.686657, -0.338874, -0.369274;
        return value;
    }();
    return kValue;
}

Vector23d GetResidualActionScale() {
    Vector23d scale;
    scale << 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2, 0.2;
    return scale;
}

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
    if (root["public_inference_module"]) {
        return root["public_inference_module"];
    }
    return root;
}

template <typename T> T GetOr(const YAML::Node &node, const char *key, const T &default_value) {
    if (!node || !node[key]) {
        return default_value;
    }
    return node[key].as<T>();
}

template <typename Derived>
void CopyFixedSequence(const YAML::Node &node, const char *key, Eigen::MatrixBase<Derived> &target) {
    const YAML::Node value = node[key];
    if (!value || !value.IsSequence() || value.size() != static_cast<std::size_t>(target.size())) {
        throw std::runtime_error("Expected sequence '" + std::string(key) + "' with size " + std::to_string(target.size()));
    }
    for (Eigen::Index i = 0; i < target.size(); ++i) {
        target(i) = value[static_cast<std::size_t>(i)].as<double>();
    }
}

igris_c::msg::dds::KinematicMode ParseKinematicMode(const YAML::Node &node) {
    if (node.IsScalar()) {
        const std::string value = node.as<std::string>();
        if (value == "MS") {
            return igris_c::msg::dds::KinematicMode::MS;
        }
        if (value == "PJS") {
            return igris_c::msg::dds::KinematicMode::PJS;
        }
    }

    const int raw = node.as<int>();
    if (raw == 0) {
        return igris_c::msg::dds::KinematicMode::MS;
    }
    if (raw == 1) {
        return igris_c::msg::dds::KinematicMode::PJS;
    }
    throw std::runtime_error("Unsupported kinematic mode value");
}

void CopyKinematicModes(const YAML::Node &node, std::array<igris_c::msg::dds::KinematicMode, kParallelGroupCount> &target) {
    if (!node || !node.IsSequence() || node.size() != target.size()) {
        throw std::runtime_error("Expected command.kinematic_modes with 5 entries");
    }
    for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] = ParseKinematicMode(node[i]);
    }
}

void CopyJointLimits(const YAML::Node &node, const char *key, Vector23d &target) {
    const YAML::Node value = node[key];
    if (!value) {
        return;
    }
    if (!value.IsSequence()) {
        throw std::runtime_error("Expected sequence for '" + std::string(key) + "'");
    }

    if (value.size() == static_cast<std::size_t>(target.size())) {
        for (Eigen::Index i = 0; i < target.size(); ++i) {
            target(i) = value[static_cast<std::size_t>(i)].as<double>();
        }
        return;
    }

    if (value.size() == static_cast<std::size_t>(kModelDof)) {
        for (std::size_t i = 0; i < kActionsToSystemJointMapping.size(); ++i) {
            target(static_cast<Eigen::Index>(i)) = value[static_cast<std::size_t>(kActionsToSystemJointMapping[i])].as<double>();
        }
        return;
    }

    throw std::runtime_error("Expected '" + std::string(key) + "' to have 23 or 31 entries");
}

void CopyCommandVector(const YAML::Node &node, const char *key, VectorQd &target) {
    const YAML::Node value = node[key];
    if (!value || !value.IsSequence()) {
        throw std::runtime_error("Expected sequence for command." + std::string(key));
    }

    if (value.size() == static_cast<std::size_t>(target.size())) {
        for (Eigen::Index i = 0; i < target.size(); ++i) {
            target(i) = value[static_cast<std::size_t>(i)].as<double>();
        }
        return;
    }

    if (value.size() == static_cast<std::size_t>(kRlNumJointActions)) {
        for (std::size_t i = 0; i < kActionsToSystemJointMapping.size(); ++i) {
            target(kActionsToSystemJointMapping[i]) = value[i].as<double>();
        }
        return;
    }

    throw std::runtime_error("Expected command." + std::string(key) + " to have 23 or " + std::to_string(target.size()) + " entries");
}

}  // namespace

InferenceConfig LoadInferenceConfig(const std::string &config_path) {
    const YAML::Node root = YAML::LoadFile(config_path);
    const YAML::Node cfg  = GetRoot(root);
    const fs::path base_dir = fs::path(config_path).parent_path();

    InferenceConfig config;
    config.kinematic_modes.fill(igris_c::msg::dds::KinematicMode::PJS);
    config.q_default            = GetDefaultQDefault();
    config.joint_pos_limit_high = GetDefaultJointPosLimitHigh();
    config.joint_pos_limit_low  = GetDefaultJointPosLimitLow();
    config.action_scale         = Vector23d::Constant(0.25);

    const YAML::Node robot_cfg = cfg["robot"];
    config.domain_id           = GetOr<int>(robot_cfg, "domain_id", config.domain_id);
    config.robot_namespace     = GetOr<std::string>(robot_cfg, "namespace", config.robot_namespace);
    config.cyclonedds_xml_path = ResolvePath(base_dir, GetOr<std::string>(robot_cfg, "cyclonedds_xml_path", ""));

    const YAML::Node loop_cfg = cfg["loop"];
    config.control_hz         = GetOr<int>(loop_cfg, "control_hz", config.control_hz);
    config.policy_hz          = GetOr<int>(loop_cfg, "policy_hz", config.policy_hz);
    if (config.control_hz <= 0 || config.policy_hz <= 0 || (config.control_hz % config.policy_hz) != 0) {
        throw std::runtime_error("loop.control_hz must be positive and divisible by loop.policy_hz");
    }

    const YAML::Node motion_source_cfg = cfg["motion_source"];
    config.motion_source.type          = GetOr<std::string>(motion_source_cfg, "type", config.motion_source.type);
    config.motion_source.csv_path      = ResolvePath(base_dir, GetOr<std::string>(motion_source_cfg, "csv_path", ""));
    config.motion_source.onnx_path     = ResolvePath(base_dir, GetOr<std::string>(motion_source_cfg, "onnx_path", ""));
    config.motion_source.ros_config_path = ResolvePath(base_dir, GetOr<std::string>(motion_source_cfg, "ros_config_path", ""));
    config.motion_source.layout        = GetOr<std::string>(motion_source_cfg, "layout", config.motion_source.layout);
    config.motion_source.fps           = GetOr<double>(motion_source_cfg, "fps", config.motion_source.fps);
    config.motion_source.loop          = GetOr<bool>(motion_source_cfg, "loop", config.motion_source.loop);

    const YAML::Node policy_cfg = cfg["policy"];
    config.policy_onnx_path     = ResolvePath(base_dir, GetOr<std::string>(policy_cfg, "onnx_path", ""));
    config.use_motion_residual_action = GetOr<bool>(policy_cfg, "use_motion_residual_action", config.use_motion_residual_action);
    config.action_scale = config.use_motion_residual_action ? GetResidualActionScale() : Vector23d::Constant(0.25);
    if (policy_cfg && policy_cfg["action_scale"]) {
        CopyFixedSequence(policy_cfg, "action_scale", config.action_scale);
    }
    if (policy_cfg && policy_cfg["q_default"]) {
        CopyFixedSequence(policy_cfg, "q_default", config.q_default);
    }

    const YAML::Node filter_cfg = cfg["filters"];
    config.q_dot_lpf_cutoff_hz       = GetOr<double>(filter_cfg, "q_dot_lpf_cutoff_hz", config.q_dot_lpf_cutoff_hz);
    config.imu_ang_vel_lpf_cutoff_hz = GetOr<double>(filter_cfg, "imu_ang_vel_lpf_cutoff_hz", config.imu_ang_vel_lpf_cutoff_hz);

    const YAML::Node safety_cfg = cfg["safety"];
    config.motion_frame_timeout_ms  = GetOr<double>(safety_cfg, "motion_frame_timeout_ms", config.motion_frame_timeout_ms);
    config.non_parallel_safety_coef = GetOr<double>(safety_cfg, "non_parallel_safety_coef", config.non_parallel_safety_coef);
    CopyJointLimits(safety_cfg, "joint_position_max", config.joint_pos_limit_high);
    CopyJointLimits(safety_cfg, "joint_position_min", config.joint_pos_limit_low);

    const YAML::Node command_cfg = cfg["command"];
    CopyKinematicModes(command_cfg["kinematic_modes"], config.kinematic_modes);
    CopyCommandVector(command_cfg, "kp", config.kp);
    CopyCommandVector(command_cfg, "kd", config.kd);

    return config;
}

RosMotionConfig LoadRosMotionConfig(const std::string &config_path) {
    const YAML::Node root = YAML::LoadFile(config_path);
    const YAML::Node cfg  = root["public_inference_module_ros"] ? root["public_inference_module_ros"] : root;
    const YAML::Node ros  = cfg["ros"] ? cfg["ros"] : cfg;

    RosMotionConfig config;
    config.domain_id  = static_cast<std::size_t>(GetOr<int>(ros, "domain_id", static_cast<int>(config.domain_id)));
    config.node_name  = GetOr<std::string>(ros, "node_name", config.node_name);
    config.topic_name = GetOr<std::string>(ros, "topic_name", config.topic_name);
    config.recorded_reference_topic_name =
        GetOr<std::string>(ros, "recorded_reference_topic_name", config.recorded_reference_topic_name);
    config.qos_depth  = static_cast<std::size_t>(GetOr<int>(ros, "qos_depth", static_cast<int>(config.qos_depth)));
    config.best_effort = GetOr<bool>(ros, "best_effort", config.best_effort);
    return config;
}

}  // namespace public_inference_module
