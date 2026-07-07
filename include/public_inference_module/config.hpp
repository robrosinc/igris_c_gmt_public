#pragma once

#include "public_inference_module/types.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace public_inference_module {

struct MotionSourceConfig {
    std::string type    = "null";
    std::string csv_path;
    std::string onnx_path;
    std::string ros_config_path;
    std::string layout = "reference_tracking_v1";
    double fps          = 50.0;
    bool loop           = true;
};

struct RosMotionConfig {
    std::size_t domain_id     = 0;
    std::string node_name     = "public_inference_module_motion_subscriber";
    std::string topic_name    = "/gmr/teleop/retarget_frame";
    std::string recorded_reference_topic_name = "/gmr/teleop/use_recorded_reference";
    std::size_t qos_depth     = 1;
    bool best_effort          = true;
};

struct InferenceConfig {
    int domain_id                  = 0;
    std::string robot_namespace;
    std::string cyclonedds_xml_path;
    int control_hz                = 300;
    int policy_hz                 = 50;
    MotionSourceConfig motion_source;
    std::string policy_onnx_path;
    bool use_motion_residual_action = false;
    bool zero_proprioception_ankle_velocity = true;
    VectorQd kp                     = VectorQd::Zero();
    VectorQd kd                     = VectorQd::Zero();
    std::array<igris_c::msg::dds::KinematicMode, kParallelGroupCount> kinematic_modes{};
    Vector23d action_scale            = Vector23d::Constant(0.25);
    Vector23d q_default               = Vector23d::Zero();
    Vector23d joint_pos_limit_high    = Vector23d::Zero();
    Vector23d joint_pos_limit_low     = Vector23d::Zero();
    double q_dot_lpf_cutoff_hz        = 16.0;
    double imu_ang_vel_lpf_cutoff_hz  = 50.0;
    double motion_frame_timeout_ms    = 100.0;
    double non_parallel_safety_coef   = 0.8;
};

InferenceConfig LoadInferenceConfig(const std::string &config_path);
RosMotionConfig LoadRosMotionConfig(const std::string &config_path);

}  // namespace public_inference_module
