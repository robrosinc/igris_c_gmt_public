#include "public_inference_module/inference_module.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace public_inference_module {
namespace {

constexpr double kPi = 3.14159265358979323846;

Eigen::Matrix3d RotateWithY(double angle_rad) {
    const double c = std::cos(angle_rad);
    const double s = std::sin(angle_rad);
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    rotation(0, 0)           = c;
    rotation(0, 2)           = s;
    rotation(2, 0)           = -s;
    rotation(2, 2)           = c;
    return rotation;
}

Eigen::Matrix3d RotationMatrixFromWxyz(const std::array<float, 4> &quat_wxyz) {
    Eigen::Quaterniond quaternion(static_cast<double>(quat_wxyz[0]), static_cast<double>(quat_wxyz[1]),
                                  static_cast<double>(quat_wxyz[2]), static_cast<double>(quat_wxyz[3]));
    quaternion.normalize();
    return quaternion.toRotationMatrix();
}

Eigen::Matrix3d ComputeImuRotationMatrix(const igris_c::msg::dds::LowState &state) {
    return RotationMatrixFromWxyz(state.imu_state().quaternion());
}

Eigen::Matrix3d ComputeBaseLinkRotm(const igris_c::msg::dds::LowState &state) {
    return ComputeImuRotationMatrix(state) * RotateWithY(-kPi / 2.0);
}

Eigen::Quaterniond ExtractYawQuaternion(const Eigen::Quaterniond &quat_wxyz) {
    const Eigen::Quaterniond quaternion = quat_wxyz.normalized();
    const double yaw = std::atan2(2.0 * (quaternion.w() * quaternion.z() + quaternion.x() * quaternion.y()),
                                  1.0 - 2.0 * (quaternion.y() * quaternion.y() + quaternion.z() * quaternion.z()));
    return Eigen::Quaterniond(std::cos(0.5 * yaw), 0.0, 0.0, std::sin(0.5 * yaw)).normalized();
}

Eigen::Quaterniond RemoveYawFromQuaternion(const Eigen::Quaterniond &quat_wxyz) {
    const Eigen::Quaterniond quaternion = quat_wxyz.normalized();
    return (ExtractYawQuaternion(quaternion).conjugate() * quaternion).normalized();
}

std::array<double, kRlNumReferenceRootStateObs> RootStateFromReferenceQuaternion(
    const igris_c::msg::dds::LowState &state, const std::array<double, 4> &reference_quaternion_wxyz) {
    const Eigen::Quaterniond robot_anchor_world(ComputeBaseLinkRotm(state));
    const Eigen::Quaterniond reference_anchor_world(reference_quaternion_wxyz[0], reference_quaternion_wxyz[1], reference_quaternion_wxyz[2],
                                                    reference_quaternion_wxyz[3]);
    const Eigen::Quaterniond robot_anchor_no_yaw = RemoveYawFromQuaternion(robot_anchor_world);
    const Eigen::Quaterniond reference_no_yaw    = RemoveYawFromQuaternion(reference_anchor_world);
    const Eigen::Matrix3d reference_root_state   = (robot_anchor_no_yaw.conjugate() * reference_no_yaw).toRotationMatrix();
    return {reference_root_state(0, 0), reference_root_state(0, 1), reference_root_state(1, 0),
            reference_root_state(1, 1), reference_root_state(2, 0), reference_root_state(2, 1)};
}

Eigen::Vector3d ComputeProjectedGravity(const igris_c::msg::dds::LowState &state) {
    static const Eigen::Vector3d kWorldGravity(0.0, 0.0, -1.0);
    return ComputeBaseLinkRotm(state).transpose() * kWorldGravity;
}

Eigen::Vector3d ComputeBaseLinkAngularVelocity(const igris_c::msg::dds::LowState &state, const Eigen::Vector3d &filtered_imu_gyro) {
    const Eigen::Matrix3d imu_rotation                    = ComputeImuRotationMatrix(state);
    const Eigen::Matrix3d base_link_rotm                 = ComputeBaseLinkRotm(state);
    const Eigen::Vector3d base_link_angular_velocity_global = imu_rotation * filtered_imu_gyro;
    return base_link_rotm.transpose() * base_link_angular_velocity_global;
}

template <int N> Eigen::Matrix<double, N, 1> LowPass(const Eigen::Matrix<double, N, 1> &input, const Eigen::Matrix<double, N, 1> &prev,
                                                     double sample_hz, double cutoff_hz) {
    if (cutoff_hz <= 0.0 || sample_hz <= 0.0) {
        return input;
    }
    const double dt    = 1.0 / sample_hz;
    const double rc    = 1.0 / (2.0 * kPi * cutoff_hz);
    const double alpha = dt / (rc + dt);
    return prev + alpha * (input - prev);
}

template <typename Derived> Eigen::VectorXd ToVectorXd(const Eigen::MatrixBase<Derived> &value) {
    Eigen::VectorXd result(value.size());
    result = value;
    return result;
}

template <std::size_t N> Eigen::VectorXd ToVectorXd(const std::array<double, N> &value) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(N));
    for (std::size_t i = 0; i < N; ++i) {
        result(static_cast<Eigen::Index>(i)) = value[i];
    }
    return result;
}

std::vector<int64_t> NormalizeTensorShape(std::vector<int64_t> shape) {
    for (auto &dimension : shape) {
        if (dimension <= 0) {
            dimension = 1;
        }
    }
    return shape;
}

std::vector<int64_t> ObservationTensorShape(const std::string &layout) {
    if (layout == "reference_tracking_v1") {
        return {1, kRlNumReferenceTrackingObs};
    }
    if (layout == "q23_dq23_quatwxyz4") {
        return {1, kRlNumObsHistory};
    }
    throw std::invalid_argument("unsupported motion_source.layout");
}

size_t CountShapeElements(const std::vector<int64_t> &shape) {
    size_t element_count = 1;
    for (const auto dimension : shape) {
        if (dimension <= 0 || static_cast<size_t>(dimension) > (std::numeric_limits<size_t>::max() / element_count)) {
            throw std::overflow_error("invalid tensor shape");
        }
        element_count *= static_cast<size_t>(dimension);
    }
    return element_count;
}

void PrintShape(const std::string &module_label, const std::string &tensor_name, const std::vector<int64_t> &shape) {
    std::cerr << module_label << " input tensor '" << tensor_name << "' shape=[";
    for (size_t j = 0; j < shape.size(); ++j) {
        if (j > 0) {
            std::cerr << ",";
        }
        std::cerr << shape[j];
    }
    std::cerr << "]\n";
}

}  // namespace

InferenceModule::InferenceModule(std::string module_label)
    : module_label_(std::move(module_label)), motion_data_buffer_(std::make_shared<MotionDataBuffer>()) {
    initializeRlState();
}

InferenceModule::~InferenceModule() = default;

int InferenceModule::initialize() {
    initializeRlState();

    session_.reset();
    onnx_input_number_  = 0;
    onnx_output_number_ = 0;
    onnx_input_names_.clear();
    onnx_output_names_.clear();
    onnx_input_names_char_.clear();
    onnx_output_names_char_.clear();
    onnx_input_tensors_.clear();
    onnx_output_tensors_.clear();
    onnx_input_states_buffer_.clear();
    onnx_input_obs_index_      = -1;
    onnx_output_actions_index_ = -1;

    return 0;
}

int InferenceModule::loadConfig(const InferenceConfig &config) {
    config_                = config;
    policy_decimation_     = config_.control_hz / config_.policy_hz;
    q_default_             = config_.q_default;
    joint_pos_limit_high_  = config_.joint_pos_limit_high;
    joint_pos_limit_low_   = config_.joint_pos_limit_low;
    action_scale_          = config_.action_scale;
    command_kp_            = config_.kp;
    command_kd_            = config_.kd;

    for (const auto mode : config_.kinematic_modes) {
        if (mode != igris_c::msg::dds::KinematicMode::PJS) {
            std::cerr << module_label_ << " only supports PJS kinematic modes because the policy/action layout is joint-space.\n";
            return -1;
        }
    }

    if (config_.policy_onnx_path.empty()) {
        std::cerr << module_label_ << " policy.onnx path is empty.\n";
        return -1;
    }
    if (!std::filesystem::exists(config_.policy_onnx_path)) {
        std::cerr << module_label_ << " policy not found: " << config_.policy_onnx_path << "\n";
        return -1;
    }

    try {
        std::cerr << module_label_ << " loading policy: " << config_.policy_onnx_path << "\n";
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        session_options_.AddConfigEntry("session.use_deterministic_compute", "1");
        session_ = std::make_unique<Ort::Session>(env_, config_.policy_onnx_path.c_str(), session_options_);

        onnx_input_number_  = session_->GetInputCount();
        onnx_output_number_ = session_->GetOutputCount();
        std::cerr << module_label_ << " policy tensors: inputs=" << onnx_input_number_ << " outputs=" << onnx_output_number_ << "\n";
    } catch (const Ort::Exception &exception) {
        std::cerr << module_label_ << " failed to create ONNX Runtime session: " << exception.what() << "\n";
        return -1;
    } catch (const std::exception &exception) {
        std::cerr << module_label_ << " failed to create ONNX Runtime session: " << exception.what() << "\n";
        return -1;
    }

    if (onnx_input_number_ == 0 || onnx_output_number_ == 0 || onnx_input_number_ > 16 || onnx_output_number_ > 16) {
        std::cerr << module_label_ << " invalid policy tensor count: inputs=" << onnx_input_number_ << " outputs=" << onnx_output_number_
                  << "\n";
        return -1;
    }

    try {
        Ort::AllocatorWithDefaultOptions allocator;

        onnx_input_names_.resize(onnx_input_number_);
        onnx_output_names_.resize(onnx_output_number_);
        onnx_input_names_char_.resize(onnx_input_number_);
        onnx_output_names_char_.resize(onnx_output_number_);

        for (size_t i = 0; i < onnx_input_number_; ++i) {
            Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(i, allocator);
            onnx_input_names_[i]               = input_name.get();
            onnx_input_names_char_[i]          = onnx_input_names_[i].c_str();
            std::cerr << module_label_ << " input[" << i << "] name='" << onnx_input_names_[i] << "'\n";
            if (onnx_input_names_[i] == "obs") {
                onnx_input_obs_index_ = static_cast<int>(i);
            }
        }

        for (size_t i = 0; i < onnx_output_number_; ++i) {
            Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(i, allocator);
            onnx_output_names_[i]               = output_name.get();
            onnx_output_names_char_[i]          = onnx_output_names_[i].c_str();
            std::cerr << module_label_ << " output[" << i << "] name='" << onnx_output_names_[i] << "'\n";
            if (onnx_output_names_[i] == "actions" || onnx_output_names_[i] == "action") {
                onnx_output_actions_index_ = static_cast<int>(i);
            }
        }

        if (onnx_input_obs_index_ < 0 || onnx_output_actions_index_ < 0) {
            std::cerr << module_label_ << " policy is missing required obs/action tensors.\n";
            return -1;
        }

        for (size_t i = 0; i < onnx_input_number_; ++i) {
            std::vector<int64_t> shape;
            Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(i);
            const auto tensor_info        = input_type_info.GetTensorTypeAndShapeInfo();
            shape = tensor_info.GetShape();
            bool has_dynamic_dimension = false;
            for (const auto dimension : shape) {
                has_dynamic_dimension = has_dynamic_dimension || dimension <= 0;
            }
            if (static_cast<int>(i) == onnx_input_obs_index_ && has_dynamic_dimension) {
                shape = ObservationTensorShape(config_.motion_source.layout);
            }
            shape = NormalizeTensorShape(shape);
            PrintShape(module_label_, onnx_input_names_[i], shape);

            const size_t element_count = CountShapeElements(shape);
            std::cerr << module_label_ << " input tensor '" << onnx_input_names_[i] << "' elements=" << element_count << "\n";
            if (static_cast<int>(i) == onnx_input_obs_index_) {
                try {
                    observation_layout_ = detectObservationLayout(element_count);
                    rl_obs_input_size_  = element_count;
                    rl_obs_flat_.setZero(static_cast<Eigen::Index>(rl_obs_input_size_));
                    std::cerr << module_label_ << " observation layout="
                              << (observation_layout_ == ObservationLayout::ReferenceTracking ? "reference_tracking" : "legacy_history")
                              << " history_len=" << rl_obs_history_len_
                              << " reference_stack_len=" << rl_reference_frame_stack_len_ << "\n";
                } catch (const std::invalid_argument &) {
                    std::cerr << module_label_ << " unsupported obs size " << element_count << ". Legacy obs must be N * "
                              << kRlNumObsCurrent << "; reference_tracking_v1 obs must be exactly "
                              << kRlNumReferenceTrackingObs << ".\n";
                    return -1;
                }
            }
            onnx_input_states_buffer_.emplace_back(element_count, 0.0f);
            onnx_input_tensors_.emplace_back(Ort::Value::CreateTensor<float>(memory_info_, onnx_input_states_buffer_.back().data(),
                                                                             onnx_input_states_buffer_.back().size(), shape.data(), shape.size()));
        }
    } catch (const Ort::Exception &exception) {
        std::cerr << module_label_ << " failed to inspect ONNX policy tensors: " << exception.what() << "\n";
        return -1;
    } catch (const std::exception &exception) {
        std::cerr << module_label_ << " failed to inspect ONNX policy tensors: " << exception.what() << "\n";
        return -1;
    }

    config_loaded_ = true;
    return 0;
}

int InferenceModule::reset(const igris_c::msg::dds::LowState &state, const MotionDataSample *motion_data_sample) {
    if (!config_loaded_) {
        return -1;
    }

    resetPolicyState();
    updateVelocityFilters(state);
    return updateObservationBuffer(state, motion_data_sample);
}

int InferenceModule::compute(const igris_c::msg::dds::LowState &state, const MotionDataSample *motion_data_sample, InferenceCommand &command) {
    if (!config_loaded_) {
        return -1;
    }

    buildHoldCommand(state, command);
    updateVelocityFilters(state);

    const bool rl_step_tick = (control_tick_ % static_cast<uint64_t>(policy_decimation_)) == 0;
    if (rl_step_tick) {
        if (updateObservationBuffer(state, motion_data_sample) != 0) {
            return -1;
        }
        if (obs_updated_) {
            if (computePolicy() != 0) {
                return -1;
            }
            obs_updated_ = false;
        }
    }

    VectorQd desired_q = command.q;
    const Vector23d target_joint_positions = composeTargetJointPositions();
    for (std::size_t i = 0; i < kActionsToSystemJointMapping.size(); ++i) {
        const int joint_index = kActionsToSystemJointMapping[i];
        desired_q(joint_index) = target_joint_positions(static_cast<Eigen::Index>(i));
        command.kp(joint_index) = command_kp_(joint_index);
        command.kd(joint_index) = command_kd_(joint_index);
    }

    processTargetPositions(state, desired_q);
    command.q = desired_q;

    ++control_tick_;
    return 0;
}

bool InferenceModule::buildMotionFrameFromMotionData(const MotionDataSample &motion_data_sample, MotionFrame &motion_frame) const {
    if (!motion_data_sample.valid) {
        return false;
    }

    if (config_.motion_source.layout == "reference_tracking_v1") {
        if (motion_data_sample.values.size() < static_cast<std::size_t>(kMotionDataReferenceTrackingValues)) {
            std::cerr << module_label_ << " motion data payload too small for reference_tracking_v1 layout. size="
                      << motion_data_sample.values.size() << "\n";
            return false;
        }

        motion_frame = MotionFrame{};
        motion_frame.seq      = motion_data_sample.seq;
        motion_frame.stamp_ns = motion_data_sample.stamp_ns;
        motion_frame.valid    = true;
        motion_frame.anchor_quaternion_valid = motion_data_sample.anchor_quaternion_valid;
        if (motion_frame.anchor_quaternion_valid) {
            motion_frame.anchor_quaternion_wxyz = motion_data_sample.anchor_quaternion_wxyz;
        }

        std::size_t offset = 0;
        for (std::size_t i = 0; i < motion_frame.joint_position.size(); ++i) {
            motion_frame.joint_position[i] = motion_data_sample.values[offset + i];
        }
        offset += motion_frame.joint_position.size();
        for (std::size_t i = 0; i < motion_frame.joint_velocity.size(); ++i) {
            motion_frame.joint_velocity[i] = motion_data_sample.values[offset + i];
        }
        offset += motion_frame.joint_velocity.size();

        motion_frame.root_position_z = motion_data_sample.values[offset++];
        for (std::size_t i = 0; i < motion_frame.root_state.size(); ++i) {
            motion_frame.root_state[i] = motion_data_sample.values[offset + i];
        }
        offset += motion_frame.root_state.size();
        for (std::size_t i = 0; i < motion_frame.body_position.size(); ++i) {
            motion_frame.body_position[i] = motion_data_sample.values[offset + i];
        }
        offset += motion_frame.body_position.size();
        for (std::size_t i = 0; i < motion_frame.root_linear_velocity.size(); ++i) {
            motion_frame.root_linear_velocity[i] = motion_data_sample.values[offset + i];
        }
        offset += motion_frame.root_linear_velocity.size();
        for (std::size_t i = 0; i < motion_frame.root_angular_velocity.size(); ++i) {
            motion_frame.root_angular_velocity[i] = motion_data_sample.values[offset + i];
        }
        if (!motion_frame.anchor_quaternion_valid &&
            motion_data_sample.values.size() >= static_cast<std::size_t>(kMotionDataReferenceTrackingValues + 4)) {
            for (std::size_t i = 0; i < motion_frame.anchor_quaternion_wxyz.size(); ++i) {
                motion_frame.anchor_quaternion_wxyz[i] = motion_data_sample.values[static_cast<std::size_t>(kMotionDataReferenceTrackingValues) + i];
            }
            motion_frame.anchor_quaternion_valid = true;
        }
        return true;
    }

    if (config_.motion_source.layout != "q23_dq23_quatwxyz4") {
        std::cerr << module_label_ << " unsupported motion_source.layout: " << config_.motion_source.layout << "\n";
        return false;
    }

    if (motion_data_sample.values.size() < (kRlNumJointActions + kRlNumJointActions + 4)) {
        std::cerr << module_label_ << " motion data payload too small for q23_dq23_quatwxyz4 layout. size="
                  << motion_data_sample.values.size() << "\n";
        return false;
    }

    motion_frame = MotionFrame{};
    motion_frame.seq      = motion_data_sample.seq;
    motion_frame.stamp_ns = motion_data_sample.stamp_ns;
    motion_frame.valid    = true;

    std::size_t offset = 0;
    for (std::size_t i = 0; i < motion_frame.joint_position.size(); ++i) {
        motion_frame.joint_position[i] = motion_data_sample.values[offset + i];
    }
    offset += motion_frame.joint_position.size();
    for (std::size_t i = 0; i < motion_frame.joint_velocity.size(); ++i) {
        motion_frame.joint_velocity[i] = motion_data_sample.values[offset + i];
    }
    offset += motion_frame.joint_velocity.size();
    for (std::size_t i = 0; i < motion_frame.anchor_quaternion_wxyz.size(); ++i) {
        motion_frame.anchor_quaternion_wxyz[i] = motion_data_sample.values[offset + i];
    }
    motion_frame.anchor_quaternion_valid = true;
    return true;
}

bool InferenceModule::buildMotionFrameStackFromMotionData(const MotionDataSample &motion_data_sample,
                                                          std::vector<MotionFrame> &motion_frames) const {
    motion_frames.clear();
    if (!motion_data_sample.valid) {
        return false;
    }

    if (config_.motion_source.layout != "reference_tracking_v1") {
        MotionFrame motion_frame;
        if (!buildMotionFrameFromMotionData(motion_data_sample, motion_frame)) {
            return false;
        }
        motion_frames.assign(kRlReferenceFrameStackLength, motion_frame);
        return true;
    }

    if (motion_data_sample.values.size() == static_cast<std::size_t>(kMotionDataReferenceTrackingValues) ||
        motion_data_sample.values.size() == static_cast<std::size_t>(kMotionDataReferenceTrackingValuesWithAnchor)) {
        MotionFrame motion_frame;
        if (!buildMotionFrameFromMotionData(motion_data_sample, motion_frame)) {
            return false;
        }
        motion_frames.assign(kRlReferenceFrameStackLength, motion_frame);
        return true;
    }

    const std::size_t frame_payload_size = static_cast<std::size_t>(kMotionDataReferenceTrackingValuesWithAnchor);
    const std::size_t expected_payload_size = frame_payload_size * static_cast<std::size_t>(kRlReferenceFrameStackLength);
    if (motion_data_sample.values.size() != expected_payload_size) {
        return false;
    }

    motion_frames.reserve(static_cast<std::size_t>(kRlReferenceFrameStackLength));
    std::size_t offset = 0;
    for (std::size_t frame_index = 0; frame_index < static_cast<std::size_t>(kRlReferenceFrameStackLength); ++frame_index) {
        MotionFrame frame;
        frame.seq      = motion_data_sample.seq + static_cast<uint64_t>(frame_index);
        frame.stamp_ns = motion_data_sample.stamp_ns;
        frame.valid    = true;

        for (std::size_t i = 0; i < frame.joint_position.size(); ++i) {
            frame.joint_position[i] = motion_data_sample.values[offset + i];
        }
        offset += frame.joint_position.size();
        for (std::size_t i = 0; i < frame.joint_velocity.size(); ++i) {
            frame.joint_velocity[i] = motion_data_sample.values[offset + i];
        }
        offset += frame.joint_velocity.size();
        frame.root_position_z = motion_data_sample.values[offset++];
        for (std::size_t i = 0; i < frame.root_state.size(); ++i) {
            frame.root_state[i] = motion_data_sample.values[offset + i];
        }
        offset += frame.root_state.size();
        for (std::size_t i = 0; i < frame.body_position.size(); ++i) {
            frame.body_position[i] = motion_data_sample.values[offset + i];
        }
        offset += frame.body_position.size();
        for (std::size_t i = 0; i < frame.root_linear_velocity.size(); ++i) {
            frame.root_linear_velocity[i] = motion_data_sample.values[offset + i];
        }
        offset += frame.root_linear_velocity.size();
        for (std::size_t i = 0; i < frame.root_angular_velocity.size(); ++i) {
            frame.root_angular_velocity[i] = motion_data_sample.values[offset + i];
        }
        offset += frame.root_angular_velocity.size();
        for (std::size_t i = 0; i < frame.anchor_quaternion_wxyz.size(); ++i) {
            frame.anchor_quaternion_wxyz[i] = motion_data_sample.values[offset + i];
        }
        offset += frame.anchor_quaternion_wxyz.size();
        frame.anchor_quaternion_valid = true;
        motion_frames.push_back(frame);
    }
    return true;
}

void InferenceModule::buildHoldCommand(const igris_c::msg::dds::LowState &state, InferenceCommand &command) const {
    command.kinematic_modes = config_.kinematic_modes;
    command.q.setZero();
    command.q_dot.setZero();
    command.tau.setZero();
    command.kp = command_kp_;
    command.kd = command_kd_;

    for (int i = 0; i < kModelDof; ++i) {
        command.q(i) = static_cast<double>(state.joint_state()[static_cast<std::size_t>(i)].q());
    }
}

void InferenceModule::initializeRlState() {
    rl_obs_flat_            = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(rl_obs_input_size_));
    q_default_              = config_loaded_ ? config_.q_default : Vector23d::Zero();
    rl_q_.setZero();
    rl_q_dot_.setZero();
    rl_motion_q_.setZero();
    rl_motion_q_dot_.setZero();
    rl_base_ang_vel_.setZero();
    rl_projected_grav_.setZero();
    rl_actions_.setZero();
    rl_last_actions_.setZero();
    joint_pos_limit_high_.setZero();
    joint_pos_limit_low_.setZero();
    action_scale_.setConstant(0.25);
    command_kp_.setZero();
    command_kd_.setZero();
    joint_velocity_lpf_.setZero();
    imu_angular_vel_lpf_.setZero();
    velocity_lpf_initialized_ = false;
    latest_motion_frame_stack_.clear();
}

void InferenceModule::resetPolicyState() {
    std::lock_guard<std::mutex> lock(mutex_);
    rl_obs_flat_.setZero();
    rl_q_.setZero();
    rl_q_dot_.setZero();
    rl_motion_q_.setZero();
    rl_motion_q_dot_.setZero();
    rl_base_ang_vel_.setZero();
    rl_projected_grav_.setZero();
    rl_actions_.setZero();
    rl_last_actions_.setZero();
    obs_update_first_                 = true;
    obs_updated_                      = false;
    control_tick_                     = 0;
    joint_velocity_lpf_.setZero();
    imu_angular_vel_lpf_.setZero();
    velocity_lpf_initialized_         = false;
    latest_motion_frame_              = MotionFrame{};
    latest_motion_frame_stack_.clear();
}

int InferenceModule::computePolicy() {
    if (!session_ || onnx_output_actions_index_ < 0) {
        return -1;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto inference_start = std::chrono::steady_clock::now();
    onnx_output_tensors_ = session_->Run(Ort::RunOptions{nullptr}, onnx_input_names_char_.data(), onnx_input_tensors_.data(), onnx_input_number_,
                                         onnx_output_names_char_.data(), onnx_output_number_);
    last_policy_inference_ms_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - inference_start).count();
    ++policy_inference_count_;

    const auto action_info = onnx_output_tensors_[onnx_output_actions_index_].GetTensorTypeAndShapeInfo();
    if (action_info.GetElementCount() < static_cast<size_t>(kRlNumTotalActions)) {
        std::cerr << module_label_ << " expected " << kRlNumTotalActions << " policy outputs but received "
                  << action_info.GetElementCount() << ".\n";
        return -1;
    }

    const float *action_data = onnx_output_tensors_[onnx_output_actions_index_].GetTensorData<float>();
    for (int i = 0; i < kRlNumJointActions; ++i) {
        rl_actions_(i) = std::clamp(static_cast<double>(action_data[i]), -100.0, 100.0);
    }
    return 0;
}

int InferenceModule::updateObservationBuffer(const igris_c::msg::dds::LowState &state, const MotionDataSample *motion_data_sample) {
    if (motion_data_sample == nullptr) {
        return -1;
    }

    std::vector<MotionFrame> motion_frames;
    if (!buildMotionFrameStackFromMotionData(*motion_data_sample, motion_frames)) {
        return -1;
    }
    MotionFrame motion_frame = motion_frames.front();

    rl_last_actions_ = rl_actions_;

    for (std::size_t i = 0; i < kObsToSystemJointMapping.size(); ++i) {
        const int joint_index  = kObsToSystemJointMapping[i];
        rl_q_(static_cast<Eigen::Index>(i)) = static_cast<double>(state.joint_state()[static_cast<std::size_t>(joint_index)].q());
        rl_q_dot_(static_cast<Eigen::Index>(i)) = joint_velocity_lpf_(joint_index);
        if (config_.zero_proprioception_ankle_velocity && isAnkleObservationJoint(joint_index)) {
            rl_q_dot_(static_cast<Eigen::Index>(i)) = 0.0;
        }
        rl_motion_q_(static_cast<Eigen::Index>(i)) = motion_frame.joint_position[i];
        rl_motion_q_dot_(static_cast<Eigen::Index>(i)) = motion_frame.joint_velocity[i];
    }

    for (MotionFrame &reference_frame : motion_frames) {
        if (reference_frame.anchor_quaternion_valid) {
            reference_frame.root_state = RootStateFromReferenceQuaternion(state, reference_frame.anchor_quaternion_wxyz);
        }
    }
    motion_frame = motion_frames.front();
    rl_base_ang_vel_   = ComputeBaseLinkAngularVelocity(state, imu_angular_vel_lpf_);
    rl_projected_grav_ = ComputeProjectedGravity(state);
    latest_motion_frame_ = motion_frame;
    latest_motion_frame_stack_ = std::move(motion_frames);

    struct ObservationSample {
        Eigen::VectorXd motion_joint_position;
        Eigen::VectorXd motion_joint_velocity;
        Eigen::VectorXd base_angular_velocity;
        Eigen::VectorXd projected_gravity;
        Eigen::VectorXd joint_position;
        Eigen::VectorXd joint_velocity;
        Eigen::VectorXd last_actions;
    };

    auto build_sample = [&]() {
        ObservationSample sample;
        sample.motion_joint_position = ToVectorXd(rl_motion_q_);
        sample.motion_joint_velocity = ToVectorXd(rl_motion_q_dot_);
        sample.base_angular_velocity = ToVectorXd(rl_base_ang_vel_);
        sample.projected_gravity     = ToVectorXd(rl_projected_grav_);
        sample.joint_position        = ToVectorXd(rl_q_ - q_default_);
        sample.joint_velocity        = ToVectorXd(rl_q_dot_);
        sample.last_actions          = ToVectorXd(rl_last_actions_);
        return sample;
    };

    if (obs_update_first_) {
        const ObservationSample sample = build_sample();
        obs_history_.motion_joint_pos.init(rl_obs_history_len_, sample.motion_joint_position);
        obs_history_.motion_joint_vel.init(rl_obs_history_len_, sample.motion_joint_velocity);
        obs_history_.base_ang_vel.init(rl_obs_history_len_, sample.base_angular_velocity);
        obs_history_.projected_gravity.init(rl_obs_history_len_, sample.projected_gravity);
        obs_history_.joint_pos.init(rl_obs_history_len_, sample.joint_position);
        obs_history_.joint_vel.init(rl_obs_history_len_, sample.joint_velocity);
        obs_history_.actions.init(rl_obs_history_len_, sample.last_actions);
        obs_update_first_ = false;
    }

    const ObservationSample sample = build_sample();
    obs_history_.motion_joint_pos.push_back(sample.motion_joint_position);
    obs_history_.motion_joint_vel.push_back(sample.motion_joint_velocity);
    obs_history_.base_ang_vel.push_back(sample.base_angular_velocity);
    obs_history_.projected_gravity.push_back(sample.projected_gravity);
    obs_history_.joint_pos.push_back(sample.joint_position);
    obs_history_.joint_vel.push_back(sample.joint_velocity);
    obs_history_.actions.push_back(sample.last_actions);

    const int rc = updateFlattenedObservation();
    obs_updated_ = (rc == 0);
    return rc;
}

int InferenceModule::updateFlattenedObservation() {
    switch (observation_layout_) {
    case ObservationLayout::LegacyHistory:
        return updateLegacyFlattenedObservation();
    case ObservationLayout::ReferenceTracking:
        return updateReferenceTrackingFlattenedObservation();
    }
    return -1;
}

int InferenceModule::updateLegacyFlattenedObservation() {
    if (onnx_input_obs_index_ < 0 || onnx_input_states_buffer_.empty()) {
        return -1;
    }

    const Eigen::VectorXd motion_joint_position_vec = obs_history_.motion_joint_pos.toEigenVector();
    const Eigen::VectorXd motion_joint_velocity_vec = obs_history_.motion_joint_vel.toEigenVector();
    const Eigen::VectorXd base_ang_vel_vec          = obs_history_.base_ang_vel.toEigenVector();
    const Eigen::VectorXd projected_gravity_vec     = obs_history_.projected_gravity.toEigenVector();
    const Eigen::VectorXd joint_position_vec        = obs_history_.joint_pos.toEigenVector();
    const Eigen::VectorXd joint_velocity_vec        = obs_history_.joint_vel.toEigenVector();
    const Eigen::VectorXd actions_vec               = obs_history_.actions.toEigenVector();

    const Eigen::Index flat_size = motion_joint_position_vec.size() + motion_joint_velocity_vec.size() + base_ang_vel_vec.size() +
                                   projected_gravity_vec.size() + joint_position_vec.size() + joint_velocity_vec.size() + actions_vec.size();
    if (flat_size != static_cast<Eigen::Index>(rl_obs_input_size_)) {
        std::cerr << module_label_ << " observation size mismatch. expected=" << rl_obs_input_size_ << " actual=" << flat_size << "\n";
        return -1;
    }

    rl_obs_flat_.resize(static_cast<Eigen::Index>(rl_obs_input_size_));
    rl_obs_flat_ << motion_joint_position_vec, motion_joint_velocity_vec, base_ang_vel_vec, projected_gravity_vec, joint_position_vec,
        joint_velocity_vec, actions_vec;

    std::copy(rl_obs_flat_.data(), rl_obs_flat_.data() + static_cast<Eigen::Index>(rl_obs_input_size_),
              onnx_input_states_buffer_[onnx_input_obs_index_].begin());
    return 0;
}

int InferenceModule::updateReferenceTrackingFlattenedObservation() {
    if (onnx_input_obs_index_ < 0 || onnx_input_states_buffer_.empty()) {
        return -1;
    }

    const Eigen::VectorXd joint_position_vec    = obs_history_.joint_pos.toEigenVector();
    const Eigen::VectorXd joint_velocity_vec    = obs_history_.joint_vel.toEigenVector();
    const Eigen::VectorXd projected_gravity_vec = obs_history_.projected_gravity.toEigenVector();
    const Eigen::VectorXd base_ang_vel_vec      = obs_history_.base_ang_vel.toEigenVector();
    const Eigen::VectorXd actions_vec           = obs_history_.actions.toEigenVector();

    rl_obs_flat_.setZero(static_cast<Eigen::Index>(rl_obs_input_size_));
    Eigen::Index offset = 0;
    auto append_vector = [&](const Eigen::VectorXd &value) {
        if ((offset + value.size()) > rl_obs_flat_.size()) {
            return false;
        }
        rl_obs_flat_.segment(offset, value.size()) = value;
        offset += value.size();
        return true;
    };

    if (!append_vector(joint_position_vec) || !append_vector(joint_velocity_vec) || !append_vector(projected_gravity_vec) ||
        !append_vector(base_ang_vel_vec) || !append_vector(actions_vec)) {
        return -1;
    }

    if (rl_reference_frame_stack_len_ == 0 || latest_motion_frame_stack_.size() < rl_reference_frame_stack_len_) {
        return -1;
    }

    auto append_reference_term = [&](const auto &getter) {
        for (std::size_t frame_index = 0; frame_index < rl_reference_frame_stack_len_; ++frame_index) {
            if (!append_vector(getter(latest_motion_frame_stack_[frame_index]))) {
                return false;
            }
        }
        return true;
    };

    const auto joint_position = [this](const MotionFrame &frame) {
        Vector23d value;
        for (std::size_t i = 0; i < frame.joint_position.size(); ++i) {
            value(static_cast<Eigen::Index>(i)) = frame.joint_position[i];
        }
        if (config_.zero_reference_motion_ankle_position) {
            zeroAnkleObservationJoints(value);
        }
        return ToVectorXd(value);
    };
    const auto joint_velocity = [](const MotionFrame &frame) { return ToVectorXd(frame.joint_velocity); };
    const auto root_position_z = [](const MotionFrame &frame) {
        Eigen::VectorXd value(1);
        value(0) = frame.root_position_z;
        return value;
    };
    const auto root_state = [](const MotionFrame &frame) { return ToVectorXd(frame.root_state); };
    const auto body_position = [](const MotionFrame &frame) { return ToVectorXd(frame.body_position); };
    const auto root_linear_velocity = [](const MotionFrame &frame) { return ToVectorXd(frame.root_linear_velocity); };
    const auto root_angular_velocity = [](const MotionFrame &frame) { return ToVectorXd(frame.root_angular_velocity); };

    if (!append_reference_term(joint_position) || !append_reference_term(joint_velocity) || !append_reference_term(root_position_z) ||
        !append_reference_term(root_state) || !append_reference_term(body_position) || !append_reference_term(root_linear_velocity) ||
        !append_reference_term(root_angular_velocity)) {
        return -1;
    }

    if (offset != static_cast<Eigen::Index>(rl_obs_input_size_)) {
        std::cerr << module_label_ << " reference observation size mismatch. expected=" << rl_obs_input_size_ << " actual=" << offset << "\n";
        return -1;
    }

    std::copy(rl_obs_flat_.data(), rl_obs_flat_.data() + static_cast<Eigen::Index>(rl_obs_input_size_),
              onnx_input_states_buffer_[onnx_input_obs_index_].begin());
    return 0;
}

InferenceModule::ObservationLayout InferenceModule::detectObservationLayout(std::size_t obs_size) {
    if (obs_size >= static_cast<std::size_t>(kRlNumObsCurrent) && (obs_size % static_cast<std::size_t>(kRlNumObsCurrent)) == 0) {
        rl_obs_history_len_              = obs_size / static_cast<std::size_t>(kRlNumObsCurrent);
        rl_reference_frame_stack_len_    = kRlReferenceFrameStackLength;
        return ObservationLayout::LegacyHistory;
    }

    if (obs_size == static_cast<std::size_t>(kRlNumReferenceTrackingObs)) {
        rl_obs_history_len_           = kRlReferenceObsHistoryLen;
        rl_reference_frame_stack_len_ = kRlReferenceFrameStackLength;
        return ObservationLayout::ReferenceTracking;
    }

    rl_obs_history_len_           = kRlObsHistoryLen;
    rl_reference_frame_stack_len_ = kRlReferenceFrameStackLength;
    throw std::invalid_argument("unsupported observation size");
}

Vector23d InferenceModule::composeTargetJointPositions() const {
    if (config_.use_motion_residual_action) {
        return q_default_ + rl_motion_q_ + rl_actions_.cwiseProduct(action_scale_);
    }
    return q_default_ + rl_actions_.cwiseProduct(action_scale_);
}

void InferenceModule::updateVelocityFilters(const igris_c::msg::dds::LowState &state) {
    VectorQd joint_velocity_raw = VectorQd::Zero();
    for (int i = 0; i < kModelDof; ++i) {
        joint_velocity_raw(i) = static_cast<double>(state.joint_state()[static_cast<std::size_t>(i)].dq());
    }

    Eigen::Vector3d imu_gyro_raw;
    for (int i = 0; i < 3; ++i) {
        imu_gyro_raw(i) = static_cast<double>(state.imu_state().gyroscope()[static_cast<std::size_t>(i)]);
    }

    if (!velocity_lpf_initialized_) {
        joint_velocity_lpf_       = joint_velocity_raw;
        imu_angular_vel_lpf_      = imu_gyro_raw;
        velocity_lpf_initialized_ = true;
        return;
    }

    joint_velocity_lpf_ = config_.q_dot_lpf_cutoff_hz > 0.0
                              ? LowPass<kModelDof>(joint_velocity_raw, joint_velocity_lpf_, static_cast<double>(config_.control_hz),
                                                   config_.q_dot_lpf_cutoff_hz)
                              : joint_velocity_raw;
    imu_angular_vel_lpf_ = config_.imu_ang_vel_lpf_cutoff_hz > 0.0
                               ? LowPass<3>(imu_gyro_raw, imu_angular_vel_lpf_, static_cast<double>(config_.control_hz),
                                            config_.imu_ang_vel_lpf_cutoff_hz)
                               : imu_gyro_raw;
}

void InferenceModule::processTargetPositions(const igris_c::msg::dds::LowState &state, VectorQd &desired_position) const {
    for (std::size_t i = 0; i < kObsToSystemJointMapping.size(); ++i) {
        const int joint_index = kObsToSystemJointMapping[i];
        if (isParallelJoint(joint_index)) {
            continue;
        }

        const double hard_limit_low  = joint_pos_limit_low_(static_cast<Eigen::Index>(i));
        const double hard_limit_high = joint_pos_limit_high_(static_cast<Eigen::Index>(i));
        const double mid_point       = 0.5 * (hard_limit_low + hard_limit_high);
        const double radius          = hard_limit_high - hard_limit_low;
        const double soft_limit_low  = mid_point - 0.5 * radius * config_.non_parallel_safety_coef;
        const double soft_limit_high = mid_point + 0.5 * radius * config_.non_parallel_safety_coef;

        double &target        = desired_position(joint_index);
        const double joint_q  = static_cast<double>(state.joint_state()[static_cast<std::size_t>(joint_index)].q());

        if (joint_q > soft_limit_high && target > hard_limit_high) {
            target -= std::clamp((joint_q - soft_limit_high) / std::max(1.0e-6, hard_limit_high - soft_limit_high), 0.0, 1.0) *
                      (target - hard_limit_high);
        } else if (joint_q < soft_limit_low && target < hard_limit_low) {
            target += std::clamp((soft_limit_low - joint_q) / std::max(1.0e-6, soft_limit_low - hard_limit_low), 0.0, 1.0) *
                      (hard_limit_low - target);
        }
    }
}

bool InferenceModule::isParallelJoint(int joint_index) {
    return joint_index == TM1_WR || joint_index == TM1_WP || joint_index == TM1_LAP || joint_index == TM1_LAR || joint_index == TM1_RAP ||
           joint_index == TM1_RAR;
}

bool InferenceModule::isAnkleObservationJoint(int joint_index) {
    return joint_index == TM1_LAP || joint_index == TM1_LAR || joint_index == TM1_RAP || joint_index == TM1_RAR;
}

void InferenceModule::zeroAnkleObservationJoints(Vector23d &joint_position) {
    for (std::size_t i = 0; i < kObsToSystemJointMapping.size(); ++i) {
        if (isAnkleObservationJoint(kObsToSystemJointMapping[i])) {
            joint_position(static_cast<Eigen::Index>(i)) = 0.0;
        }
    }
}

}  // namespace public_inference_module
