#pragma once

#include "public_inference_module/config.hpp"
#include "public_inference_module/motion_data.hpp"
#include "public_inference_module/rolling_history.hpp"
#include "public_inference_module/types.hpp"

#include "onnxruntime_cxx_api.h"

#include <Eigen/Dense>

#include <array>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace public_inference_module {

class InferenceModule {
  public:
    explicit InferenceModule(std::string module_label = "PublicInferenceModule");
    ~InferenceModule();

    int initialize();
    int loadConfig(const InferenceConfig &config);
    int reset(const igris_c::msg::dds::LowState &state, const MotionDataSample *motion_data_sample);
    int compute(const igris_c::msg::dds::LowState &state, const MotionDataSample *motion_data_sample, InferenceCommand &command);
    int runPolicyStep(const igris_c::msg::dds::LowState &state, const MotionDataSample *motion_data_sample,
                      InferenceCommand &command);

    void buildHoldCommand(const igris_c::msg::dds::LowState &state, InferenceCommand &command) const;
    std::shared_ptr<MotionDataBuffer> getMotionDataBuffer() const { return motion_data_buffer_; }
    std::shared_ptr<const MotionDataSample> readMotionDataSample() const {
        return motion_data_buffer_ ? motion_data_buffer_->readLatest() : nullptr;
    }
    double lastPolicyInferenceMs() const { return last_policy_inference_ms_; }
    uint64_t policyInferenceCount() const { return policy_inference_count_; }
    void writePlotFiles(const std::string &root_dir) const;

  private:
    struct ObservationHistory {
        RollingHistory motion_joint_pos;
        RollingHistory motion_joint_vel;
        RollingHistory motion_body_pos;
        RollingHistory motion_anchor_lin_vel;
        RollingHistory motion_anchor_ang_vel;
        RollingHistory motion_anchor_ori;
        RollingHistory motion_anchor_height;
        RollingHistory base_ang_vel;
        RollingHistory projected_gravity;
        RollingHistory joint_pos;
        RollingHistory joint_vel;
        RollingHistory actions;
    };

    enum class ObservationLayout {
        LegacyHistory,
        ReferenceTracking,
        GeneralMotionTracking,
    };

    struct PlotSample {
        double time_sec = 0.0;
        uint64_t policy_step = 0;
        Vector23d desired_joint_position = Vector23d::Zero();
        Vector23d joint_pos_rel = Vector23d::Zero();
        Vector23d joint_vel = Vector23d::Zero();
        Eigen::Vector3d base_ang_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d projected_gravity = Eigen::Vector3d::Zero();
        Vector23d motion_joint_pos = Vector23d::Zero();
        double motion_anchor_height = 0.0;
    };

    void initializeRlState();
    void resetPlotRecorder();
    void recordPlotSample(const Vector23d &desired_joint_position);
    void writeSvgPlot(const std::filesystem::path &path, const std::string &title, const std::string &y_label,
                      const std::function<double(const PlotSample &)> &value_getter) const;
    void writeVector23Plots(const std::filesystem::path &directory, const std::string &title_prefix, const std::string &y_label,
                            const std::function<double(const PlotSample &, Eigen::Index)> &value_getter) const;
    void resetPolicyState();
    int computePolicy();
    bool buildMotionFrameFromMotionData(const MotionDataSample &motion_data_sample, MotionFrame &motion_frame) const;
    bool buildMotionFrameStackFromMotionData(const MotionDataSample &motion_data_sample, std::vector<MotionFrame> &motion_frames) const;
    int updateObservationBuffer(const igris_c::msg::dds::LowState &state, const MotionDataSample *motion_data_sample);
    int updateFlattenedObservation();
    int updateLegacyFlattenedObservation();
    int updateReferenceTrackingFlattenedObservation();
    int updateGeneralMotionTrackingFlattenedObservation();
    ObservationLayout detectObservationLayout(std::size_t obs_size);
    Vector23d composeTargetJointPositions() const;
    void updateVelocityFilters(const igris_c::msg::dds::LowState &state);
    void processTargetPositions(const igris_c::msg::dds::LowState &state, VectorQd &desired_position) const;
    static bool isParallelJoint(int joint_index);

  private:
    std::string module_label_;
    InferenceConfig config_;
    int policy_decimation_ = 6;
    uint64_t control_tick_ = 0;
    uint64_t policy_inference_count_ = 0;
    double last_policy_inference_ms_ = 0.0;
    bool obs_update_first_ = true;
    bool obs_updated_      = false;
    bool velocity_lpf_initialized_ = false;
    bool config_loaded_            = false;

    ObservationHistory obs_history_;
    Eigen::VectorXd rl_obs_flat_;
    ObservationLayout observation_layout_ = ObservationLayout::LegacyHistory;
    std::size_t rl_obs_input_size_        = kRlNumObsHistory;
    std::size_t rl_obs_history_len_       = kRlObsHistoryLen;
    std::size_t rl_reference_frame_stack_len_ = kRlReferenceFrameStackLength;

    Vector23d q_default_               = Vector23d::Zero();
    Vector23d rl_q_                    = Vector23d::Zero();
    Vector23d rl_q_dot_                = Vector23d::Zero();
    Vector23d rl_motion_q_             = Vector23d::Zero();
    Vector23d rl_motion_q_dot_         = Vector23d::Zero();
    Eigen::Vector3d rl_base_ang_vel_   = Eigen::Vector3d::Zero();
    Eigen::Vector3d rl_projected_grav_ = Eigen::Vector3d::Zero();
    Vector23d rl_actions_              = Vector23d::Zero();
    Vector23d rl_last_actions_         = Vector23d::Zero();
    Vector23d joint_pos_limit_high_    = Vector23d::Zero();
    Vector23d joint_pos_limit_low_     = Vector23d::Zero();
    Vector23d action_scale_            = Vector23d::Constant(0.25);

    VectorQd command_kp_ = VectorQd::Zero();
    VectorQd command_kd_ = VectorQd::Zero();

    VectorQd joint_velocity_lpf_         = VectorQd::Zero();
    Eigen::Vector3d imu_angular_vel_lpf_ = Eigen::Vector3d::Zero();

    MotionFrame latest_motion_frame_;
    std::vector<MotionFrame> latest_motion_frame_stack_;
    std::vector<PlotSample> plot_samples_;
    std::chrono::steady_clock::time_point plot_start_time_{};
    bool plot_start_time_valid_ = false;

    size_t onnx_input_number_  = 0;
    size_t onnx_output_number_ = 0;
    std::vector<std::string> onnx_input_names_;
    std::vector<std::string> onnx_output_names_;
    std::vector<const char *> onnx_input_names_char_;
    std::vector<const char *> onnx_output_names_char_;
    std::vector<Ort::Value> onnx_input_tensors_;
    std::vector<Ort::Value> onnx_output_tensors_;
    std::vector<std::vector<float>> onnx_input_states_buffer_;
    int onnx_input_obs_index_      = -1;
    int onnx_output_actions_index_ = -1;

    mutable std::mutex mutex_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "public_inference_module"};
    Ort::SessionOptions session_options_{};
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::shared_ptr<MotionDataBuffer> motion_data_buffer_;
};

}  // namespace public_inference_module
