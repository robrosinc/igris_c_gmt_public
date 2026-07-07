#include "public_inference_module/motion_frame_source.hpp"

#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace public_inference_module {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<const char *, kRlReferenceKeybodyCount> kReferenceTrackingGoalBodyNames = {
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
    "Link_Neck_Pitch",
};

std::vector<std::string> SplitCommaSeparated(const std::string &value) {
    std::vector<std::string> values;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        const auto begin = std::find_if_not(token.begin(), token.end(), [](unsigned char ch) { return std::isspace(ch) != 0; });
        const auto end = std::find_if_not(token.rbegin(), token.rend(), [](unsigned char ch) { return std::isspace(ch) != 0; }).base();
        if (begin < end) {
            values.emplace_back(begin, end);
        }
    }
    return values;
}

std::string LookupMetadataString(const Ort::ModelMetadata &metadata, Ort::AllocatorWithDefaultOptions &allocator, const char *key) {
    Ort::AllocatedStringPtr value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
    return value ? std::string(value.get()) : std::string();
}

double ParsePositiveDouble(const std::string &value, const std::string &label) {
    char *end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0' || parsed <= 0.0) {
        throw std::runtime_error("Invalid " + label + " metadata: " + value);
    }
    return parsed;
}

std::size_t ParsePositiveSize(const std::string &value, const std::string &label) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed == 0) {
        throw std::runtime_error("Invalid " + label + " metadata: " + value);
    }
    return static_cast<std::size_t>(parsed);
}

std::size_t TensorElementCount(const Ort::Value &value) {
    return value.GetTensorTypeAndShapeInfo().GetElementCount();
}

void RequireFloatOutput(const Ort::TypeInfo &type_info, const std::string &name) {
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("ONNX motion output '" + name + "' must be float");
    }
}

class NullMotionFrameSource : public MotionFrameSource {
  public:
    bool getLatest(MotionFrame &frame) override {
        frame = MotionFrame{};
        return false;
    }

    void reset() override {}
};

std::vector<double> ParseCsvLine(const std::string &line) {
    std::vector<double> values;
    std::stringstream stream(line);
    std::string token;
    while (std::getline(stream, token, ',')) {
        char *end   = nullptr;
        const auto value = std::strtod(token.c_str(), &end);
        if (end == token.c_str()) {
            return {};
        }
        values.push_back(value);
    }
    return values;
}

class CsvReplayMotionFrameSource : public MotionFrameSource {
  public:
    explicit CsvReplayMotionFrameSource(const MotionSourceConfig &config) : config_(config) {
        if (config_.csv_path.empty()) {
            throw std::runtime_error("motion_source.csv_path must be set when motion_source.type=csv_replay");
        }
        if (config_.fps <= 0.0) {
            throw std::runtime_error("motion_source.fps must be positive");
        }

        std::ifstream file(config_.csv_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open motion CSV: " + config_.csv_path);
        }

        std::string line;
        while (std::getline(file, line)) {
            const auto first_non_space = std::find_if_not(line.begin(), line.end(), [](unsigned char ch) { return std::isspace(ch) != 0; });
            if (first_non_space == line.end() || *first_non_space == '#') {
                continue;
            }

            const std::vector<double> values = ParseCsvLine(line);
            if (values.empty()) {
                continue;
            }
            if (values.size() != 52 && values.size() != 103) {
                throw std::runtime_error("Motion CSV row must have 52 legacy columns or 103 reference-tracking columns: " + line);
            }

            MotionFrame frame;
            frame.seq      = static_cast<uint64_t>(values[0]);
            frame.stamp_ns = static_cast<uint64_t>(values[1]);
            frame.valid    = true;

            std::size_t offset = 2;
            for (std::size_t i = 0; i < frame.joint_position.size(); ++i) {
                frame.joint_position[i] = values[offset + i];
            }
            offset += frame.joint_position.size();
            for (std::size_t i = 0; i < frame.joint_velocity.size(); ++i) {
                frame.joint_velocity[i] = values[offset + i];
            }
            offset += frame.joint_velocity.size();
            if (values.size() == 52) {
                for (std::size_t i = 0; i < frame.anchor_quaternion_wxyz.size(); ++i) {
                    frame.anchor_quaternion_wxyz[i] = values[offset + i];
                }
            } else {
                frame.root_position_z = values[offset++];
                for (std::size_t i = 0; i < frame.root_state.size(); ++i) {
                    frame.root_state[i] = values[offset + i];
                }
                offset += frame.root_state.size();
                for (std::size_t i = 0; i < frame.body_position.size(); ++i) {
                    frame.body_position[i] = values[offset + i];
                }
                offset += frame.body_position.size();
                for (std::size_t i = 0; i < frame.root_linear_velocity.size(); ++i) {
                    frame.root_linear_velocity[i] = values[offset + i];
                }
                offset += frame.root_linear_velocity.size();
                for (std::size_t i = 0; i < frame.root_angular_velocity.size(); ++i) {
                    frame.root_angular_velocity[i] = values[offset + i];
                }
            }
            frames_.push_back(frame);
        }

        if (frames_.empty()) {
            throw std::runtime_error("Motion CSV contains no valid frames: " + config_.csv_path);
        }

        reset();
    }

    bool getLatest(MotionFrame &frame) override {
        if (frames_.empty()) {
            return false;
        }

        const auto now         = std::chrono::steady_clock::now();
        const auto elapsed_sec = std::chrono::duration<double>(now - start_time_).count();
        const auto elapsed_frames =
            static_cast<std::size_t>(std::max(0.0, elapsed_sec * config_.fps));

        std::size_t index = elapsed_frames;
        if (config_.loop) {
            index %= frames_.size();
        } else if (index >= frames_.size()) {
            index = frames_.size() - 1;
        }

        frame = frames_[index];
        return true;
    }

    void reset() override { start_time_ = std::chrono::steady_clock::now(); }

  private:
    MotionSourceConfig config_;
    std::vector<MotionFrame> frames_;
    std::chrono::steady_clock::time_point start_time_ = std::chrono::steady_clock::now();
};

class OnnxReplayMotionFrameSource : public MotionFrameSource {
  public:
    explicit OnnxReplayMotionFrameSource(const MotionSourceConfig &config)
        : config_(config), env_(ORT_LOGGING_LEVEL_WARNING, "public_inference_motion_source") {
        if (config_.layout != "reference_tracking_v1") {
            throw std::runtime_error("motion_source.type=onnx_replay requires motion_source.layout=reference_tracking_v1");
        }
        if (config_.onnx_path.empty()) {
            throw std::runtime_error("motion_source.onnx_path must be set when motion_source.type=onnx_replay");
        }
        if (!std::filesystem::exists(config_.onnx_path)) {
            throw std::runtime_error("Motion ONNX not found: " + config_.onnx_path);
        }

        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        session_options_.AddConfigEntry("session.use_deterministic_compute", "1");
        session_ = std::make_unique<Ort::Session>(env_, config_.onnx_path.c_str(), session_options_);

        loadTensorInfo();
        loadMetadata();
        reset();
    }

    bool getLatest(MotionFrame &frame) override {
        const auto now         = Clock::now();
        const auto elapsed_sec = std::chrono::duration<double>(now - start_time_).count();
        const auto elapsed_frames =
            static_cast<std::size_t>(std::max(0.0, elapsed_sec * fps_));

        std::size_t frame_index = elapsed_frames;
        if (config_.loop) {
            frame_index %= num_frames_;
        } else if (frame_index >= num_frames_) {
            frame_index = num_frames_ - 1;
        }

        if (!cached_frame_valid_ || frame_index != cached_frame_index_) {
            cached_frame_       = runFrame(frame_index);
            cached_frame_index_ = frame_index;
            cached_frame_valid_ = true;
        }

        frame          = cached_frame_;
        frame.seq      = elapsed_frames;
        frame.stamp_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time_).count());
        return true;
    }

    void reset() override {
        start_time_         = Clock::now();
        cached_frame_valid_ = false;
        cached_frame_index_ = 0;
    }

  private:
    void loadTensorInfo() {
        if (session_->GetInputCount() != 1) {
            throw std::runtime_error("Motion ONNX must have exactly one input tensor");
        }

        Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(0, allocator_);
        input_names_.push_back(input_name.get());
        input_name_ptrs_.push_back(input_names_.back().c_str());
        if (input_names_.front() != "time_step") {
            throw std::runtime_error("Motion ONNX input must be named 'time_step'");
        }

        const Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
        const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
        if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            throw std::runtime_error("Motion ONNX input 'time_step' must be int64");
        }

        output_names_.resize(session_->GetOutputCount());
        output_name_ptrs_.resize(session_->GetOutputCount());
        for (std::size_t i = 0; i < session_->GetOutputCount(); ++i) {
            Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(i, allocator_);
            output_names_[i]                    = output_name.get();
            output_name_ptrs_[i]                = output_names_[i].c_str();
            output_indices_[output_names_[i]]   = i;
            RequireFloatOutput(session_->GetOutputTypeInfo(i), output_names_[i]);
        }

        for (const auto *name : {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_pos_b", "body_lin_vel_b", "body_ang_vel_b"}) {
            if (output_indices_.find(name) == output_indices_.end()) {
                throw std::runtime_error("Motion ONNX is missing required output '" + std::string(name) + "'");
            }
        }
    }

    void loadMetadata() {
        Ort::ModelMetadata metadata = session_->GetModelMetadata();
        const std::string fps_metadata = LookupMetadataString(metadata, allocator_, "motion_fps");
        fps_ = fps_metadata.empty() ? config_.fps : ParsePositiveDouble(fps_metadata, "motion_fps");
        if (fps_ <= 0.0) {
            throw std::runtime_error("motion_source.fps must be positive when motion_fps metadata is missing");
        }

        const std::string frame_count_metadata = LookupMetadataString(metadata, allocator_, "motion_num_frames");
        if (frame_count_metadata.empty()) {
            throw std::runtime_error("Motion ONNX metadata must include motion_num_frames");
        }
        num_frames_ = ParsePositiveSize(frame_count_metadata, "motion_num_frames");

        const std::vector<std::string> body_names = SplitCommaSeparated(LookupMetadataString(metadata, allocator_, "body_names"));
        if (body_names.empty()) {
            throw std::runtime_error("Motion ONNX metadata must include body_names");
        }

        std::unordered_map<std::string, std::size_t> body_name_to_index;
        for (std::size_t i = 0; i < body_names.size(); ++i) {
            body_name_to_index.emplace(body_names[i], i);
        }

        auto base_it = body_name_to_index.find("base_link");
        if (base_it == body_name_to_index.end()) {
            throw std::runtime_error("Motion ONNX body_names metadata must include base_link");
        }
        root_body_index_ = base_it->second;

        for (std::size_t i = 0; i < goal_body_indices_.size(); ++i) {
            const auto it = body_name_to_index.find(kReferenceTrackingGoalBodyNames[i]);
            if (it == body_name_to_index.end()) {
                throw std::runtime_error("Motion ONNX body_names metadata is missing " + std::string(kReferenceTrackingGoalBodyNames[i]));
            }
            goal_body_indices_[i] = it->second;
        }
    }

    MotionFrame runFrame(std::size_t frame_index) {
        std::vector<int64_t> time_step{static_cast<int64_t>(frame_index)};
        std::vector<int64_t> input_shape{1, 1};
        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info_, time_step.data(), time_step.size(), input_shape.data(),
                                                                     input_shape.size()));

        std::vector<Ort::Value> outputs =
            session_->Run(Ort::RunOptions{nullptr}, input_name_ptrs_.data(), input_tensors.data(), input_tensors.size(), output_name_ptrs_.data(),
                          output_name_ptrs_.size());

        MotionFrame frame;
        frame.valid = true;
        copyJointOutput(outputs, "joint_pos", frame.joint_position);
        copyJointOutput(outputs, "joint_vel", frame.joint_velocity);
        copyRootOutputs(outputs, frame);
        copyBodyPositions(outputs, frame);
        return frame;
    }

    const Ort::Value &outputByName(const std::vector<Ort::Value> &outputs, const std::string &name) const {
        return outputs[output_indices_.at(name)];
    }

    template <std::size_t N>
    void copyJointOutput(const std::vector<Ort::Value> &outputs, const std::string &name, std::array<double, N> &target) const {
        const Ort::Value &output = outputByName(outputs, name);
        if (TensorElementCount(output) < target.size()) {
            throw std::runtime_error("Motion ONNX output '" + name + "' is smaller than expected");
        }
        const float *data = output.GetTensorData<float>();
        for (std::size_t i = 0; i < target.size(); ++i) {
            target[i] = static_cast<double>(data[i]);
        }
    }

    void copyRootOutputs(const std::vector<Ort::Value> &outputs, MotionFrame &frame) const {
        const Ort::Value &body_pos_w     = outputByName(outputs, "body_pos_w");
        const Ort::Value &body_quat_w    = outputByName(outputs, "body_quat_w");
        const Ort::Value &body_lin_vel_b = outputByName(outputs, "body_lin_vel_b");
        const Ort::Value &body_ang_vel_b = outputByName(outputs, "body_ang_vel_b");

        const std::size_t root_pos_offset = root_body_index_ * 3;
        const std::size_t root_quat_offset = root_body_index_ * 4;
        if (TensorElementCount(body_pos_w) < root_pos_offset + 3 || TensorElementCount(body_quat_w) < root_quat_offset + 4 ||
            TensorElementCount(body_lin_vel_b) < root_pos_offset + 3 || TensorElementCount(body_ang_vel_b) < root_pos_offset + 3) {
            throw std::runtime_error("Motion ONNX root body outputs are smaller than expected");
        }

        const float *root_pos_data     = body_pos_w.GetTensorData<float>() + root_pos_offset;
        const float *root_quat_data    = body_quat_w.GetTensorData<float>() + root_quat_offset;
        const float *root_lin_vel_data = body_lin_vel_b.GetTensorData<float>() + root_pos_offset;
        const float *root_ang_vel_data = body_ang_vel_b.GetTensorData<float>() + root_pos_offset;

        frame.root_position_z = static_cast<double>(root_pos_data[2]);
        frame.anchor_quaternion_wxyz = {static_cast<double>(root_quat_data[0]), static_cast<double>(root_quat_data[1]),
                                        static_cast<double>(root_quat_data[2]), static_cast<double>(root_quat_data[3])};
        frame.anchor_quaternion_valid = true;
        frame.root_linear_velocity = {static_cast<double>(root_lin_vel_data[0]), static_cast<double>(root_lin_vel_data[1]),
                                      static_cast<double>(root_lin_vel_data[2])};
        frame.root_angular_velocity = {static_cast<double>(root_ang_vel_data[0]), static_cast<double>(root_ang_vel_data[1]),
                                       static_cast<double>(root_ang_vel_data[2])};
    }

    void copyBodyPositions(const std::vector<Ort::Value> &outputs, MotionFrame &frame) const {
        const Ort::Value &body_pos_b = outputByName(outputs, "body_pos_b");
        const float *data = body_pos_b.GetTensorData<float>();
        const std::size_t element_count = TensorElementCount(body_pos_b);

        for (std::size_t i = 0; i < goal_body_indices_.size(); ++i) {
            const std::size_t source_offset = goal_body_indices_[i] * 3;
            const std::size_t target_offset = i * 3;
            if (element_count < source_offset + 3) {
                throw std::runtime_error("Motion ONNX body_pos_b output is smaller than expected");
            }
            frame.body_position[target_offset]     = static_cast<double>(data[source_offset]);
            frame.body_position[target_offset + 1] = static_cast<double>(data[source_offset + 1]);
            frame.body_position[target_offset + 2] = static_cast<double>(data[source_offset + 2]);
        }
    }

    MotionSourceConfig config_;
    double fps_             = 50.0;
    std::size_t num_frames_ = 0;
    std::size_t root_body_index_ = 0;
    std::array<std::size_t, kRlReferenceKeybodyCount> goal_body_indices_{};
    Clock::time_point start_time_ = Clock::now();
    bool cached_frame_valid_      = false;
    std::size_t cached_frame_index_ = 0;
    MotionFrame cached_frame_;

    Ort::Env env_;
    Ort::SessionOptions session_options_{};
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char *> input_name_ptrs_;
    std::vector<const char *> output_name_ptrs_;
    std::unordered_map<std::string, std::size_t> output_indices_;
};

}  // namespace

std::unique_ptr<MotionFrameSource> CreateMotionFrameSource(const InferenceConfig &config) {
    if (config.motion_source.type == "null" || config.motion_source.type == "hold") {
        return std::make_unique<NullMotionFrameSource>();
    }
    if (config.motion_source.type == "csv_replay") {
        return std::make_unique<CsvReplayMotionFrameSource>(config.motion_source);
    }
    if (config.motion_source.type == "onnx_replay") {
        return std::make_unique<OnnxReplayMotionFrameSource>(config.motion_source);
    }

    throw std::runtime_error("Unsupported motion_source.type: " + config.motion_source.type);
}

}  // namespace public_inference_module
