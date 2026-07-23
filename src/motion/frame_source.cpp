#include "motion/frame_source.hpp"

#include "onnxruntime_cxx_api.h"
#include "utils/obs_functions.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace igris_c_gmt_public {
bool MotionFrameSource::getLatestStack(std::vector<MotionFrame> &frames,
                                       std::size_t length) {
  frames.clear();
  MotionFrame frame;
  if (length == 0 || !getLatest(frame)) {
    return false;
  }
  frames.assign(length, frame);
  return true;
}

bool MotionFrameSource::getFrameAtStep(std::size_t step, MotionFrame &frame) {
  (void)step;
  return getLatest(frame);
}

bool MotionFrameSource::getFrameStackAtStep(std::size_t step,
                                            std::vector<MotionFrame> &frames,
                                            std::size_t length) {
  frames.clear();
  MotionFrame frame;
  if (length == 0 || !getFrameAtStep(step, frame)) {
    return false;
  }

  frames.assign(length, frame);
  return true;
}

namespace {

using Clock = std::chrono::steady_clock;

std::vector<std::string> SplitCommaSeparated(const std::string &value) {
  std::vector<std::string> values;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const auto begin =
        std::find_if_not(token.begin(), token.end(), [](unsigned char ch) {
          return std::isspace(ch) != 0;
        });
    const auto end =
        std::find_if_not(token.rbegin(), token.rend(), [](unsigned char ch) {
          return std::isspace(ch) != 0;
        }).base();
    if (begin < end) {
      values.emplace_back(begin, end);
    }
  }
  return values;
}

std::string LookupMetadataString(const Ort::ModelMetadata &metadata,
                                 Ort::AllocatorWithDefaultOptions &allocator,
                                 const char *key) {
  Ort::AllocatedStringPtr value =
      metadata.LookupCustomMetadataMapAllocated(key, allocator);
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

std::size_t ParsePositiveSize(const std::string &value,
                              const std::string &label) {
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

void RequireFloatOutput(const Ort::TypeInfo &type_info,
                        const std::string &name) {
  const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("ONNX motion output '" + name + "' must be float");
  }
}

Eigen::Matrix3d RotationMatrixFromWxyz(const std::array<double, 4> &quat_wxyz) {
  Eigen::Quaterniond quaternion(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2],
                                quat_wxyz[3]);
  quaternion.normalize();
  return quaternion.toRotationMatrix();
}

std::array<double, 3>
RotateAnchorVectorToWorldFrame(const std::array<double, 4> &quat_wxyz,
                               const std::array<double, 3> &values) {
  const Eigen::Vector3d local_value(values[0], values[1], values[2]);
  const Eigen::Vector3d world_value =
      RotationMatrixFromWxyz(quat_wxyz) * local_value;
  return {world_value(0), world_value(1), world_value(2)};
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
    char *end = nullptr;
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
  explicit CsvReplayMotionFrameSource(const MotionSourceConfig &config)
      : config_(config) {
    if (config_.csv_path.empty()) {
      throw std::runtime_error("motion_source.csv_path must be set when "
                               "motion_source.type=csv_replay");
    }
    if (config_.fps <= 0.0) {
      throw std::runtime_error("motion_source.fps must be positive");
    }

    std::ifstream file(config_.csv_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open motion CSV: " +
                               config_.csv_path);
    }

    std::string line;
    while (std::getline(file, line)) {
      const auto first_non_space =
          std::find_if_not(line.begin(), line.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
          });
      if (first_non_space == line.end() || *first_non_space == '#') {
        continue;
      }

      const std::vector<double> values = ParseCsvLine(line);
      if (values.empty()) {
        continue;
      }
      const std::size_t base_csv_values =
          2 + static_cast<std::size_t>(kRlNumJointActions) +
          obs_functions::kMotionBodyPositionObsSize;
      const std::size_t csv_tail_with_height = 3 + 3 + 1 + 4;
      const std::size_t csv_tail_with_anchor_pos = 3 + 3 + 3 + 4;
      const std::size_t csv_body_orientation_values =
          obs_functions::kMotionBodyOrientationObsSize;
      const std::size_t remaining_values = values.size() >= base_csv_values
                                               ? values.size() - base_csv_values
                                               : 0;
      const bool has_body_orientation =
          remaining_values ==
              csv_body_orientation_values + csv_tail_with_height ||
          remaining_values ==
              csv_body_orientation_values + csv_tail_with_anchor_pos;
      const std::size_t tail_values =
          has_body_orientation ? remaining_values - csv_body_orientation_values
                               : remaining_values;
      const bool has_anchor_pos = tail_values == csv_tail_with_anchor_pos;
      const bool has_anchor_height = tail_values == csv_tail_with_height;
      if (values.size() < base_csv_values ||
          (!has_anchor_pos && !has_anchor_height)) {
        throw std::runtime_error("Motion CSV row must have current "
                                 "general-motion-tracking columns: " +
                                 line);
      }

      MotionFrame frame;
      frame.seq = static_cast<uint64_t>(values[0]);
      frame.valid = true;

      std::size_t offset = 2;
      for (std::size_t i = 0; i < frame.joint_position.size(); ++i) {
        frame.joint_position[i] = values[offset + i];
      }
      offset += frame.joint_position.size();

      frame.body_position.assign(
          values.begin() + static_cast<std::ptrdiff_t>(offset),
          values.begin() +
              static_cast<std::ptrdiff_t>(
                  offset + obs_functions::kMotionBodyPositionObsSize));
      frame.body_names.assign(obs_functions::kMotionBodyNames.begin(),
                              obs_functions::kMotionBodyNames.end());
      offset += obs_functions::kMotionBodyPositionObsSize;
      if (has_body_orientation) {
        frame.body_orientation.assign(
            values.begin() + static_cast<std::ptrdiff_t>(offset),
            values.begin() +
                static_cast<std::ptrdiff_t>(
                    offset + obs_functions::kMotionBodyOrientationObsSize));
        offset += obs_functions::kMotionBodyOrientationObsSize;
      }
      for (std::size_t i = 0; i < frame.anchor_linear_velocity_b.size(); ++i) {
        frame.anchor_linear_velocity_b[i] = values[offset + i];
      }
      offset += frame.anchor_linear_velocity_b.size();
      for (std::size_t i = 0; i < frame.anchor_angular_velocity_b.size(); ++i) {
        frame.anchor_angular_velocity_b[i] = values[offset + i];
      }
      offset += frame.anchor_angular_velocity_b.size();
      if (has_anchor_pos) {
        for (std::size_t i = 0; i < frame.anchor_position.size(); ++i) {
          frame.anchor_position[i] = values[offset + i];
        }
        offset += frame.anchor_position.size();
      } else {
        frame.anchor_position = {0.0, 0.0, values[offset++]};
      }
      for (std::size_t i = 0; i < frame.anchor_quaternion_wxyz.size(); ++i) {
        frame.anchor_quaternion_wxyz[i] = values[offset + i];
      }
      frame.anchor_quaternion_valid = true;
      frame.anchor_linear_velocity = RotateAnchorVectorToWorldFrame(
          frame.anchor_quaternion_wxyz, frame.anchor_linear_velocity_b);
      frame.anchor_angular_velocity = RotateAnchorVectorToWorldFrame(
          frame.anchor_quaternion_wxyz, frame.anchor_angular_velocity_b);
      frames_.push_back(frame);
    }

    if (frames_.empty()) {
      throw std::runtime_error("Motion CSV contains no valid frames: " +
                               config_.csv_path);
    }

    reset();
  }

  bool getLatest(MotionFrame &frame) override {
    if (frames_.empty()) {
      return false;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed_sec =
        std::chrono::duration<double>(now - start_time_).count();
    const auto elapsed_frames =
        static_cast<std::size_t>(std::max(0.0, elapsed_sec * config_.fps));

    const std::size_t index = frameIndexForStep(elapsed_frames);
    frame = frames_[index];
    return true;
  }

  bool getFrameAtStep(std::size_t step, MotionFrame &frame) override {
    if (frames_.empty()) {
      return false;
    }

    frame = frames_[frameIndexForStep(step)];
    frame.seq = static_cast<uint64_t>(step);
    return true;
  }

  bool getFrameStackAtStep(std::size_t step, std::vector<MotionFrame> &frames,
                           std::size_t length) override {
    frames.clear();
    if (length == 0 || frames_.empty()) {
      return false;
    }

    frames.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
      MotionFrame frame;
      if (!getFrameAtStep(step + i, frame)) {
        return false;
      }
      frames.push_back(frame);
    }
    return true;
  }

  void reset() override { start_time_ = std::chrono::steady_clock::now(); }

private:
  std::size_t frameIndexForStep(std::size_t step) const {
    if (config_.loop) {
      return step % frames_.size();
    }
    return std::min(step, frames_.size() - 1);
  }

  MotionSourceConfig config_;
  std::vector<MotionFrame> frames_;
  std::chrono::steady_clock::time_point start_time_ =
      std::chrono::steady_clock::now();
};

class OnnxReplayMotionFrameSource : public MotionFrameSource {
public:
  explicit OnnxReplayMotionFrameSource(const MotionSourceConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_WARNING, "public_inference_motion_source") {
    if (config_.onnx_path.empty()) {
      throw std::runtime_error("motion_source.onnx_path must be set when "
                               "motion_source.type=onnx_replay");
    }
    if (!std::filesystem::exists(config_.onnx_path)) {
      throw std::runtime_error("Motion ONNX not found: " + config_.onnx_path);
    }

    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options_.AddConfigEntry("session.use_deterministic_compute", "1");
    session_ = std::make_unique<Ort::Session>(env_, config_.onnx_path.c_str(),
                                              session_options_);

    loadTensorInfo();
    loadMetadata();
    reset();
  }

  bool getLatest(MotionFrame &frame) override {
    const auto now = Clock::now();
    const auto elapsed_sec =
        std::chrono::duration<double>(now - start_time_).count();
    const auto elapsed_frames =
        static_cast<std::size_t>(std::max(0.0, elapsed_sec * fps_));

    const std::size_t frame_index = frameIndexForStep(elapsed_frames);

    if (!cached_frame_valid_ || frame_index != cached_frame_index_) {
      cached_frame_ = runFrame(frame_index);
      cached_frame_index_ = frame_index;
      cached_frame_valid_ = true;
    }

    frame = cached_frame_;
    frame.seq = elapsed_frames;
    return true;
  }

  bool getLatestStack(std::vector<MotionFrame> &frames,
                      std::size_t length) override {
    frames.clear();
    if (length == 0) {
      return false;
    }

    const auto now = Clock::now();
    const auto elapsed_sec =
        std::chrono::duration<double>(now - start_time_).count();
    const auto elapsed_frames =
        static_cast<std::size_t>(std::max(0.0, elapsed_sec * fps_));

    frames.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
      const std::size_t step = elapsed_frames + i;
      const std::size_t frame_index = frameIndexForStep(step);

      MotionFrame frame = runFrame(frame_index);
      frame.seq = step;
      frames.push_back(frame);
    }
    return true;
  }

  bool getFrameAtStep(std::size_t step, MotionFrame &frame) override {
    const std::size_t frame_index = frameIndexForStep(step);
    if (!cached_frame_valid_ || frame_index != cached_frame_index_) {
      cached_frame_ = runFrame(frame_index);
      cached_frame_index_ = frame_index;
      cached_frame_valid_ = true;
    }

    frame = cached_frame_;
    frame.seq = static_cast<uint64_t>(step);
    return true;
  }

  bool getFrameStackAtStep(std::size_t step, std::vector<MotionFrame> &frames,
                           std::size_t length) override {
    frames.clear();
    if (length == 0) {
      return false;
    }

    frames.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
      const std::size_t frame_step = step + i;
      MotionFrame frame = runFrame(frameIndexForStep(frame_step));
      frame.seq = static_cast<uint64_t>(frame_step);
      frames.push_back(frame);
    }
    return true;
  }

  void reset() override {
    start_time_ = Clock::now();
    cached_frame_valid_ = false;
    cached_frame_index_ = 0;
  }

private:
  std::size_t frameIndexForStep(std::size_t step) const {
    if (config_.loop) {
      return step % num_frames_;
    }
    return std::min(step, num_frames_ - 1);
  }

  void loadTensorInfo() {
    if (session_->GetInputCount() != 1) {
      throw std::runtime_error(
          "Motion ONNX must have exactly one input tensor");
    }

    Ort::AllocatedStringPtr input_name =
        session_->GetInputNameAllocated(0, allocator_);
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
      Ort::AllocatedStringPtr output_name =
          session_->GetOutputNameAllocated(i, allocator_);
      output_names_[i] = output_name.get();
      output_name_ptrs_[i] = output_names_[i].c_str();
      output_indices_[output_names_[i]] = i;
      RequireFloatOutput(session_->GetOutputTypeInfo(i), output_names_[i]);
    }

    for (const auto *name :
         {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_pos_b",
          "body_lin_vel_b", "body_ang_vel_b"}) {
      if (output_indices_.find(name) == output_indices_.end()) {
        throw std::runtime_error("Motion ONNX is missing required output '" +
                                 std::string(name) + "'");
      }
    }
  }

  void loadMetadata() {
    Ort::ModelMetadata metadata = session_->GetModelMetadata();
    const std::string fps_metadata =
        LookupMetadataString(metadata, allocator_, "motion_fps");
    fps_ = fps_metadata.empty()
               ? config_.fps
               : ParsePositiveDouble(fps_metadata, "motion_fps");
    if (fps_ <= 0.0) {
      throw std::runtime_error("motion_source.fps must be positive when "
                               "motion_fps metadata is missing");
    }

    const std::string frame_count_metadata =
        LookupMetadataString(metadata, allocator_, "motion_num_frames");
    if (frame_count_metadata.empty()) {
      throw std::runtime_error(
          "Motion ONNX metadata must include motion_num_frames");
    }
    num_frames_ = ParsePositiveSize(frame_count_metadata, "motion_num_frames");

    body_names_ = SplitCommaSeparated(
        LookupMetadataString(metadata, allocator_, "body_names"));
    if (body_names_.empty()) {
      throw std::runtime_error("Motion ONNX metadata must include body_names");
    }

    std::unordered_map<std::string, std::size_t> body_name_to_index;
    for (std::size_t i = 0; i < body_names_.size(); ++i) {
      body_name_to_index.emplace(body_names_[i], i);
    }

    auto base_it = body_name_to_index.find("base_link");
    if (base_it == body_name_to_index.end()) {
      throw std::runtime_error(
          "Motion ONNX body_names metadata must include base_link");
    }
    root_body_index_ = base_it->second;
  }

  MotionFrame runFrame(std::size_t frame_index) {
    std::vector<int64_t> time_step{static_cast<int64_t>(frame_index)};
    std::vector<int64_t> input_shape{1, 1};
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, time_step.data(), time_step.size(), input_shape.data(),
        input_shape.size()));

    std::vector<Ort::Value> outputs =
        session_->Run(Ort::RunOptions{nullptr}, input_name_ptrs_.data(),
                      input_tensors.data(), input_tensors.size(),
                      output_name_ptrs_.data(), output_name_ptrs_.size());

    MotionFrame frame;
    frame.valid = true;
    copyJointOutput(outputs, "joint_pos", frame.joint_position);
    copyJointOutput(outputs, "joint_vel", frame.joint_velocity);
    copyRootOutputs(outputs, frame);
    copyBodyPositions(outputs, frame);
    return frame;
  }

  const Ort::Value &outputByName(const std::vector<Ort::Value> &outputs,
                                 const std::string &name) const {
    return outputs[output_indices_.at(name)];
  }

  bool hasOutput(const std::string &name) const {
    return output_indices_.find(name) != output_indices_.end();
  }

  template <std::size_t N>
  void copyJointOutput(const std::vector<Ort::Value> &outputs,
                       const std::string &name,
                       std::array<double, N> &target) const {
    const Ort::Value &output = outputByName(outputs, name);
    if (TensorElementCount(output) < target.size()) {
      throw std::runtime_error("Motion ONNX output '" + name +
                               "' is smaller than expected");
    }
    const float *data = output.GetTensorData<float>();
    for (std::size_t i = 0; i < target.size(); ++i) {
      target[i] = static_cast<double>(data[i]);
    }
  }

  void copyRootOutputs(const std::vector<Ort::Value> &outputs,
                       MotionFrame &frame) const {
    const Ort::Value &body_pos_w = outputByName(outputs, "body_pos_w");
    const Ort::Value &body_quat_w = outputByName(outputs, "body_quat_w");
    const Ort::Value &body_lin_vel_b = outputByName(outputs, "body_lin_vel_b");
    const Ort::Value &body_ang_vel_b = outputByName(outputs, "body_ang_vel_b");

    const std::size_t root_pos_offset = root_body_index_ * 3;
    const std::size_t root_quat_offset = root_body_index_ * 4;
    if (TensorElementCount(body_pos_w) < root_pos_offset + 3 ||
        TensorElementCount(body_quat_w) < root_quat_offset + 4 ||
        TensorElementCount(body_lin_vel_b) < root_pos_offset + 3 ||
        TensorElementCount(body_ang_vel_b) < root_pos_offset + 3) {
      throw std::runtime_error(
          "Motion ONNX root body outputs are smaller than expected");
    }

    const float *root_pos_data =
        body_pos_w.GetTensorData<float>() + root_pos_offset;
    const float *root_quat_data =
        body_quat_w.GetTensorData<float>() + root_quat_offset;
    const float *root_lin_vel_data =
        body_lin_vel_b.GetTensorData<float>() + root_pos_offset;
    const float *root_ang_vel_data =
        body_ang_vel_b.GetTensorData<float>() + root_pos_offset;

    frame.anchor_position = {static_cast<double>(root_pos_data[0]),
                             static_cast<double>(root_pos_data[1]),
                             static_cast<double>(root_pos_data[2])};
    frame.anchor_quaternion_wxyz = {static_cast<double>(root_quat_data[0]),
                                    static_cast<double>(root_quat_data[1]),
                                    static_cast<double>(root_quat_data[2]),
                                    static_cast<double>(root_quat_data[3])};
    frame.anchor_quaternion_valid = true;
    frame.anchor_linear_velocity_b = {
        static_cast<double>(root_lin_vel_data[0]),
        static_cast<double>(root_lin_vel_data[1]),
        static_cast<double>(root_lin_vel_data[2])};
    frame.anchor_angular_velocity_b = {
        static_cast<double>(root_ang_vel_data[0]),
        static_cast<double>(root_ang_vel_data[1]),
        static_cast<double>(root_ang_vel_data[2])};

    if (hasOutput("body_lin_vel_w")) {
      const Ort::Value &body_lin_vel_w =
          outputByName(outputs, "body_lin_vel_w");
      if (TensorElementCount(body_lin_vel_w) < root_pos_offset + 3) {
        throw std::runtime_error("Motion ONNX body_lin_vel_w output is smaller "
                                 "than body_names metadata");
      }
      const float *root_lin_vel_w_data =
          body_lin_vel_w.GetTensorData<float>() + root_pos_offset;
      frame.anchor_linear_velocity = {
          static_cast<double>(root_lin_vel_w_data[0]),
          static_cast<double>(root_lin_vel_w_data[1]),
          static_cast<double>(root_lin_vel_w_data[2])};
    } else {
      frame.anchor_linear_velocity = RotateAnchorVectorToWorldFrame(
          frame.anchor_quaternion_wxyz, frame.anchor_linear_velocity_b);
    }

    if (hasOutput("body_ang_vel_w")) {
      const Ort::Value &body_ang_vel_w =
          outputByName(outputs, "body_ang_vel_w");
      if (TensorElementCount(body_ang_vel_w) < root_pos_offset + 3) {
        throw std::runtime_error("Motion ONNX body_ang_vel_w output is smaller "
                                 "than body_names metadata");
      }
      const float *root_ang_vel_w_data =
          body_ang_vel_w.GetTensorData<float>() + root_pos_offset;
      frame.anchor_angular_velocity = {
          static_cast<double>(root_ang_vel_w_data[0]),
          static_cast<double>(root_ang_vel_w_data[1]),
          static_cast<double>(root_ang_vel_w_data[2])};
    } else {
      frame.anchor_angular_velocity = RotateAnchorVectorToWorldFrame(
          frame.anchor_quaternion_wxyz, frame.anchor_angular_velocity_b);
    }
  }

  void copyBodyPositions(const std::vector<Ort::Value> &outputs,
                         MotionFrame &frame) const {
    const Ort::Value &body_pos_b = outputByName(outputs, "body_pos_b");
    const float *data = body_pos_b.GetTensorData<float>();
    const std::size_t element_count = TensorElementCount(body_pos_b);

    if (element_count < body_names_.size() * 3) {
      throw std::runtime_error(
          "Motion ONNX body_pos_b output is smaller than body_names metadata");
    }

    frame.body_names = body_names_;
    frame.body_position.resize(body_names_.size() * 3);
    for (std::size_t i = 0; i < frame.body_position.size(); ++i) {
      frame.body_position[i] = static_cast<double>(data[i]);
    }

    const Ort::Value &body_lin_vel_b = outputByName(outputs, "body_lin_vel_b");
    const Ort::Value &body_ang_vel_b = outputByName(outputs, "body_ang_vel_b");
    if (TensorElementCount(body_lin_vel_b) < body_names_.size() * 3 ||
        TensorElementCount(body_ang_vel_b) < body_names_.size() * 3) {
      throw std::runtime_error("Motion ONNX body velocity outputs are smaller "
                               "than body_names metadata");
    }
    const float *lin_vel_data = body_lin_vel_b.GetTensorData<float>();
    const float *ang_vel_data = body_ang_vel_b.GetTensorData<float>();
    frame.body_linear_velocity.resize(body_names_.size() * 3);
    frame.body_angular_velocity.resize(body_names_.size() * 3);
    for (std::size_t i = 0; i < frame.body_linear_velocity.size(); ++i) {
      frame.body_linear_velocity[i] = static_cast<double>(lin_vel_data[i]);
      frame.body_angular_velocity[i] = static_cast<double>(ang_vel_data[i]);
    }

    if (hasOutput("body_quat_b")) {
      const Ort::Value &body_quat_b = outputByName(outputs, "body_quat_b");
      const float *quat_data = body_quat_b.GetTensorData<float>();
      const std::size_t quat_element_count = TensorElementCount(body_quat_b);
      if (quat_element_count < body_names_.size() * 4) {
        throw std::runtime_error("Motion ONNX body_quat_b output is smaller "
                                 "than body_names metadata");
      }
      frame.body_orientation.resize(body_names_.size() * 4);
      for (std::size_t i = 0; i < frame.body_orientation.size(); ++i) {
        frame.body_orientation[i] = static_cast<double>(quat_data[i]);
      }
    }
  }

  MotionSourceConfig config_;
  double fps_ = 50.0;
  std::size_t num_frames_ = 0;
  std::size_t root_body_index_ = 0;
  std::vector<std::string> body_names_;
  Clock::time_point start_time_ = Clock::now();
  bool cached_frame_valid_ = false;
  std::size_t cached_frame_index_ = 0;
  MotionFrame cached_frame_;

  Ort::Env env_;
  Ort::SessionOptions session_options_{};
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo memory_info_ =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char *> input_name_ptrs_;
  std::vector<const char *> output_name_ptrs_;
  std::unordered_map<std::string, std::size_t> output_indices_;
};

} // namespace

std::unique_ptr<MotionFrameSource>
CreateMotionFrameSource(const InferenceConfig &config) {
  if (config.motion_source.type == "null" ||
      config.motion_source.type == "hold") {
    return std::make_unique<NullMotionFrameSource>();
  }
  if (config.motion_source.type == "csv_replay") {
    return std::make_unique<CsvReplayMotionFrameSource>(config.motion_source);
  }
  if (config.motion_source.type == "onnx_replay") {
    return std::make_unique<OnnxReplayMotionFrameSource>(config.motion_source);
  }

  throw std::runtime_error("Unsupported motion_source.type: " +
                           config.motion_source.type);
}

} // namespace igris_c_gmt_public
