#include "modules/onnx_runner.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace igris_c_gmt_public {
namespace {

std::size_t CountStaticElements(const std::vector<int64_t> &shape) {
  std::size_t count = 1;
  for (const int64_t dimension : shape) {
    if (dimension <= 0) {
      continue;
    }
    if (static_cast<std::size_t>(dimension) >
        (std::numeric_limits<std::size_t>::max() / count)) {
      throw std::overflow_error("invalid ONNX input shape");
    }
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}

bool HasDynamicDimension(const std::vector<int64_t> &shape) {
  return std::any_of(shape.begin(), shape.end(),
                     [](int64_t dimension) { return dimension <= 0; });
}

} // namespace

OnnxRunner::OnnxRunner(std::string label) : label_(std::move(label)) {}

int OnnxRunner::configure(const std::string &onnx_path) {
  if (onnx_path.empty()) {
    std::cerr << label_ << " policy.onnx path is empty\n";
    return -1;
  }
  if (!std::filesystem::exists(onnx_path)) {
    std::cerr << label_ << " policy not found: " << onnx_path << "\n";
    return -1;
  }

  try {
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options_.AddConfigEntry("session.use_deterministic_compute", "1");
    session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(),
                                              session_options_);

    input_names_.clear();
    output_names_.clear();
    input_name_ptrs_.clear();
    output_name_ptrs_.clear();
    input_shapes_.clear();
    output_actions_index_ = -1;

    const std::size_t input_count = session_->GetInputCount();
    const std::size_t output_count = session_->GetOutputCount();
    if (input_count == 0 || output_count == 0) {
      std::cerr << label_ << " invalid ONNX tensor count\n";
      return -1;
    }

    input_names_.resize(input_count);
    input_name_ptrs_.resize(input_count);
    input_shapes_.resize(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
      Ort::AllocatedStringPtr input_name =
          session_->GetInputNameAllocated(i, allocator_);
      input_names_[i] = input_name.get();
      input_name_ptrs_[i] = input_names_[i].c_str();
      const Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
      const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::cerr << label_ << " input '" << input_names_[i]
                  << "' must be float\n";
        return -1;
      }
      input_shapes_[i] = tensor_info.GetShape();
      std::cerr << label_ << " input[" << i << "] name='" << input_names_[i]
                << "'\n";
    }

    output_names_.resize(output_count);
    output_name_ptrs_.resize(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
      Ort::AllocatedStringPtr output_name =
          session_->GetOutputNameAllocated(i, allocator_);
      output_names_[i] = output_name.get();
      output_name_ptrs_[i] = output_names_[i].c_str();
      if (output_names_[i] == "actions" || output_names_[i] == "action") {
        output_actions_index_ = static_cast<int>(i);
      }
      std::cerr << label_ << " output[" << i << "] name='" << output_names_[i]
                << "'\n";
    }
    if (output_actions_index_ < 0) {
      std::cerr << label_
                << " policy is missing required action/actions output\n";
      return -1;
    }
  } catch (const Ort::Exception &exception) {
    std::cerr << label_
              << " failed to create ONNX Runtime session: " << exception.what()
              << "\n";
    return -1;
  } catch (const std::exception &exception) {
    std::cerr << label_
              << " failed to inspect ONNX policy: " << exception.what() << "\n";
    return -1;
  }

  return 0;
}

int OnnxRunner::run(
    const std::map<std::string, Eigen::VectorXd> &observation_groups,
    Vector23d &actions) {
  if (!session_ || output_actions_index_ < 0) {
    return -1;
  }

  std::vector<std::vector<float>> input_buffers;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<Ort::Value> input_tensors;
  input_buffers.reserve(input_names_.size());
  input_shapes.reserve(input_names_.size());
  input_tensors.reserve(input_names_.size());

  try {
    for (std::size_t i = 0; i < input_names_.size(); ++i) {
      const auto obs_it = observation_groups.find(input_names_[i]);
      if (obs_it == observation_groups.end()) {
        std::cerr << label_ << " missing observation group for ONNX input '"
                  << input_names_[i] << "'\n";
        return -1;
      }

      const Eigen::VectorXd &obs = obs_it->second;
      input_buffers.emplace_back(static_cast<std::size_t>(obs.size()), 0.0f);
      for (Eigen::Index j = 0; j < obs.size(); ++j) {
        input_buffers.back()[static_cast<std::size_t>(j)] =
            static_cast<float>(obs(j));
      }
      input_shapes.push_back(resolveInputShape(i, input_buffers.back().size()));
      input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
          memory_info_, input_buffers.back().data(),
          input_buffers.back().size(), input_shapes.back().data(),
          input_shapes.back().size()));
    }

    const auto inference_start = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs =
        session_->Run(Ort::RunOptions{nullptr}, input_name_ptrs_.data(),
                      input_tensors.data(), input_tensors.size(),
                      output_name_ptrs_.data(), output_name_ptrs_.size());
    last_inference_ms_ = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - inference_start)
                             .count();
    ++inference_count_;

    const Ort::Value &action_tensor =
        outputs[static_cast<std::size_t>(output_actions_index_)];
    const auto action_info = action_tensor.GetTensorTypeAndShapeInfo();
    if (action_info.GetElementCount() <
        static_cast<std::size_t>(kRlNumJointActions)) {
      std::cerr << label_ << " expected " << kRlNumJointActions
                << " policy outputs but received "
                << action_info.GetElementCount() << "\n";
      return -1;
    }
    const float *action_data = action_tensor.GetTensorData<float>();
    for (int i = 0; i < kRlNumJointActions; ++i) {
      actions(i) =
          std::clamp(static_cast<double>(action_data[i]), -100.0, 100.0);
    }
  } catch (const Ort::Exception &exception) {
    std::cerr << label_ << " ONNX inference failed: " << exception.what()
              << "\n";
    return -1;
  } catch (const std::exception &exception) {
    std::cerr << label_ << " inference failed: " << exception.what() << "\n";
    return -1;
  }

  return 0;
}

std::vector<int64_t>
OnnxRunner::resolveInputShape(std::size_t input_index,
                              std::size_t element_count) const {
  std::vector<int64_t> shape = input_shapes_.at(input_index);
  if (shape.empty()) {
    shape.push_back(static_cast<int64_t>(element_count));
    return shape;
  }

  if (!HasDynamicDimension(shape)) {
    std::size_t expected = 1;
    for (const int64_t dimension : shape) {
      expected *= static_cast<std::size_t>(dimension);
    }
    if (expected != element_count) {
      throw std::runtime_error(
          "observation group '" + input_names_.at(input_index) + "' has " +
          std::to_string(element_count) + " elements but ONNX input expects " +
          std::to_string(expected));
    }
    return shape;
  }

  std::size_t known_count = CountStaticElements(shape);
  int dynamic_count = 0;
  int dynamic_index = -1;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0) {
      ++dynamic_count;
      dynamic_index = static_cast<int>(i);
    }
  }
  if (dynamic_count == 1 && known_count > 0 &&
      (element_count % known_count) == 0) {
    shape[static_cast<std::size_t>(dynamic_index)] =
        static_cast<int64_t>(element_count / known_count);
    return shape;
  }
  if (shape.size() == 2) {
    return {1, static_cast<int64_t>(element_count)};
  }
  return {static_cast<int64_t>(element_count)};
}

} // namespace igris_c_gmt_public
