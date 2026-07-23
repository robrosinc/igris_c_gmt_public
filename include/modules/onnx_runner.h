#pragma once

#include "core/pipeline_types.hpp"

#include "onnxruntime_cxx_api.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace igris_c_gmt_public {

class OnnxRunner {
public:
  explicit OnnxRunner(std::string label = "OnnxRunner");

  int configure(const std::string &onnx_path);
  int run(const std::map<std::string, Eigen::VectorXd> &observation_groups,
          Vector23d &actions);

  double lastInferenceMs() const { return last_inference_ms_; }
  uint64_t inferenceCount() const { return inference_count_; }

private:
  std::vector<int64_t> resolveInputShape(std::size_t input_index,
                                         std::size_t element_count) const;

private:
  std::string label_;
  double last_inference_ms_ = 0.0;
  uint64_t inference_count_ = 0;

  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "public_inference_onnx_runner"};
  Ort::SessionOptions session_options_{};
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo memory_info_ =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::unique_ptr<Ort::Session> session_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char *> input_name_ptrs_;
  std::vector<const char *> output_name_ptrs_;
  std::vector<std::vector<int64_t>> input_shapes_;
  int output_actions_index_ = -1;
};

} // namespace igris_c_gmt_public
