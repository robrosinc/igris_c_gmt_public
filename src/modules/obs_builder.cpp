#include "modules/obs_builder.h"

#include "yaml-cpp/yaml.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace igris_c_gmt_public {
namespace {

YAML::Node GetRoot(const YAML::Node &root) {
  const YAML::Node observation = root["igris_c_gmt_public_observation"];
  if (!observation) {
    throw std::runtime_error(
        "obs.yaml must contain igris_c_gmt_public_observation root");
  }
  return observation;
}

template <typename T>
T GetOr(const YAML::Node &node, const char *key, const T &default_value) {
  if (!node || !node[key]) {
    return default_value;
  }
  return node[key].as<T>();
}

void CopyVector23(const YAML::Node &node, const char *key, Vector23d &target) {
  const YAML::Node value = node[key];
  if (!value) {
    throw std::runtime_error("observation." + std::string(key) +
                             " is required");
  }
  if (!value.IsSequence() ||
      value.size() != static_cast<std::size_t>(target.size())) {
    throw std::runtime_error("Expected observation." + std::string(key) +
                             " with 23 entries");
  }
  for (Eigen::Index i = 0; i < target.size(); ++i) {
    target(i) = value[static_cast<std::size_t>(i)].as<double>();
  }
}

std::vector<int> BuildMotionFrameOffsets(const YAML::Node &params) {
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

obs_functions::ObsFunction MakeMotionFrameStackFunction(
    const YAML::Node &params,
    const std::unordered_map<std::string, obs_functions::ObsFunction>
        &registry) {
  const std::string term_func_name =
      GetOr<std::string>(params, "term_func", "");
  if (term_func_name.empty()) {
    throw std::runtime_error("motion_frame_stack.params.term_func is required");
  }
  if (params && params["term_params"]) {
    throw std::runtime_error(
        "motion_frame_stack.params.term_params is not supported in deploy");
  }

  const auto term_it = registry.find(term_func_name);
  if (term_it == registry.end() || term_func_name == "motion_frame_stack") {
    throw std::runtime_error("Unknown motion_frame_stack term_func: " +
                             term_func_name);
  }

  const obs_functions::ObsFunction term_func = term_it->second;
  const std::vector<int> offsets = BuildMotionFrameOffsets(params);
  return [term_func, offsets](const ObservationInput &input) {
    std::vector<Eigen::VectorXd> frame_values;
    frame_values.reserve(offsets.size());
    Eigen::Index total_size = 0;
    Eigen::Index frame_value_size = -1;

    for (int offset : offsets) {
      ObservationInput frame_input = input;
      frame_input.motion_frames = {obs_functions::MotionFrameAtOffset(input,
                                                                       offset)};
      frame_input.motion_frame_offsets = {0};
      Eigen::VectorXd value = term_func(frame_input);
      if (frame_value_size < 0) {
        frame_value_size = value.size();
      } else if (value.size() != frame_value_size) {
        throw std::runtime_error(
            "motion_frame_stack term produced inconsistent frame sizes");
      }
      total_size += value.size();
      frame_values.push_back(std::move(value));
    }

    Eigen::VectorXd stacked(total_size);
    Eigen::Index output_offset = 0;
    for (const Eigen::VectorXd &value : frame_values) {
      stacked.segment(output_offset, value.size()) = value;
      output_offset += value.size();
    }
    return stacked;
  };
}

} // namespace

int ObservationBuilder::configure(const InferenceConfig &config) {
  obs_config_path_ = config.obs_config_path;
  q_default_.setZero();
  groups_.clear();

  try {
    const YAML::Node root = YAML::LoadFile(obs_config_path_);
    const YAML::Node obs = GetRoot(root);
    CopyVector23(obs, "q_default", q_default_);

    const YAML::Node groups = obs["groups"];
    if (!groups || !groups.IsSequence() || groups.size() == 0) {
      throw std::runtime_error(
          "observation.groups must be a non-empty sequence");
    }

    const auto &registry = obs_functions::Registry();
    for (const YAML::Node &group_node : groups) {
      GroupState group;
      group.name = GetOr<std::string>(group_node, "name", "");
      if (group.name.empty()) {
        throw std::runtime_error("observation group name must not be empty");
      }

      const YAML::Node terms = group_node["terms"];
      if (!terms || !terms.IsSequence() || terms.size() == 0) {
        throw std::runtime_error("observation group '" + group.name +
                                 "' must define terms");
      }

      for (const YAML::Node &term_node : terms) {
        TermState term;
        term.function_name = GetOr<std::string>(term_node, "function", "");
        term.history_length =
            static_cast<std::size_t>(GetOr<int>(term_node, "history", 1));
        if (term.function_name.empty() || term.history_length == 0) {
          throw std::runtime_error(
              "observation term must define function and positive history");
        }
        if (term.function_name == "motion_frame_stack") {
          term.function =
              MakeMotionFrameStackFunction(term_node["params"], registry);
        } else {
          const auto it = registry.find(term.function_name);
          if (it == registry.end()) {
            throw std::runtime_error("Unknown observation function: " +
                                     term.function_name);
          }
          term.function = it->second;
        }
        group.terms.push_back(std::move(term));
      }
      groups_.push_back(std::move(group));
    }
  } catch (const std::exception &exception) {
    std::cerr << "ObservationBuilder failed to load " << obs_config_path_
              << ": " << exception.what() << "\n";
    return -1;
  }

  reset();
  return 0;
}

void ObservationBuilder::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  latest_result_ = ObservationResult{};
  has_result_ = false;
  resetHistories();
}

int ObservationBuilder::build(const ObservationInput &input,
                              ObservationResult &result) {
  if (!input.state.valid || input.motion_frames.empty()) {
    return -1;
  }

  ObservationInput prepared_input = input;
  prepared_input.q_default = q_default_;

  result = ObservationResult{};
  const MotionFrame &latest_motion_frame =
      obs_functions::LatestMotionFrame(prepared_input);
  for (std::size_t i = 0; i < latest_motion_frame.joint_position.size(); ++i) {
    result.motion_joint_position(static_cast<Eigen::Index>(i)) =
        latest_motion_frame.joint_position[i];
  }

  try {
    for (GroupState &group : groups_) {
      Eigen::Index group_size = 0;
      std::vector<Eigen::VectorXd> flattened_terms;
      flattened_terms.reserve(group.terms.size());

      for (TermState &term : group.terms) {
        const Eigen::VectorXd value = term.function(prepared_input);
        if (!term.initialized) {
          term.history.init(term.history_length, value);
          term.initialized = true;
        }
        term.history.push_back(value);
        Eigen::VectorXd flattened = term.history.toEigenVector();
        group_size += flattened.size();
        flattened_terms.push_back(std::move(flattened));
      }

      Eigen::VectorXd group_obs(group_size);
      Eigen::Index offset = 0;
      for (const Eigen::VectorXd &term_obs : flattened_terms) {
        group_obs.segment(offset, term_obs.size()) = term_obs;
        offset += term_obs.size();
      }
      result.groups[group.name] = std::move(group_obs);
    }
  } catch (const std::exception &exception) {
    std::cerr << "ObservationBuilder failed to build observation: "
              << exception.what() << "\n";
    return -1;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_result_ = result;
    has_result_ = true;
  }
  return 0;
}

bool ObservationBuilder::readLatest(ObservationResult &result) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!has_result_) {
    return false;
  }
  result = latest_result_;
  return true;
}

void ObservationBuilder::resetHistories() {
  for (GroupState &group : groups_) {
    for (TermState &term : group.terms) {
      term.history = RollingHistory{};
      term.initialized = false;
    }
  }
}

} // namespace igris_c_gmt_public
