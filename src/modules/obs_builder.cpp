#include "modules/obs_builder.h"

#include "yaml-cpp/yaml.h"

#include <iostream>
#include <stdexcept>
#include <utility>

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
        const auto it = registry.find(term.function_name);
        if (it == registry.end()) {
          throw std::runtime_error("Unknown observation function: " +
                                   term.function_name);
        }
        term.function = it->second;
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
  const MotionFrame &latest_motion_frame = prepared_input.motion_frames.front();
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
