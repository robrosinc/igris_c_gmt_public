#include "modules/action_builder.h"

#include "yaml-cpp/yaml.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace igris_c_gmt_public {
namespace {

YAML::Node GetRoot(const YAML::Node &root) {
  const YAML::Node action = root["igris_c_gmt_public_action"];
  if (!action) {
    throw std::runtime_error(
        "action.yaml must contain igris_c_gmt_public_action root");
  }
  return action;
}

template <typename T>
T ReadRequiredScalar(const YAML::Node &node, const char *key) {
  if (!node || !node[key]) {
    throw std::runtime_error("action." + std::string(key) + " is required");
  }
  return node[key].as<T>();
}

template <typename Derived>
void CopyFixedSequence(const YAML::Node &node, const char *key,
                       Eigen::MatrixBase<Derived> &target) {
  const YAML::Node value = node[key];
  if (!value) {
    throw std::runtime_error("action." + std::string(key) + " is required");
  }
  if (!value.IsSequence() ||
      value.size() != static_cast<std::size_t>(target.size())) {
    throw std::runtime_error("Expected action." + std::string(key) + " with " +
                             std::to_string(target.size()) + " entries");
  }
  for (Eigen::Index i = 0; i < target.size(); ++i) {
    target(i) = value[static_cast<std::size_t>(i)].as<double>();
  }
}

void CopyCommandVector(const YAML::Node &node, const char *key,
                       VectorQd &target) {
  const YAML::Node value = node[key];
  if (!value) {
    throw std::runtime_error("action." + std::string(key) + " is required");
  }
  if (!value.IsSequence()) {
    throw std::runtime_error("Expected action." + std::string(key) +
                             " sequence");
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
  throw std::runtime_error("Expected action." + std::string(key) +
                           " to have 23 or " + std::to_string(target.size()) +
                           " entries");
}

void CopyJointLimits(const YAML::Node &node, const char *key,
                     Vector23d &target) {
  const YAML::Node value = node[key];
  if (!value) {
    throw std::runtime_error("action." + std::string(key) + " is required");
  }
  if (!value.IsSequence()) {
    throw std::runtime_error("Expected action." + std::string(key) +
                             " sequence");
  }
  if (value.size() == static_cast<std::size_t>(target.size())) {
    for (Eigen::Index i = 0; i < target.size(); ++i) {
      target(i) = value[static_cast<std::size_t>(i)].as<double>();
    }
    return;
  }
  if (value.size() == static_cast<std::size_t>(kModelDof)) {
    for (std::size_t i = 0; i < kActionsToSystemJointMapping.size(); ++i) {
      target(static_cast<Eigen::Index>(i)) =
          value[static_cast<std::size_t>(kActionsToSystemJointMapping[i])]
              .as<double>();
    }
    return;
  }
  throw std::runtime_error("Expected action." + std::string(key) +
                           " to have 23 or " + std::to_string(kModelDof) +
                           " entries");
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

void CopyKinematicModes(
    const YAML::Node &node,
    std::array<igris_c::msg::dds::KinematicMode, kParallelGroupCount> &target) {
  if (!node) {
    throw std::runtime_error("action.kinematic_modes is required");
  }
  if (!node.IsSequence() || node.size() != target.size()) {
    throw std::runtime_error("Expected action.kinematic_modes with 5 entries");
  }
  for (std::size_t i = 0; i < target.size(); ++i) {
    target[i] = ParseKinematicMode(node[i]);
  }
}

} // namespace

int ActionBuilder::configure(const InferenceConfig &config) {
  try {
    if (config.action_config_path.empty() ||
        !std::filesystem::exists(config.action_config_path) ||
        std::filesystem::file_size(config.action_config_path) == 0) {
      throw std::runtime_error("action config path is empty or missing");
    }
    const YAML::Node action =
        GetRoot(YAML::LoadFile(config.action_config_path));
    use_motion_residual_action_ =
        ReadRequiredScalar<bool>(action, "use_motion_residual_action");
    non_parallel_safety_coef_ =
        ReadRequiredScalar<double>(action, "non_parallel_safety_coef");
    CopyFixedSequence(action, "q_default", q_default_);
    CopyFixedSequence(action, "action_scale", action_scale_);
    CopyJointLimits(action, "joint_position_max", joint_pos_limit_high_);
    CopyJointLimits(action, "joint_position_min", joint_pos_limit_low_);
    CopyCommandVector(action, "kp", command_kp_);
    CopyCommandVector(action, "kd", command_kd_);
    CopyKinematicModes(action["kinematic_modes"], kinematic_modes_);
  } catch (const std::exception &exception) {
    std::cerr << "ActionBuilder failed to load " << config.action_config_path
              << ": " << exception.what() << "\n";
    return -1;
  }

  for (const auto mode : kinematic_modes_) {
    if (mode != igris_c::msg::dds::KinematicMode::PJS) {
      std::cerr << "ActionBuilder only supports PJS kinematic modes because "
                   "the policy/action layout is joint-space\n";
      return -1;
    }
  }

  reset();
  return 0;
}

void ActionBuilder::reset() { last_actions_.setZero(); }

int ActionBuilder::build(const ProcessedState &state,
                         const ObservationResult &observation,
                         const Vector23d &actions, InferenceCommand &command) {
  if (!state.valid) {
    return -1;
  }

  buildHoldCommand(state, command);

  VectorQd desired_q = command.q;
  const Vector23d target_joint_positions =
      composeTargetJointPositions(observation, actions);
  for (std::size_t i = 0; i < kActionsToSystemJointMapping.size(); ++i) {
    const int joint_index = kActionsToSystemJointMapping[i];
    desired_q(joint_index) =
        target_joint_positions(static_cast<Eigen::Index>(i));
    command.kp(joint_index) = command_kp_(joint_index);
    command.kd(joint_index) = command_kd_(joint_index);
  }

  processTargetPositions(state, desired_q);
  command.q = desired_q;

  last_actions_ = actions;
  return 0;
}

void ActionBuilder::buildHoldCommand(const ProcessedState &state,
                                     InferenceCommand &command) const {
  command.kinematic_modes = kinematic_modes_;
  command.q.setZero();
  command.q_dot.setZero();
  command.tau.setZero();
  command.kp = command_kp_;
  command.kd = command_kd_;
  command.q = state.joint_position;
}

Vector23d
ActionBuilder::composeTargetJointPositions(const ObservationResult &observation,
                                           const Vector23d &actions) const {
  if (use_motion_residual_action_) {
    return observation.motion_joint_position +
           actions.cwiseProduct(action_scale_);
  }
  return q_default_ + actions.cwiseProduct(action_scale_);
}

void ActionBuilder::processTargetPositions(const ProcessedState &state,
                                           VectorQd &desired_position) const {
  for (std::size_t i = 0; i < kObsToSystemJointMapping.size(); ++i) {
    const int joint_index = kObsToSystemJointMapping[i];
    if (isParallelJoint(joint_index)) {
      continue;
    }

    const double hard_limit_low =
        joint_pos_limit_low_(static_cast<Eigen::Index>(i));
    const double hard_limit_high =
        joint_pos_limit_high_(static_cast<Eigen::Index>(i));
    const double mid_point = 0.5 * (hard_limit_low + hard_limit_high);
    const double radius = hard_limit_high - hard_limit_low;
    const double soft_limit_low =
        mid_point - 0.5 * radius * non_parallel_safety_coef_;
    const double soft_limit_high =
        mid_point + 0.5 * radius * non_parallel_safety_coef_;

    double &target = desired_position(joint_index);
    const double joint_q = state.joint_position(joint_index);

    if (joint_q > soft_limit_high && target > hard_limit_high) {
      target -=
          std::clamp((joint_q - soft_limit_high) /
                         std::max(1.0e-6, hard_limit_high - soft_limit_high),
                     0.0, 1.0) *
          (target - hard_limit_high);
    } else if (joint_q < soft_limit_low && target < hard_limit_low) {
      target +=
          std::clamp((soft_limit_low - joint_q) /
                         std::max(1.0e-6, soft_limit_low - hard_limit_low),
                     0.0, 1.0) *
          (hard_limit_low - target);
    }
  }
}

bool ActionBuilder::isParallelJoint(int joint_index) {
  return joint_index == _WR || joint_index == _WP || joint_index == _LAP ||
         joint_index == _LAR || joint_index == _RAP || joint_index == _RAR;
}

} // namespace igris_c_gmt_public
