#pragma once

#include "core/config.hpp"
#include "core/pipeline_types.hpp"

#include <array>

namespace igris_c_gmt_public {

class ActionBuilder {
public:
  int configure(const InferenceConfig &config);
  void reset();
  int build(const ProcessedState &state, const ObservationResult &observation,
            const Vector23d &actions, InferenceCommand &command);
  void buildHoldCommand(const ProcessedState &state,
                        InferenceCommand &command) const;

  const Vector23d &lastActions() const { return last_actions_; }

private:
  Vector23d composeTargetJointPositions(const ObservationResult &observation,
                                        const Vector23d &actions) const;
  void processTargetPositions(const ProcessedState &state,
                              VectorQd &desired_position) const;
  static bool isParallelJoint(int joint_index);

private:
  bool use_motion_residual_action_ = false;
  Vector23d q_default_ = Vector23d::Zero();
  Vector23d action_scale_ = Vector23d::Zero();
  Vector23d joint_pos_limit_high_ = Vector23d::Zero();
  Vector23d joint_pos_limit_low_ = Vector23d::Zero();
  VectorQd command_kp_ = VectorQd::Zero();
  VectorQd command_kd_ = VectorQd::Zero();
  std::array<igris_c::msg::dds::KinematicMode, kParallelGroupCount>
      kinematic_modes_{};
  double non_parallel_safety_coef_ = 0.0;
  Vector23d last_actions_ = Vector23d::Zero();
};

} // namespace igris_c_gmt_public
