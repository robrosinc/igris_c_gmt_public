#pragma once

#include "igris_c_sdk/igris_c_msgs.hpp"

#include <Eigen/Dense>

#include <array>
#include <cstdint>

namespace public_inference_module {

inline constexpr int kModelDof           = static_cast<int>(igris_c::msg::dds::N_JOINTS);
inline constexpr int kParallelGroupCount = 5;

enum JointIndex {
    TM1_WY  = 0,
    TM1_WR  = 1,
    TM1_WP  = 2,
    TM1_LHP = 3,
    TM1_LHR = 4,
    TM1_LHY = 5,
    TM1_LKP = 6,
    TM1_LAP = 7,
    TM1_LAR = 8,
    TM1_RHP = 9,
    TM1_RHR = 10,
    TM1_RHY = 11,
    TM1_RKP = 12,
    TM1_RAP = 13,
    TM1_RAR = 14,
    TM1_LSP = 15,
    TM1_LSR = 16,
    TM1_LSY = 17,
    TM1_LEP = 18,
    TM1_RSP = 22,
    TM1_RSR = 23,
    TM1_RSY = 24,
    TM1_REP = 25,
};

inline constexpr int kRlNumTotalActions           = 23;
inline constexpr int kRlNumJointActions           = 23;
inline constexpr int kRlObsHistoryLen             = 10;
inline constexpr int kRlReferenceObsHistoryLen    = 20;
inline constexpr int kRlReferenceFutureFrameCount = 0;
inline constexpr int kRlReferenceFrameStackLength = kRlReferenceFutureFrameCount + 1;
inline constexpr int kRlNumMotionJointPositionObs = kRlNumJointActions;
inline constexpr int kRlNumMotionJointVelocityObs = kRlNumJointActions;
inline constexpr int kRlNumBaseAngularVelocityObs = 3;
inline constexpr int kRlNumProjectedGravityObs    = 3;
inline constexpr int kRlNumJointPositionObs       = kRlNumJointActions;
inline constexpr int kRlNumJointVelocityObs       = kRlNumJointActions;
inline constexpr int kRlNumLastActionsObs         = kRlNumJointActions;
inline constexpr int kRlNumReferenceRootPositionZObs       = 1;
inline constexpr int kRlNumReferenceRootStateObs           = 6;
inline constexpr int kRlReferenceKeybodyCount              = 14;
inline constexpr int kRlReferenceSourceKeybodyCount        = 34;
inline constexpr int kRlNumReferenceBodyPositionObs        = kRlReferenceKeybodyCount * 3;
inline constexpr int kRlNumReferenceRootLinearVelocityObs  = 3;
inline constexpr int kRlNumReferenceRootAngularVelocityObs = 3;
inline constexpr int kRlNumObsCurrent             = kRlNumMotionJointPositionObs + kRlNumMotionJointVelocityObs +
                                         kRlNumBaseAngularVelocityObs + kRlNumProjectedGravityObs + kRlNumJointPositionObs +
                                         kRlNumJointVelocityObs + kRlNumLastActionsObs;
inline constexpr int kRlNumObsHistory = kRlObsHistoryLen * kRlNumObsCurrent;
inline constexpr int kRlNumReferenceHistoryCurrent = kRlNumJointPositionObs + kRlNumJointVelocityObs + kRlNumProjectedGravityObs +
                                                     kRlNumBaseAngularVelocityObs + kRlNumLastActionsObs;
inline constexpr int kMotionDataReferenceTrackingValues =
    kRlNumMotionJointPositionObs + kRlNumMotionJointVelocityObs + kRlNumReferenceRootPositionZObs + kRlNumReferenceRootStateObs +
    kRlNumReferenceBodyPositionObs + kRlNumReferenceRootLinearVelocityObs + kRlNumReferenceRootAngularVelocityObs;
inline constexpr int kMotionDataReferenceTrackingValuesWithAnchor = kMotionDataReferenceTrackingValues + 4;
inline constexpr int kRlNumReferenceNonHistory = kRlReferenceFrameStackLength * kMotionDataReferenceTrackingValues;
inline constexpr int kRlNumReferenceTrackingObs = kRlReferenceObsHistoryLen * kRlNumReferenceHistoryCurrent + kRlNumReferenceNonHistory;

using VectorQd  = Eigen::Matrix<double, kModelDof, 1>;
using Vector23d = Eigen::Matrix<double, kRlNumJointActions, 1>;

inline constexpr std::array<int, kRlNumJointActions> kObsToSystemJointMapping = {
    TM1_LSP, TM1_RSP, TM1_WY,  TM1_LSR, TM1_RSR, TM1_WR,  TM1_LSY, TM1_RSY, TM1_WP,  TM1_LEP, TM1_REP, TM1_LHP,
    TM1_RHP, TM1_LHR, TM1_RHR, TM1_LHY, TM1_RHY, TM1_LKP, TM1_RKP, TM1_LAP, TM1_RAP, TM1_LAR, TM1_RAR,
};

inline constexpr std::array<int, kRlNumJointActions> kActionsToSystemJointMapping = {
    TM1_LSP, TM1_RSP, TM1_WY,  TM1_LSR, TM1_RSR, TM1_WR,  TM1_LSY, TM1_RSY, TM1_WP,  TM1_LEP, TM1_REP, TM1_LHP,
    TM1_RHP, TM1_LHR, TM1_RHR, TM1_LHY, TM1_RHY, TM1_LKP, TM1_RKP, TM1_LAP, TM1_RAP, TM1_LAR, TM1_RAR,
};

inline constexpr std::array<const char *, kRlNumJointActions> kExpectedJointNames = {
    "Joint_Shoulder_Pitch_Left", "Joint_Shoulder_Pitch_Right", "Joint_Waist_Yaw",         "Joint_Shoulder_Roll_Left",
    "Joint_Shoulder_Roll_Right", "Joint_Waist_Roll",           "Joint_Shoulder_Yaw_Left", "Joint_Shoulder_Yaw_Right",
    "Joint_Waist_Pitch",         "Joint_Elbow_Pitch_Left",     "Joint_Elbow_Pitch_Right", "Joint_Hip_Pitch_Left",
    "Joint_Hip_Pitch_Right",     "Joint_Hip_Roll_Left",        "Joint_Hip_Roll_Right",    "Joint_Hip_Yaw_Left",
    "Joint_Hip_Yaw_Right",       "Joint_Knee_Pitch_Left",      "Joint_Knee_Pitch_Right",  "Joint_Ankle_Pitch_Left",
    "Joint_Ankle_Pitch_Right",   "Joint_Ankle_Roll_Left",      "Joint_Ankle_Roll_Right",
};

struct MotionFrame {
    uint64_t seq       = 0;
    uint64_t stamp_ns  = 0;
    bool valid         = false;
    std::array<double, kRlNumJointActions> joint_position{};
    std::array<double, kRlNumJointActions> joint_velocity{};
    double root_position_z = 0.0;
    std::array<double, kRlNumReferenceRootStateObs> root_state{};
    std::array<double, kRlNumReferenceBodyPositionObs> body_position{};
    std::array<double, kRlNumReferenceRootLinearVelocityObs> root_linear_velocity{};
    std::array<double, kRlNumReferenceRootAngularVelocityObs> root_angular_velocity{};
    std::array<double, 4> anchor_quaternion_wxyz{1.0, 0.0, 0.0, 0.0};
    bool anchor_quaternion_valid = false;

    MotionFrame() {
        root_state = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    }
};

struct InferenceCommand {
    std::array<igris_c::msg::dds::KinematicMode, kParallelGroupCount> kinematic_modes{};
    VectorQd q     = VectorQd::Zero();
    VectorQd q_dot = VectorQd::Zero();
    VectorQd tau   = VectorQd::Zero();
    VectorQd kp    = VectorQd::Zero();
    VectorQd kd    = VectorQd::Zero();

    InferenceCommand() { kinematic_modes.fill(igris_c::msg::dds::KinematicMode::MS); }
};

}  // namespace public_inference_module
