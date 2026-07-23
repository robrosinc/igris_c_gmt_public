#pragma once

#include "public_inference_module/types.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace public_inference_module {

struct MotionDataSample {
    uint64_t seq      = 0;
    uint64_t stamp_ns = 0;
    bool valid        = false;
    bool anchor_quaternion_valid = false;
    std::array<double, 4> anchor_quaternion_wxyz{1.0, 0.0, 0.0, 0.0};
    std::vector<double> values;
};

class MotionDataBuffer {
  public:
    void write(MotionDataSample sample) {
        sample.valid = true;
        auto snapshot = std::make_shared<const MotionDataSample>(std::move(sample));
        std::atomic_store_explicit(&latest_sample_, snapshot, std::memory_order_release);
    }

    std::shared_ptr<const MotionDataSample> readLatest() const {
        return std::atomic_load_explicit(&latest_sample_, std::memory_order_acquire);
    }

  private:
    mutable std::shared_ptr<const MotionDataSample> latest_sample_;
};

inline MotionDataSample EncodeMotionFrameAsMotionData(const MotionFrame &frame, const std::string &layout) {
    MotionDataSample sample;
    sample.seq      = frame.seq;
    sample.stamp_ns = frame.stamp_ns;
    sample.valid    = frame.valid;
    sample.anchor_quaternion_valid = frame.anchor_quaternion_valid;
    sample.anchor_quaternion_wxyz = frame.anchor_quaternion_wxyz;

    if (layout == "q23_dq23_quatwxyz4") {
        sample.values.reserve(frame.joint_position.size() + frame.joint_velocity.size() + frame.anchor_quaternion_wxyz.size());
        sample.values.insert(sample.values.end(), frame.joint_position.begin(), frame.joint_position.end());
        sample.values.insert(sample.values.end(), frame.joint_velocity.begin(), frame.joint_velocity.end());
        sample.values.insert(sample.values.end(), frame.anchor_quaternion_wxyz.begin(), frame.anchor_quaternion_wxyz.end());
        return sample;
    }

    if (layout == "general_motion_tracking_v1") {
        sample.values.reserve(kMotionDataGmtValues);
        sample.values.insert(sample.values.end(), frame.joint_position.begin(), frame.joint_position.end());
        sample.values.insert(sample.values.end(), frame.gmt_body_position.begin(), frame.gmt_body_position.end());
        sample.values.insert(sample.values.end(), frame.root_linear_velocity.begin(), frame.root_linear_velocity.end());
        sample.values.insert(sample.values.end(), frame.root_angular_velocity.begin(), frame.root_angular_velocity.end());
        sample.values.push_back(frame.root_position_z);
        return sample;
    }

    sample.values.reserve(kMotionDataReferenceTrackingValues);
    sample.values.insert(sample.values.end(), frame.joint_position.begin(), frame.joint_position.end());
    sample.values.insert(sample.values.end(), frame.joint_velocity.begin(), frame.joint_velocity.end());
    sample.values.push_back(frame.root_position_z);
    sample.values.insert(sample.values.end(), frame.root_state.begin(), frame.root_state.end());
    sample.values.insert(sample.values.end(), frame.body_position.begin(), frame.body_position.end());
    sample.values.insert(sample.values.end(), frame.root_linear_velocity.begin(), frame.root_linear_velocity.end());
    sample.values.insert(sample.values.end(), frame.root_angular_velocity.begin(), frame.root_angular_velocity.end());
    return sample;
}

inline MotionDataSample EncodeMotionFrameAsMotionData(const MotionFrame &frame) {
    return EncodeMotionFrameAsMotionData(frame, "reference_tracking_v1");
}

inline MotionDataSample EncodeMotionFrameStackAsMotionData(const std::vector<MotionFrame> &frames, const std::string &layout) {
    if (frames.empty()) {
        return {};
    }

    MotionDataSample sample;
    sample.seq      = frames.front().seq;
    sample.stamp_ns = frames.front().stamp_ns;
    sample.valid    = frames.front().valid;
    sample.anchor_quaternion_valid = frames.front().anchor_quaternion_valid;
    sample.anchor_quaternion_wxyz = frames.front().anchor_quaternion_wxyz;

    if (layout != "reference_tracking_v1") {
        return EncodeMotionFrameAsMotionData(frames.front(), layout);
    }

    sample.values.reserve(frames.size() * kMotionDataReferenceTrackingValuesWithAnchor);
    for (const MotionFrame &frame : frames) {
        sample.values.insert(sample.values.end(), frame.joint_position.begin(), frame.joint_position.end());
        sample.values.insert(sample.values.end(), frame.joint_velocity.begin(), frame.joint_velocity.end());
        sample.values.push_back(frame.root_position_z);
        sample.values.insert(sample.values.end(), frame.root_state.begin(), frame.root_state.end());
        sample.values.insert(sample.values.end(), frame.body_position.begin(), frame.body_position.end());
        sample.values.insert(sample.values.end(), frame.root_linear_velocity.begin(), frame.root_linear_velocity.end());
        sample.values.insert(sample.values.end(), frame.root_angular_velocity.begin(), frame.root_angular_velocity.end());
        sample.values.insert(sample.values.end(), frame.anchor_quaternion_wxyz.begin(), frame.anchor_quaternion_wxyz.end());
    }
    return sample;
}

}  // namespace public_inference_module
