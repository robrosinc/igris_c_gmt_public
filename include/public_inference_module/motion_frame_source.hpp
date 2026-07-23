#pragma once

#include "public_inference_module/config.hpp"
#include "public_inference_module/types.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace public_inference_module {

class MotionFrameSource {
  public:
    virtual ~MotionFrameSource() = default;

    virtual bool getLatest(MotionFrame &frame) = 0;
    virtual bool getLatestStack(std::vector<MotionFrame> &frames, std::size_t length);
    virtual bool getFrameAtStep(std::size_t step, MotionFrame &frame);
    virtual bool getFrameStackAtStep(std::size_t step, std::vector<MotionFrame> &frames, std::size_t length);
    virtual void reset()                       = 0;
};

std::unique_ptr<MotionFrameSource> CreateMotionFrameSource(const InferenceConfig &config);

}  // namespace public_inference_module
