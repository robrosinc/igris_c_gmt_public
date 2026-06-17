#pragma once

#include "public_inference_module/config.hpp"
#include "public_inference_module/types.hpp"

#include <memory>

namespace public_inference_module {

class MotionFrameSource {
  public:
    virtual ~MotionFrameSource() = default;

    virtual bool getLatest(MotionFrame &frame) = 0;
    virtual void reset()                       = 0;
};

std::unique_ptr<MotionFrameSource> CreateMotionFrameSource(const InferenceConfig &config);

}  // namespace public_inference_module
