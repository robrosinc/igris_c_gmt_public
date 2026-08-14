#pragma once

#include "core/config.hpp"
#include "core/types.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace igris_c_gmt_public {

class MotionFrameSource {
public:
  virtual ~MotionFrameSource() = default;

  virtual bool getLatest(MotionFrame &frame) = 0;
  virtual bool getLatestStack(std::vector<MotionFrame> &frames,
                              std::size_t length);
  virtual bool getFrameAtStep(std::size_t step, MotionFrame &frame);
  virtual bool getFrameStackAtStep(std::size_t step,
                                   std::vector<MotionFrame> &frames,
                                   std::size_t length);
  virtual bool getFrameStackAtStep(std::size_t step,
                                   std::vector<MotionFrame> &frames,
                                   const std::vector<int> &offsets);
  virtual void reset() = 0;
};

std::unique_ptr<MotionFrameSource>
CreateMotionFrameSource(const InferenceConfig &config);

} // namespace igris_c_gmt_public
