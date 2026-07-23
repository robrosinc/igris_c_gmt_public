#pragma once

#include "core/types.hpp"

#include <atomic>
#include <memory>
#include <utility>

namespace igris_c_gmt_public {

class MotionFrameBuffer {
public:
  void write(MotionFrame frame) {
    frame.valid = true;
    auto snapshot = std::make_shared<const MotionFrame>(std::move(frame));
    std::atomic_store_explicit(&latest_frame_, snapshot,
                               std::memory_order_release);
  }

  std::shared_ptr<const MotionFrame> readLatest() const {
    return std::atomic_load_explicit(&latest_frame_, std::memory_order_acquire);
  }

private:
  mutable std::shared_ptr<const MotionFrame> latest_frame_;
};

} // namespace igris_c_gmt_public
