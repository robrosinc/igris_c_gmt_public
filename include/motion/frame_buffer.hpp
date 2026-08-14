#pragma once

#include "core/types.hpp"

#include <atomic>
#include <algorithm>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace igris_c_gmt_public {

class MotionFrameBuffer {
public:
  void write(MotionFrame frame) {
    frame.valid = true;
    auto snapshot = std::make_shared<const MotionFrame>(std::move(frame));
    {
      std::lock_guard<std::mutex> lock(history_mutex_);
      history_.push_back(*snapshot);
      while (history_.size() > max_history_length_) {
        history_.pop_front();
      }
    }
    std::atomic_store_explicit(&latest_frame_, snapshot,
                               std::memory_order_release);
  }

  std::shared_ptr<const MotionFrame> readLatest() const {
    return std::atomic_load_explicit(&latest_frame_, std::memory_order_acquire);
  }

  bool readOffsetStack(const std::vector<int> &offsets,
                       std::vector<MotionFrame> &frames) const {
    frames.clear();
    if (offsets.empty()) {
      return false;
    }

    const int max_offset = std::max(0, *std::max_element(offsets.begin(),
                                                        offsets.end()));
    std::lock_guard<std::mutex> lock(history_mutex_);
    if (history_.size() <= static_cast<std::size_t>(max_offset)) {
      return false;
    }

    const int anchor_index =
        static_cast<int>(history_.size() - 1) - max_offset;
    frames.reserve(offsets.size());
    for (int offset : offsets) {
      const int frame_index = anchor_index + offset;
      if (frame_index < 0 ||
          static_cast<std::size_t>(frame_index) >= history_.size()) {
        frames.clear();
        return false;
      }
      frames.push_back(history_[static_cast<std::size_t>(frame_index)]);
    }
    return true;
  }

  void clearHistory() {
    {
      std::lock_guard<std::mutex> lock(history_mutex_);
      history_.clear();
    }
    std::shared_ptr<const MotionFrame> empty;
    std::atomic_store_explicit(&latest_frame_, empty,
                               std::memory_order_release);
  }

  void keepLatestOnly() {
    const std::shared_ptr<const MotionFrame> latest = readLatest();
    std::lock_guard<std::mutex> lock(history_mutex_);
    history_.clear();
    if (latest && latest->valid) {
      history_.push_back(*latest);
    }
  }

private:
  static constexpr std::size_t max_history_length_ = 512;
  mutable std::mutex history_mutex_;
  std::deque<MotionFrame> history_;
  mutable std::shared_ptr<const MotionFrame> latest_frame_;
};

} // namespace igris_c_gmt_public
