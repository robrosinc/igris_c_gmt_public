#pragma once

#include "public_inference_module/types.hpp"

#include "igris_c_sdk/igris_c_msgs.hpp"
#include "igris_c_sdk/publisher.hpp"
#include "igris_c_sdk/subscriber.hpp"

#include <condition_variable>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>

namespace public_inference_module {

class RobotIo {
  public:
    bool initialize(int domain_id, const std::string &robot_namespace, const std::string &cyclonedds_xml_path);
    bool waitForFirstState(std::chrono::milliseconds timeout);
    bool snapshotState(igris_c::msg::dds::LowState &state) const;
    bool publish(const InferenceCommand &command);

  private:
    void lowStateCallback(const igris_c::msg::dds::LowState &state);
    static std::string loadCycloneConfig(const std::string &xml_path);

  private:
    mutable std::mutex state_mutex_;
    std::condition_variable state_cv_;
    igris_c::msg::dds::LowState latest_state_;
    bool has_state_ = false;
    uint32_t publish_seq_ = 0;

    std::unique_ptr<igris_c_sdk::Subscriber<igris_c::msg::dds::LowState>> lowstate_sub_;
    std::unique_ptr<igris_c_sdk::Publisher<igris_c::msg::dds::LowCmd>> lowcmd_pub_;
};

}  // namespace public_inference_module
