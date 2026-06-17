#include "public_inference_module/robot_io.hpp"

#include "igris_c_sdk/channel_factory.hpp"
#include "igris_c_sdk/qos.hpp"

#include <array>
#include <chrono>
#include <fstream>
#include <stdexcept>

namespace public_inference_module {
namespace {

igris_c::msg::dds::Header BuildHeader(uint32_t seq, const char *frame_id) {
    igris_c::msg::dds::Header header;
    header.seq(seq);
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto ns  = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    header.sec(static_cast<uint32_t>(ns / 1000000000ULL));
    header.nanosec(static_cast<uint32_t>(ns % 1000000000ULL));

    std::array<char, 256> frame_id_array{};
    for (std::size_t i = 0; i < frame_id_array.size() - 1 && frame_id[i] != '\0'; ++i) {
        frame_id_array[i] = frame_id[i];
    }
    header.frame_id(frame_id_array);
    return header;
}

}  // namespace

bool RobotIo::initialize(int domain_id, const std::string &robot_namespace, const std::string &cyclonedds_xml_path) {
    const std::string cyclonedds_xml = loadCycloneConfig(cyclonedds_xml_path);
    igris_c_sdk::ChannelFactory::Instance()->Init(domain_id, robot_namespace, cyclonedds_xml);
    if (!igris_c_sdk::ChannelFactory::Instance()->IsInitialized()) {
        return false;
    }

    lowstate_sub_ = std::make_unique<igris_c_sdk::Subscriber<igris_c::msg::dds::LowState>>("rt/lowstate",
                                                                                             igris_c_sdk::QosProfile::SensorData());
    if (!lowstate_sub_->init([this](const igris_c::msg::dds::LowState &state) { lowStateCallback(state); })) {
        return false;
    }

    lowcmd_pub_ = std::make_unique<igris_c_sdk::Publisher<igris_c::msg::dds::LowCmd>>("rt/lowcmd",
                                                                                        igris_c_sdk::QosProfile::SensorData());
    return lowcmd_pub_->init();
}

bool RobotIo::waitForFirstState(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(state_mutex_);
    return state_cv_.wait_for(lock, timeout, [this] { return has_state_; });
}

bool RobotIo::snapshotState(igris_c::msg::dds::LowState &state) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!has_state_) {
        return false;
    }
    state = latest_state_;
    return true;
}

bool RobotIo::publish(const InferenceCommand &command) {
    if (!lowcmd_pub_) {
        return false;
    }

    igris_c::msg::dds::LowCmd low_cmd;
    low_cmd.header(BuildHeader(++publish_seq_, "public_inference_module"));
    for (std::size_t i = 0; i < command.kinematic_modes.size(); ++i) {
        low_cmd.kinematic_modes()[i] = command.kinematic_modes[i];
    }
    for (int i = 0; i < kModelDof; ++i) {
        auto &motor = low_cmd.motors()[static_cast<std::size_t>(i)];
        motor.id(static_cast<uint16_t>(i));
        motor.q(static_cast<float>(command.q(i)));
        motor.dq(static_cast<float>(command.q_dot(i)));
        motor.tau(static_cast<float>(command.tau(i)));
        motor.kp(static_cast<float>(command.kp(i)));
        motor.kd(static_cast<float>(command.kd(i)));
    }
    return lowcmd_pub_->write(low_cmd);
}

void RobotIo::lowStateCallback(const igris_c::msg::dds::LowState &state) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        latest_state_ = state;
        has_state_    = true;
    }
    state_cv_.notify_all();
}

std::string RobotIo::loadCycloneConfig(const std::string &xml_path) {
    if (xml_path.empty()) {
        return "";
    }

    std::ifstream file(xml_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CycloneDDS XML file: " + xml_path);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

}  // namespace public_inference_module
