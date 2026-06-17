#include "public_inference_module/ros_motion_receiver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace public_inference_module {
namespace {

bool IsNumericChar(char value) {
    return (value >= '0' && value <= '9') || value == '-' || value == '+' || value == '.' || value == 'e' || value == 'E';
}

bool ExtractArrayNumbers(const std::string &payload, const std::string &key, std::vector<double> &values) {
    values.clear();
    const std::size_t key_pos = payload.find(key);
    if (key_pos == std::string::npos) {
        return false;
    }
    const std::size_t array_start = payload.find('[', key_pos);
    if (array_start == std::string::npos) {
        return false;
    }

    int depth = 0;
    std::size_t array_end = std::string::npos;
    for (std::size_t i = array_start; i < payload.size(); ++i) {
        if (payload[i] == '[') {
            ++depth;
        } else if (payload[i] == ']') {
            --depth;
            if (depth == 0) {
                array_end = i;
                break;
            }
        }
    }
    if (array_end == std::string::npos) {
        return false;
    }

    std::string token;
    for (std::size_t i = array_start + 1; i < array_end; ++i) {
        const char value = payload[i];
        if (IsNumericChar(value)) {
            token.push_back(value);
        } else if (!token.empty()) {
            try {
                values.push_back(std::stod(token));
            } catch (const std::exception &) {
            }
            token.clear();
        }
    }
    if (!token.empty()) {
        try {
            values.push_back(std::stod(token));
        } catch (const std::exception &) {
        }
    }
    return !values.empty();
}

bool ExtractRootHeight(const std::string &payload, double &height) {
    std::vector<double> values;
    if (ExtractArrayNumbers(payload, "root_height", values) && !values.empty()) {
        height = values.front();
        return true;
    }

    const std::size_t key_pos = payload.find("root_height");
    if (key_pos == std::string::npos) {
        return false;
    }
    const std::size_t colon_pos = payload.find(':', key_pos);
    if (colon_pos == std::string::npos) {
        return false;
    }

    std::string token;
    for (std::size_t i = colon_pos + 1; i < payload.size(); ++i) {
        const char value = payload[i];
        if (IsNumericChar(value)) {
            token.push_back(value);
        } else if (!token.empty()) {
            break;
        }
    }
    if (token.empty()) {
        return false;
    }

    try {
        height = std::stod(token);
    } catch (const std::exception &) {
        return false;
    }
    return true;
}

std::array<double, kRlNumReferenceBodyPositionObs> MapBodyPosition(const std::vector<double> &values, bool &valid) {
    static constexpr std::array<int, kRlReferenceKeybodyCount> kTopicGoalKeybodyIndices = {0, 23, 31, 9, 15, 19, 27, 16, 24, 7, 13, 4, 10, 33};

    valid = false;
    std::array<double, kRlNumReferenceBodyPositionObs> body_position{};
    if (values.size() == static_cast<std::size_t>(kRlNumReferenceBodyPositionObs)) {
        std::copy(values.begin(), values.end(), body_position.begin());
        valid = true;
        return body_position;
    }
    if (values.empty() || (values.size() % 3) != 0) {
        return body_position;
    }

    const int source_keybody_count = static_cast<int>(values.size() / 3);
    const bool source_is_rootless  = source_keybody_count == (kRlReferenceSourceKeybodyCount - 1);
    for (int i = 0; i < kRlReferenceKeybodyCount; ++i) {
        int source_index = kTopicGoalKeybodyIndices[static_cast<std::size_t>(i)];
        if (source_is_rootless) {
            source_index = (source_index == 0) ? -1 : (source_index - 1);
        }
        if (source_index < 0 || source_index >= source_keybody_count) {
            continue;
        }
        const std::size_t source_offset = static_cast<std::size_t>(source_index * 3);
        const std::size_t target_offset = static_cast<std::size_t>(i * 3);
        body_position[target_offset]     = values[source_offset];
        body_position[target_offset + 1] = values[source_offset + 1];
        body_position[target_offset + 2] = values[source_offset + 2];
        valid = true;
    }
    return body_position;
}

std::array<double, kRlNumJointActions> MapJointVector(const std::vector<double> &values, bool &valid) {
    static constexpr std::array<int, kRlNumJointActions> kPolicyToControlIndex = {
        15, 19, 0, 16, 20, 1, 17, 21, 2, 18, 22, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14,
    };
    static constexpr std::array<int, kRlNumJointActions> kPolicyToSystemJointIndex = {
        TM1_LSP, TM1_RSP, TM1_WY,  TM1_LSR, TM1_RSR, TM1_WR,  TM1_LSY, TM1_RSY, TM1_WP,  TM1_LEP, TM1_REP, TM1_LHP,
        TM1_RHP, TM1_LHR, TM1_RHR, TM1_LHY, TM1_RHY, TM1_LKP, TM1_RKP, TM1_LAP, TM1_RAP, TM1_LAR, TM1_RAR,
    };

    valid = false;
    std::array<double, kRlNumJointActions> mapped{};
    if (values.size() == static_cast<std::size_t>(kRlNumJointActions)) {
        for (int i = 0; i < kRlNumJointActions; ++i) {
            mapped[static_cast<std::size_t>(i)] = values[static_cast<std::size_t>(kPolicyToControlIndex[static_cast<std::size_t>(i)])];
        }
        valid = true;
        return mapped;
    }
    if (values.size() >= static_cast<std::size_t>(kModelDof)) {
        for (int i = 0; i < kRlNumJointActions; ++i) {
            mapped[static_cast<std::size_t>(i)] = values[static_cast<std::size_t>(kPolicyToSystemJointIndex[static_cast<std::size_t>(i)])];
        }
        valid = true;
    }
    return mapped;
}

MotionDataSample DecodeRetargetFramePayload(const std::string &payload, uint64_t seq, uint64_t stamp_ns) {
    MotionFrame frame;
    frame.seq      = seq;
    frame.stamp_ns = stamp_ns;
    frame.valid    = true;

    bool has_body = false;
    std::vector<double> body_values;
    has_body = ExtractArrayNumbers(payload, "keybody_pos_local", body_values);
    if (!has_body) {
        has_body = ExtractArrayNumbers(payload, "keybody_pos_world", body_values);
    }
    if (has_body) {
        bool valid = false;
        frame.body_position = MapBodyPosition(body_values, valid);
        has_body = valid;
    }

    bool position_valid = false;
    std::vector<double> dof_pos_values;
    if (ExtractArrayNumbers(payload, "dof_pos", dof_pos_values)) {
        frame.joint_position = MapJointVector(dof_pos_values, position_valid);
    }

    bool velocity_valid = false;
    std::vector<double> dof_vel_values;
    if (ExtractArrayNumbers(payload, "dof_vel", dof_vel_values)) {
        frame.joint_velocity = MapJointVector(dof_vel_values, velocity_valid);
    }

    std::vector<double> root_position_values;
    if (ExtractArrayNumbers(payload, "root_pos", root_position_values) && root_position_values.size() >= 3) {
        frame.root_position_z = root_position_values[2];
    } else {
        double root_height = 0.0;
        if (ExtractRootHeight(payload, root_height)) {
            frame.root_position_z = root_height;
        } else if (has_body && frame.body_position.size() >= 3) {
            frame.root_position_z = frame.body_position[2];
        }
    }

    std::vector<double> root_rotation_values;
    if (ExtractArrayNumbers(payload, "root_rot", root_rotation_values) && root_rotation_values.size() >= 4) {
        frame.anchor_quaternion_wxyz = {root_rotation_values[0], root_rotation_values[1], root_rotation_values[2], root_rotation_values[3]};
        frame.anchor_quaternion_valid = true;
    }

    std::vector<double> root_velocity_values;
    if (ExtractArrayNumbers(payload, "root_vel", root_velocity_values) && root_velocity_values.size() >= 3) {
        frame.root_linear_velocity = {root_velocity_values[0], root_velocity_values[1], root_velocity_values[2]};
    }

    std::vector<double> root_angular_velocity_values;
    if (ExtractArrayNumbers(payload, "root_angvel", root_angular_velocity_values) && root_angular_velocity_values.size() >= 3) {
        frame.root_angular_velocity = {root_angular_velocity_values[0], root_angular_velocity_values[1], root_angular_velocity_values[2]};
    }

    if (!has_body && !(position_valid && velocity_valid)) {
        return {};
    }
    return EncodeMotionFrameAsMotionData(frame, "reference_tracking_v1");
}

}  // namespace

RosMotionReceiver::RosMotionReceiver(std::shared_ptr<MotionDataBuffer> buffer) : buffer_(std::move(buffer)) {}

RosMotionReceiver::~RosMotionReceiver() { stop(); }

bool RosMotionReceiver::start(const RosMotionConfig &config) {
    if (running_.load(std::memory_order_acquire)) {
        return true;
    }

    config_ = config;

    if (!rclcpp::contexts::get_global_default_context()->is_valid()) {
        rclcpp::InitOptions options;
        options.set_domain_id(config_.domain_id);
        rclcpp::init(0, nullptr, options);
    }

    node_ = rclcpp::Node::make_shared(config_.node_name);

    auto qos = rclcpp::SensorDataQoS();
    qos.keep_last(config_.qos_depth);
    qos.reliability(config_.best_effort ? RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT : RMW_QOS_POLICY_RELIABILITY_RELIABLE);

    subscription_ = node_->create_subscription<std_msgs::msg::String>(
        config_.topic_name, qos, [this](const std_msgs::msg::String &msg) { callback(msg); });

    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);

    running_.store(true, std::memory_order_release);
    spin_thread_ = std::thread([this] { executor_->spin(); });

    return true;
}

void RosMotionReceiver::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    running_.store(false, std::memory_order_release);
    if (executor_) {
        executor_->cancel();
    }
    if (spin_thread_.joinable()) {
        spin_thread_.join();
    }
    if (executor_ && node_) {
        executor_->remove_node(node_);
    }
    executor_.reset();
    subscription_.reset();
    node_.reset();
}

std::shared_ptr<const MotionDataSample> RosMotionReceiver::readLatest() const { return buffer_ ? buffer_->readLatest() : nullptr; }

void RosMotionReceiver::callback(const std_msgs::msg::String &msg) {
    if (!buffer_) {
        return;
    }

    const uint64_t seq      = ++seq_;
    const uint64_t stamp_ns = nowSteadyNs();
    MotionDataSample sample = DecodeRetargetFramePayload(msg.data, seq, stamp_ns);
    if (!sample.valid) {
        return;
    }
    buffer_->write(std::move(sample));
}

uint64_t RosMotionReceiver::nowSteadyNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

}  // namespace public_inference_module
