#include "motion/ros_receiver.hpp"

#include "motion/frame_buffer.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace igris_c_gmt_public {
namespace {

bool IsNumericChar(char value) {
  return (value >= '0' && value <= '9') || value == '-' || value == '+' ||
         value == '.' || value == 'e' || value == 'E';
}

bool IsIdentifierChar(char value) {
  return (value >= '0' && value <= '9') || (value >= 'A' && value <= 'Z') ||
         (value >= 'a' && value <= 'z') || value == '_';
}

std::size_t FindPayloadKey(const std::string &payload, const std::string &key) {
  std::size_t key_pos = payload.find(key);
  while (key_pos != std::string::npos) {
    const bool valid_prefix =
        key_pos == 0 || !IsIdentifierChar(payload[key_pos - 1]);
    const std::size_t suffix_pos = key_pos + key.size();
    const bool valid_suffix =
        suffix_pos >= payload.size() || !IsIdentifierChar(payload[suffix_pos]);
    if (valid_prefix && valid_suffix) {
      return key_pos;
    }
    key_pos = payload.find(key, key_pos + 1);
  }
  return std::string::npos;
}

bool ExtractArrayNumbers(const std::string &payload, const std::string &key,
                         std::vector<double> &values) {
  values.clear();
  const std::size_t key_pos = FindPayloadKey(payload, key);
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

bool ExtractStringArray(const std::string &payload, const std::string &key,
                        std::vector<std::string> &values) {
  values.clear();
  const std::size_t key_pos = FindPayloadKey(payload, key);
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

  bool in_string = false;
  char quote = '\0';
  std::string token;
  for (std::size_t i = array_start + 1; i < array_end; ++i) {
    const char value = payload[i];
    if (!in_string && (value == '\'' || value == '"')) {
      in_string = true;
      quote = value;
      token.clear();
      continue;
    }
    if (in_string && value == quote) {
      values.push_back(token);
      in_string = false;
      quote = '\0';
      token.clear();
      continue;
    }
    if (in_string && value == '\\' && i + 1 < array_end) {
      token.push_back(payload[++i]);
      continue;
    }
    if (in_string) {
      token.push_back(value);
    }
  }
  return !values.empty();
}

Eigen::Matrix3d RotationMatrixFromWxyz(const std::array<double, 4> &quat_wxyz) {
  Eigen::Quaterniond quaternion(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2],
                                quat_wxyz[3]);
  quaternion.normalize();
  return quaternion.toRotationMatrix();
}

std::array<double, 3>
RotateWorldVectorToAnchorFrame(const std::array<double, 4> &quat_wxyz,
                               const std::vector<double> &values) {
  const Eigen::Vector3d world_value(values[0], values[1], values[2]);
  const Eigen::Vector3d local_value =
      RotationMatrixFromWxyz(quat_wxyz).transpose() * world_value;
  return {local_value(0), local_value(1), local_value(2)};
}

bool NormalizeGmrBodyData(MotionFrame &frame) {
  if (frame.body_names.empty() || frame.body_position.empty() ||
      (frame.body_position.size() % 3) != 0) {
    return false;
  }

  const std::size_t body_count = frame.body_position.size() / 3;
  if (frame.body_names.size() != body_count) {
    return false;
  }
  if (!frame.body_orientation.empty() &&
      frame.body_orientation.size() != body_count * 4) {
    return false;
  }
  if (!frame.body_linear_velocity.empty() &&
      frame.body_linear_velocity.size() != body_count * 3) {
    return false;
  }
  if (!frame.body_angular_velocity.empty() &&
      frame.body_angular_velocity.size() != body_count * 3) {
    return false;
  }

  std::unordered_set<std::string> seen_names;
  for (const std::string &body_name : frame.body_names) {
    if (!seen_names.insert(body_name).second) {
      return false;
    }
  }

  auto base_it =
      std::find(frame.body_names.begin(), frame.body_names.end(), "base_link");
  if (base_it == frame.body_names.end()) {
    frame.body_names.insert(frame.body_names.begin(), "base_link");
    frame.body_position.insert(frame.body_position.begin(), {0.0, 0.0, 0.0});
    if (!frame.body_orientation.empty()) {
      frame.body_orientation.insert(frame.body_orientation.begin(),
                                    {1.0, 0.0, 0.0, 0.0});
    }
    if (!frame.body_linear_velocity.empty()) {
      frame.body_linear_velocity.insert(frame.body_linear_velocity.begin(),
                                        {frame.anchor_linear_velocity_b[0],
                                         frame.anchor_linear_velocity_b[1],
                                         frame.anchor_linear_velocity_b[2]});
    }
    if (!frame.body_angular_velocity.empty()) {
      frame.body_angular_velocity.insert(frame.body_angular_velocity.begin(),
                                         {frame.anchor_angular_velocity_b[0],
                                          frame.anchor_angular_velocity_b[1],
                                          frame.anchor_angular_velocity_b[2]});
    }
    return true;
  }

  const std::size_t base_index = static_cast<std::size_t>(
      std::distance(frame.body_names.begin(), base_it));
  const std::size_t position_offset = base_index * 3;
  frame.body_position[position_offset] = 0.0;
  frame.body_position[position_offset + 1] = 0.0;
  frame.body_position[position_offset + 2] = 0.0;

  if (!frame.body_orientation.empty()) {
    const std::size_t orientation_offset = base_index * 4;
    frame.body_orientation[orientation_offset] = 1.0;
    frame.body_orientation[orientation_offset + 1] = 0.0;
    frame.body_orientation[orientation_offset + 2] = 0.0;
    frame.body_orientation[orientation_offset + 3] = 0.0;
  }
  if (!frame.body_linear_velocity.empty()) {
    const std::size_t velocity_offset = base_index * 3;
    frame.body_linear_velocity[velocity_offset] =
        frame.anchor_linear_velocity_b[0];
    frame.body_linear_velocity[velocity_offset + 1] =
        frame.anchor_linear_velocity_b[1];
    frame.body_linear_velocity[velocity_offset + 2] =
        frame.anchor_linear_velocity_b[2];
  }
  if (!frame.body_angular_velocity.empty()) {
    const std::size_t velocity_offset = base_index * 3;
    frame.body_angular_velocity[velocity_offset] =
        frame.anchor_angular_velocity_b[0];
    frame.body_angular_velocity[velocity_offset + 1] =
        frame.anchor_angular_velocity_b[1];
    frame.body_angular_velocity[velocity_offset + 2] =
        frame.anchor_angular_velocity_b[2];
  }
  return true;
}

std::array<double, kRlNumJointActions>
MapJointVector(const std::vector<double> &values, bool &valid) {
  static constexpr std::array<int, kRlNumJointActions> kPolicyToControlIndex = {
      15, 19, 0,  16, 20, 1, 17, 21, 2,  18, 22, 3,
      9,  4,  10, 5,  11, 6, 12, 7,  13, 8,  14,
  };
  valid = false;
  std::array<double, kRlNumJointActions> mapped{};
  if (values.size() == static_cast<std::size_t>(kRlNumJointActions)) {
    for (int i = 0; i < kRlNumJointActions; ++i) {
      mapped[static_cast<std::size_t>(i)] = values[static_cast<std::size_t>(
          kPolicyToControlIndex[static_cast<std::size_t>(i)])];
    }
    valid = true;
    return mapped;
  }
  if (values.size() >= static_cast<std::size_t>(kModelDof)) {
    for (int i = 0; i < kRlNumJointActions; ++i) {
      mapped[static_cast<std::size_t>(i)] = values[static_cast<std::size_t>(
          kObsToSystemJointMapping[static_cast<std::size_t>(i)])];
    }
    valid = true;
  }
  return mapped;
}

MotionFrame DecodeRetargetFramePayload(const std::string &payload,
                                       uint64_t seq) {
  MotionFrame frame;
  frame.seq = seq;

  const bool has_body_names =
      ExtractStringArray(payload, "link_body_list", frame.body_names);
  bool has_body =
      ExtractArrayNumbers(payload, "keybody_pos_local", frame.body_position);
  has_body = has_body && !frame.body_position.empty() &&
             (frame.body_position.size() % 3) == 0;
  std::vector<double> body_orientation_values;
  if (ExtractArrayNumbers(payload, "keybody_rot_local",
                          body_orientation_values)) {
    frame.body_orientation = std::move(body_orientation_values);
  }
  std::vector<double> body_linear_velocity_values;
  if (ExtractArrayNumbers(payload, "keybody_lin_vel_local",
                          body_linear_velocity_values) ||
      ExtractArrayNumbers(payload, "body_lin_vel_b",
                          body_linear_velocity_values)) {
    frame.body_linear_velocity = std::move(body_linear_velocity_values);
  }
  std::vector<double> body_angular_velocity_values;
  if (ExtractArrayNumbers(payload, "keybody_ang_vel_local",
                          body_angular_velocity_values) ||
      ExtractArrayNumbers(payload, "body_ang_vel_b",
                          body_angular_velocity_values)) {
    frame.body_angular_velocity = std::move(body_angular_velocity_values);
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

  bool anchor_position_valid = false;
  std::vector<double> anchor_position_values;
  if (ExtractArrayNumbers(payload, "root_pos", anchor_position_values) &&
      anchor_position_values.size() >= 3) {
    frame.anchor_position = {anchor_position_values[0],
                             anchor_position_values[1],
                             anchor_position_values[2]};
    anchor_position_valid = true;
  }

  std::vector<double> anchor_rotation_values;
  if (ExtractArrayNumbers(payload, "root_rot", anchor_rotation_values) &&
      anchor_rotation_values.size() >= 4) {
    frame.anchor_quaternion_wxyz = {
        anchor_rotation_values[0], anchor_rotation_values[1],
        anchor_rotation_values[2], anchor_rotation_values[3]};
    frame.anchor_quaternion_valid = true;
  }

  bool anchor_linear_velocity_valid = false;
  std::vector<double> anchor_linear_velocity_values;
  if (ExtractArrayNumbers(payload, "root_vel", anchor_linear_velocity_values) &&
      anchor_linear_velocity_values.size() >= 3 &&
      frame.anchor_quaternion_valid) {
    frame.anchor_linear_velocity = {anchor_linear_velocity_values[0],
                                    anchor_linear_velocity_values[1],
                                    anchor_linear_velocity_values[2]};
    frame.anchor_linear_velocity_b = RotateWorldVectorToAnchorFrame(
        frame.anchor_quaternion_wxyz, anchor_linear_velocity_values);
    anchor_linear_velocity_valid = true;
  }

  bool anchor_angular_velocity_valid = false;
  std::vector<double> anchor_angular_velocity_values;
  if (ExtractArrayNumbers(payload, "root_angvel",
                          anchor_angular_velocity_values) &&
      anchor_angular_velocity_values.size() >= 3 &&
      frame.anchor_quaternion_valid) {
    frame.anchor_angular_velocity = {anchor_angular_velocity_values[0],
                                     anchor_angular_velocity_values[1],
                                     anchor_angular_velocity_values[2]};
    frame.anchor_angular_velocity_b = RotateWorldVectorToAnchorFrame(
        frame.anchor_quaternion_wxyz, anchor_angular_velocity_values);
    anchor_angular_velocity_valid = true;
  }

  const bool body_data_valid =
      has_body_names && has_body && NormalizeGmrBodyData(frame);

  frame.valid = body_data_valid && position_valid && velocity_valid &&
                anchor_position_valid && frame.anchor_quaternion_valid &&
                anchor_linear_velocity_valid && anchor_angular_velocity_valid;
  return frame;
}

} // namespace

RosMotionReceiver::RosMotionReceiver(std::shared_ptr<MotionFrameBuffer> buffer)
    : buffer_(std::move(buffer)) {}

RosMotionReceiver::~RosMotionReceiver() { stop(); }

bool RosMotionReceiver::start(const RosMotionConfig &config) {
  if (running_.load(std::memory_order_acquire)) {
    return true;
  }

  config_ = config;

  rclcpp::InitOptions init_options;
  init_options.set_domain_id(config_.domain_id);
  context_ = std::make_shared<rclcpp::Context>();
  context_->init(0, nullptr, init_options);

  rclcpp::NodeOptions node_options;
  node_options.context(context_);
  node_ = rclcpp::Node::make_shared(config_.node_name, node_options);

  auto qos = rclcpp::SensorDataQoS();
  qos.keep_last(config_.qos_depth);
  qos.reliability(config_.best_effort ? RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
                                      : RMW_QOS_POLICY_RELIABILITY_RELIABLE);

  if (!config_.recorded_reference_topic_name.empty()) {
    mode_subscription_ = node_->create_subscription<std_msgs::msg::Bool>(
        config_.recorded_reference_topic_name, qos,
        [this](const std_msgs::msg::Bool &msg) { modeCallback(msg); });
  }

  subscription_ = node_->create_subscription<std_msgs::msg::String>(
      config_.topic_name, qos,
      [this](const std_msgs::msg::String &msg) { callback(msg); });

  rclcpp::ExecutorOptions executor_options;
  executor_options.context = context_;
  executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>(
      executor_options);
  executor_->add_node(node_);

  running_.store(true, std::memory_order_release);
  spin_thread_ = std::thread([this] { executor_->spin(); });

  return true;
}

void RosMotionReceiver::stop() {
  const bool was_running = running_.exchange(false, std::memory_order_acq_rel);
  if (!was_running && !executor_ && !node_ && !context_) {
    return;
  }

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
  mode_subscription_.reset();
  subscription_.reset();
  node_.reset();
  if (context_ && context_->is_valid()) {
    context_->shutdown("RosMotionReceiver stopped");
  }
  context_.reset();
}

std::shared_ptr<const MotionFrame> RosMotionReceiver::readLatest() const {
  return buffer_ ? buffer_->readLatest() : nullptr;
}

bool RosMotionReceiver::useRecordedReference() const {
  return use_recorded_reference_.load(std::memory_order_acquire);
}

void RosMotionReceiver::modeCallback(const std_msgs::msg::Bool &msg) {
  use_recorded_reference_.store(msg.data, std::memory_order_release);
}

void RosMotionReceiver::callback(const std_msgs::msg::String &msg) {
  if (!buffer_) {
    return;
  }

  const uint64_t seq = ++seq_;
  MotionFrame frame = DecodeRetargetFramePayload(msg.data, seq);
  if (!frame.valid) {
    return;
  }
  buffer_->write(std::move(frame));
}

} // namespace igris_c_gmt_public
