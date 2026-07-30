#include "igris_c_sdk/channel_factory.hpp"
#include "igris_c_sdk/igris_c_msgs.hpp"
#include "igris_c_sdk/publisher.hpp"
#include "igris_c_sdk/qos.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

igris_c::msg::dds::Header BuildHeader(uint32_t seq, const char *frame_id) {
  igris_c::msg::dds::Header header;
  header.seq(seq);

  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  header.sec(static_cast<uint32_t>(ns / 1000000000ULL));
  header.nanosec(static_cast<uint32_t>(ns % 1000000000ULL));

  std::array<char, 256> frame_id_array{};
  for (std::size_t i = 0; i < frame_id_array.size() - 1 && frame_id[i] != '\0';
       ++i) {
    frame_id_array[i] = frame_id[i];
  }
  header.frame_id(frame_id_array);
  return header;
}

std::string ReadFile(const std::string &path) {
  if (path.empty()) {
    return "";
  }

  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open CycloneDDS XML file: " + path);
  }
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

int DefaultDomainId() {
  const char *env = std::getenv("ROS_DOMAIN_ID");
  if (env == nullptr || *env == '\0') {
    return 0;
  }
  char *end = nullptr;
  const long value = std::strtol(env, &end, 10);
  if (end == env || *end != '\0') {
    return 0;
  }
  return static_cast<int>(value);
}

class GmrControlModeDdsBridge : public rclcpp::Node {
public:
  GmrControlModeDdsBridge()
      : rclcpp::Node("gmr_control_mode_dds_bridge") {
    const std::string lowcmd_start_topic =
        declare_parameter<std::string>("lowcmd_start_topic",
                                       "/gmr/lowcmd_start");
    const std::string stop_topic =
        declare_parameter<std::string>("stop_topic", "/gmr/stop");
    const std::string go_home_topic =
        declare_parameter<std::string>("go_home_topic", "/gmr/go_home");
    const std::string request_topic = declare_parameter<std::string>(
        "dds_request_topic", "rt/service/control_mode/request");
    const int dds_domain_id =
        declare_parameter<int>("dds_domain_id", DefaultDomainId());
    const std::string robot_namespace =
        declare_parameter<std::string>("robot_namespace", "");
    const std::string cyclonedds_xml_path =
        declare_parameter<std::string>("cyclonedds_xml_path", "");
    const int qos_depth = declare_parameter<int>("qos_depth", 10);

    igris_c_sdk::ChannelFactory::Instance()->Init(
        dds_domain_id, robot_namespace, ReadFile(cyclonedds_xml_path));
    if (!igris_c_sdk::ChannelFactory::Instance()->IsInitialized()) {
      throw std::runtime_error("failed to initialize DDS ChannelFactory");
    }

    control_mode_pub_ = std::make_unique<
        igris_c_sdk::Publisher<igris_c::msg::dds::ControlModeCommandRequest>>(
        request_topic, igris_c_sdk::QosProfile::Services());
    if (!control_mode_pub_->init()) {
      throw std::runtime_error("failed to initialize control mode DDS publisher");
    }

    auto event_qos = rclcpp::SensorDataQoS();
    event_qos.keep_last(qos_depth);
    lowcmd_start_sub_ = create_subscription<std_msgs::msg::Bool>(
        lowcmd_start_topic, event_qos,
        [this](const std_msgs::msg::Bool &msg) {
          if (msg.data) {
            publishControlModeRequest(
                "lowcmd_start",
                igris_c::msg::dds::ControlModeCommandType::
                    CONTROL_MODE_CMD_LOW_LEVEL_JOINT_CONTROL);
          }
        });
    stop_sub_ = create_subscription<std_msgs::msg::Bool>(
        stop_topic, event_qos, [this](const std_msgs::msg::Bool &msg) {
          if (msg.data) {
            publishControlModeRequest(
                "stop",
                igris_c::msg::dds::ControlModeCommandType::
                    CONTROL_MODE_CMD_JOINT_POSITION_HOLD);
          }
        });
    go_home_sub_ = create_subscription<std_msgs::msg::Bool>(
        go_home_topic, event_qos, [this](const std_msgs::msg::Bool &msg) {
          if (msg.data) {
            publishControlModeRequest(
                "go_home",
                igris_c::msg::dds::ControlModeCommandType::
                    CONTROL_MODE_CMD_MOTION_PRESET,
                "HOME");
          }
        });

    RCLCPP_INFO(get_logger(),
                "listening lowcmd_start=%s stop=%s go_home=%s; DDS request "
                "topic=%s domain_id=%d namespace='%s'",
                lowcmd_start_topic.c_str(), stop_topic.c_str(),
                go_home_topic.c_str(),
                igris_c_sdk::ChannelFactory::Instance()
                    ->resolve(request_topic)
                    .c_str(),
                dds_domain_id, robot_namespace.c_str());
  }

private:
  void publishControlModeRequest(
      const std::string &label,
      igris_c::msg::dds::ControlModeCommandType command_type,
      const std::string &preset_id = "") {
    if (!control_mode_pub_) {
      RCLCPP_ERROR(get_logger(), "control mode DDS publisher is not initialized");
      return;
    }

    const uint32_t seq = ++request_seq_;
    igris_c::msg::dds::ControlModeCommandRequest request;
    request.header(BuildHeader(seq, "gmr_control_mode_dds_bridge"));
    request.request_id("gmr_control_mode_dds_bridge_" + label + "_" +
                       std::to_string(seq));
    request.command_type(command_type);
    request.preset_id(preset_id);
    request.is_cyclic(false);

    if (!control_mode_pub_->write(request)) {
      RCLCPP_ERROR(get_logger(),
                   "failed to publish %s control mode DDS request: "
                   "request_id=%s",
                   label.c_str(), request.request_id().c_str());
      return;
    }

    RCLCPP_INFO(get_logger(),
                "published %s control mode DDS request: request_id=%s "
                "command_type=%u preset_id='%s'",
                label.c_str(), request.request_id().c_str(),
                static_cast<unsigned>(command_type), preset_id.c_str());
  }

  uint32_t request_seq_ = 0;
  std::unique_ptr<
      igris_c_sdk::Publisher<igris_c::msg::dds::ControlModeCommandRequest>>
      control_mode_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr lowcmd_start_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr stop_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr go_home_sub_;
};

} // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<GmrControlModeDdsBridge>());
  } catch (const std::exception &exception) {
    std::cerr << exception.what() << "\n";
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
