#include "igris_c_sdk/channel_factory.hpp"
#include "igris_c_sdk/igris_c_client.hpp"
#include "igris_c_sdk/igris_c_msgs.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

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
    const int dds_domain_id =
        declare_parameter<int>("dds_domain_id", DefaultDomainId());
    const std::string robot_namespace =
        declare_parameter<std::string>("robot_namespace", "");
    const std::string cyclonedds_xml_path =
        declare_parameter<std::string>("cyclonedds_xml_path", "");
    const int qos_depth = declare_parameter<int>("qos_depth", 10);
    service_timeout_ms_ = declare_parameter<int>("service_timeout_ms", 3000);

    igris_c_sdk::ChannelFactory::Instance()->Init(
        dds_domain_id, robot_namespace, ReadFile(cyclonedds_xml_path));
    if (!igris_c_sdk::ChannelFactory::Instance()->IsInitialized()) {
      throw std::runtime_error("failed to initialize DDS ChannelFactory");
    }

    client_.Init();

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
                "topic=%s response_topic=%s domain_id=%d namespace='%s' "
                "service_timeout_ms=%d",
                lowcmd_start_topic.c_str(), stop_topic.c_str(),
                go_home_topic.c_str(),
                igris_c_sdk::ChannelFactory::Instance()
                    ->resolve("rt/service/control_mode/request")
                    .c_str(),
                igris_c_sdk::ChannelFactory::Instance()
                    ->resolve("rt/service/control_mode/response")
                    .c_str(),
                dds_domain_id, robot_namespace.c_str(), service_timeout_ms_);
  }

private:
  void publishControlModeRequest(
      const std::string &label,
      igris_c::msg::dds::ControlModeCommandType command_type,
      const std::string &preset_id = "") {
    const auto response = client_.SendControlModeCommand(
        command_type, preset_id, false, service_timeout_ms_);
    if (!response.success()) {
      RCLCPP_ERROR(get_logger(),
                   "%s control mode request failed: request_id=%s "
                   "error_code=%d message='%s'",
                   label.c_str(), response.request_id().c_str(),
                   response.error_code(), response.message().c_str());
      return;
    }

    RCLCPP_INFO(get_logger(),
                "%s control mode request accepted: request_id=%s message='%s'",
                label.c_str(), response.request_id().c_str(),
                response.message().c_str());
  }

  int service_timeout_ms_ = 3000;
  igris_c_sdk::IgrisC_Client client_;
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
