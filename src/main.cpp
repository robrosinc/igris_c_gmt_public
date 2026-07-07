#include "public_inference_module/config.hpp"
#include "public_inference_module/inference_module.hpp"
#include "public_inference_module/motion_data.hpp"
#include "public_inference_module/motion_frame_source.hpp"
#include "public_inference_module/robot_io.hpp"
#include "public_inference_module/ros_motion_receiver.hpp"

#include "igris_c_sdk/namespace_resolver.hpp"

#include <rclcpp/parameter_client.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_running{true};

void HandleSignal(int) { g_running.store(false, std::memory_order_relaxed); }

void PrintUsage(const char *program) {
    std::cout << "Usage: " << program << " --config <path> [--domain-id N] [--namespace NS] [--cyclonedds-xml PATH]\n";
}

bool ParseInt(const char *value, int &parsed) {
    char *end = nullptr;
    const long number = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return false;
    }
    parsed = static_cast<int>(number);
    return true;
}

void ResolveRobotNetworkSettings(public_inference_module::InferenceConfig &config, bool override_domain_id, bool override_namespace) {
    if (!rclcpp::contexts::get_global_default_context()->is_valid()) {
        rclcpp::init(0, nullptr);
    }

    auto node = rclcpp::Node::make_shared("public_inference_module_config_resolver");
    const std::string param_server = node->declare_parameter<std::string>("param_server", "igris_c_param_server");
    auto param_client = std::make_shared<rclcpp::SyncParametersClient>(node, param_server);

    std::string suffix_policy = "ap_suffix";
    std::string user_suffix;
    const bool has_param_server = param_client->wait_for_service(std::chrono::seconds(1));
    if (has_param_server) {
        if (!override_domain_id) {
            config.domain_id = param_client->get_parameter<int>("igris_c.network.cyclonedds.domain_id", config.domain_id);
        }
        suffix_policy = param_client->get_parameter<std::string>("igris_c.network.cyclonedds.namespace.suffix_policy", suffix_policy);
        user_suffix   = param_client->get_parameter<std::string>("igris_c.network.cyclonedds.namespace.user_suffix", user_suffix);
    }

    if (!override_namespace && config.robot_namespace.empty()) {
        config.robot_namespace = igris_c_sdk::resolve_robot_namespace(suffix_policy, user_suffix);
    }
}

}  // namespace

int main(int argc, char **argv) {
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    std::string config_path;
    bool override_domain_id         = false;
    int domain_id_override          = 0;
    bool override_namespace         = false;
    std::string namespace_override;
    bool override_cyclonedds_xml    = false;
    std::string cyclonedds_xml_override;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        }
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
            continue;
        }
        if (arg == "--domain-id" && i + 1 < argc) {
            override_domain_id = ParseInt(argv[++i], domain_id_override);
            if (!override_domain_id) {
                std::cerr << "Invalid --domain-id value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--namespace" && i + 1 < argc) {
            override_namespace = true;
            namespace_override = argv[++i];
            continue;
        }
        if (arg == "--cyclonedds-xml" && i + 1 < argc) {
            override_cyclonedds_xml = true;
            cyclonedds_xml_override = argv[++i];
            continue;
        }

        std::cerr << "Unexpected argument: " << arg << "\n";
        PrintUsage(argv[0]);
        return 1;
    }

    if (config_path.empty()) {
        PrintUsage(argv[0]);
        return 1;
    }

    try {
        public_inference_module::InferenceConfig config = public_inference_module::LoadInferenceConfig(config_path);
        if (override_domain_id) {
            config.domain_id = domain_id_override;
        }
        if (override_namespace) {
            config.robot_namespace = namespace_override;
        }
        if (override_cyclonedds_xml) {
            config.cyclonedds_xml_path = cyclonedds_xml_override;
        }
        ResolveRobotNetworkSettings(config, override_domain_id, override_namespace);

        public_inference_module::RobotIo robot_io;
        if (!robot_io.initialize(config.domain_id, config.robot_namespace, config.cyclonedds_xml_path)) {
            std::cerr << "Failed to initialize robot I/O\n";
            return 1;
        }

        public_inference_module::InferenceModule module;
        module.initialize();
        std::cerr << "Loading inference config\n";
        if (module.loadConfig(config) != 0) {
            std::cerr << "Failed to load inference module config\n";
            return 1;
        }
        std::cerr << "Inference config loaded\n";

        std::unique_ptr<public_inference_module::MotionFrameSource> motion_source;
        std::unique_ptr<public_inference_module::MotionFrameSource> recorded_motion_source;
        std::unique_ptr<public_inference_module::RosMotionReceiver> ros_motion_receiver;
        std::shared_ptr<public_inference_module::MotionDataBuffer> ros_motion_buffer;
        if (config.motion_source.type == "ros2") {
            if (config.motion_source.ros_config_path.empty()) {
                throw std::runtime_error("motion_source.ros_config_path must be set when motion_source.type=ros2");
            }
            const public_inference_module::RosMotionConfig ros_config =
                public_inference_module::LoadRosMotionConfig(config.motion_source.ros_config_path);
            ros_motion_buffer = std::make_shared<public_inference_module::MotionDataBuffer>();
            ros_motion_receiver = std::make_unique<public_inference_module::RosMotionReceiver>(ros_motion_buffer);
            std::cerr << "Starting ROS motion receiver\n";
            if (!ros_motion_receiver->start(ros_config)) {
                throw std::runtime_error("Failed to start ROS 2 motion receiver");
            }
            std::cerr << "ROS motion receiver started\n";
            if (!config.motion_source.onnx_path.empty()) {
                public_inference_module::InferenceConfig recorded_config = config;
                recorded_config.motion_source.type = "onnx_replay";
                recorded_motion_source = public_inference_module::CreateMotionFrameSource(recorded_config);
                std::cerr << "Recorded reference source loaded\n";
            }
        } else {
            motion_source = public_inference_module::CreateMotionFrameSource(config);
        }

        std::cout << "Waiting for first LowState on DDS topic rt/lowstate..." << std::endl;
        while (g_running.load(std::memory_order_relaxed) &&
               !robot_io.waitForFirstState(std::chrono::milliseconds(100))) {
        }
        if (!g_running.load(std::memory_order_relaxed)) {
            return 0;
        }

        const auto control_period = std::chrono::microseconds(static_cast<int>(1000000.0 / static_cast<double>(config.control_hz)));
        auto next_time            = std::chrono::steady_clock::now();

        bool module_ready = false;
        bool live_motion_seen  = false;
        uint64_t last_live_motion_seq = 0;
        auto last_live_motion_wall_time = std::chrono::steady_clock::now();
        auto last_status_print     = std::chrono::steady_clock::now();
        double last_action_compute_ms = 0.0;
        double action_compute_sum_ms  = 0.0;
        double action_compute_max_ms  = 0.0;
        uint64_t action_compute_count = 0;
        double last_onnx_inference_ms = 0.0;
        double onnx_inference_sum_ms  = 0.0;
        double onnx_inference_max_ms  = 0.0;
        uint64_t onnx_inference_count = 0;
        uint64_t last_policy_inference_count = module.policyInferenceCount();
        std::string active_motion_mode = "waiting";

        while (g_running.load(std::memory_order_relaxed)) {
            public_inference_module::InferenceCommand command;
            igris_c::msg::dds::LowState state;
            if (!robot_io.snapshotState(state)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            const auto now = std::chrono::steady_clock::now();
            auto read_recorded_motion_sample = [&](bool restart) -> std::shared_ptr<const public_inference_module::MotionDataSample> {
                if (!recorded_motion_source) {
                    return nullptr;
                }
                if (restart) {
                    recorded_motion_source->reset();
                }
                std::vector<public_inference_module::MotionFrame> motion_frames;
                if (!recorded_motion_source->getLatestStack(motion_frames, public_inference_module::kRlReferenceFrameStackLength)) {
                    return nullptr;
                }
                module.getMotionDataBuffer()->write(
                    public_inference_module::EncodeMotionFrameStackAsMotionData(motion_frames, config.motion_source.layout));
                return module.readMotionDataSample();
            };

            std::shared_ptr<const public_inference_module::MotionDataSample> motion_data_sample;
            bool motion_fresh = false;
            std::string motion_mode = "waiting";
            if (ros_motion_receiver) {
                if (recorded_motion_source && ros_motion_receiver->useRecordedReference()) {
                    motion_data_sample = read_recorded_motion_sample(active_motion_mode != "recorded");
                    motion_fresh = motion_data_sample && motion_data_sample->valid;
                    motion_mode = motion_fresh ? "recorded" : "waiting";
                } else {
                    motion_data_sample = ros_motion_receiver->readLatest();
                    if (motion_data_sample && motion_data_sample->valid) {
                        if (!live_motion_seen || motion_data_sample->seq != last_live_motion_seq) {
                            live_motion_seen           = true;
                            last_live_motion_seq       = motion_data_sample->seq;
                            last_live_motion_wall_time = now;
                        }
                        const auto live_age_ms =
                            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_live_motion_wall_time).count();
                        motion_fresh = static_cast<double>(live_age_ms) <= config.motion_frame_timeout_ms;
                        motion_mode = motion_fresh ? "live" : "waiting";
                    }
                }
            } else if (motion_source) {
                std::vector<public_inference_module::MotionFrame> motion_frames;
                if (motion_source->getLatestStack(motion_frames, public_inference_module::kRlReferenceFrameStackLength)) {
                    module.getMotionDataBuffer()->write(
                        public_inference_module::EncodeMotionFrameStackAsMotionData(motion_frames, config.motion_source.layout));
                    motion_data_sample = module.readMotionDataSample();
                    motion_fresh = motion_data_sample && motion_data_sample->valid;
                    motion_mode = motion_fresh ? config.motion_source.type : "waiting";
                }
            }
            active_motion_mode = motion_mode;

            const auto action_compute_start = std::chrono::steady_clock::now();
            bool publish_command = false;
            if (motion_fresh) {
                if (!module_ready) {
                    if (module.reset(state, motion_data_sample.get()) == 0) {
                        module_ready = true;
                    }
                }

                if (module_ready && module.compute(state, motion_data_sample.get(), command) != 0) {
                    module.buildHoldCommand(state, command);
                    module_ready = false;
                }
                publish_command = module_ready;
            } else {
                module_ready = false;
            }
            last_action_compute_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - action_compute_start).count();
            action_compute_sum_ms += last_action_compute_ms;
            if (last_action_compute_ms > action_compute_max_ms) {
                action_compute_max_ms = last_action_compute_ms;
            }
            ++action_compute_count;

            const uint64_t policy_inference_count = module.policyInferenceCount();
            if (policy_inference_count != last_policy_inference_count) {
                last_onnx_inference_ms = module.lastPolicyInferenceMs();
                onnx_inference_sum_ms += last_onnx_inference_ms;
                if (last_onnx_inference_ms > onnx_inference_max_ms) {
                    onnx_inference_max_ms = last_onnx_inference_ms;
                }
                onnx_inference_count += policy_inference_count - last_policy_inference_count;
                last_policy_inference_count = policy_inference_count;
            }

            if (publish_command) {
                if (!robot_io.publish(command)) {
                    std::cerr << "Failed to publish LowCmd\n";
                    return 1;
                }
            }

            if (now - last_status_print >= std::chrono::seconds(1)) {
                const double avg_action_compute_ms =
                    action_compute_count > 0 ? action_compute_sum_ms / static_cast<double>(action_compute_count) : 0.0;
                std::cout << std::fixed << std::setprecision(3)
                          << "control_tick=" << (module_ready ? "policy" : "waiting")
                          << " motion_mode=" << motion_mode
                          << " motion_fresh=" << (motion_fresh ? "Y" : "N")
                          << " action_compute_ms=" << last_action_compute_ms
                          << " avg_action_compute_ms=" << avg_action_compute_ms
                          << " max_action_compute_ms=" << action_compute_max_ms;
                if (onnx_inference_count > 0) {
                    const double avg_onnx_inference_ms =
                        onnx_inference_sum_ms / static_cast<double>(onnx_inference_count);
                    std::cout << " onnx_inference_ms=" << last_onnx_inference_ms
                              << " avg_onnx_inference_ms=" << avg_onnx_inference_ms
                              << " max_onnx_inference_ms=" << onnx_inference_max_ms
                              << " onnx_inference_count=" << onnx_inference_count;
                } else {
                    std::cout << " onnx_inference_ms=N/A"
                              << " avg_onnx_inference_ms=N/A"
                              << " max_onnx_inference_ms=N/A"
                              << " onnx_inference_count=0";
                }
                std::cout << std::endl;
                last_status_print = now;
                action_compute_sum_ms  = 0.0;
                action_compute_max_ms  = 0.0;
                action_compute_count   = 0;
                onnx_inference_sum_ms  = 0.0;
                onnx_inference_max_ms  = 0.0;
                onnx_inference_count   = 0;
            }

            next_time += control_period;
            std::this_thread::sleep_until(next_time);
        }

        if (ros_motion_receiver) {
            ros_motion_receiver->stop();
        }
        if (rclcpp::contexts::get_global_default_context()->is_valid()) {
            rclcpp::shutdown();
        }
    } catch (const std::exception &exception) {
        if (rclcpp::contexts::get_global_default_context()->is_valid()) {
            rclcpp::shutdown();
        }
        std::cerr << exception.what() << "\n";
        return 1;
    }

    return 0;
}
