#include "public_inference_module/config.hpp"
#include "public_inference_module/inference_module.hpp"
#include "public_inference_module/motion_data.hpp"
#include "public_inference_module/motion_frame_source.hpp"
#include "public_inference_module/robot_io.hpp"
#include "public_inference_module/ros_motion_receiver.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

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
        std::unique_ptr<public_inference_module::RosMotionReceiver> ros_motion_receiver;
        if (config.motion_source.type == "ros2") {
            if (config.motion_source.ros_config_path.empty()) {
                throw std::runtime_error("motion_source.ros_config_path must be set when motion_source.type=ros2");
            }
            const public_inference_module::RosMotionConfig ros_config =
                public_inference_module::LoadRosMotionConfig(config.motion_source.ros_config_path);
            ros_motion_receiver = std::make_unique<public_inference_module::RosMotionReceiver>(module.getMotionDataBuffer());
            std::cerr << "Starting ROS motion receiver\n";
            if (!ros_motion_receiver->start(ros_config)) {
                throw std::runtime_error("Failed to start ROS 2 motion receiver");
            }
            std::cerr << "ROS motion receiver started\n";
        } else {
            motion_source = public_inference_module::CreateMotionFrameSource(config);
        }

        std::cout << "Waiting for first LowState..." << std::endl;
        while (g_running.load(std::memory_order_relaxed) &&
               !robot_io.waitForFirstState(std::chrono::milliseconds(100))) {
        }
        if (!g_running.load(std::memory_order_relaxed)) {
            return 0;
        }

        const auto control_period = std::chrono::microseconds(static_cast<int>(1000000.0 / static_cast<double>(config.control_hz)));
        auto next_time            = std::chrono::steady_clock::now();

        bool module_ready = false;
        bool motion_seen  = false;
        uint64_t last_motion_seq = 0;
        auto last_motion_wall_time = std::chrono::steady_clock::now();
        auto last_status_print     = std::chrono::steady_clock::now();

        while (g_running.load(std::memory_order_relaxed)) {
            public_inference_module::InferenceCommand command;
            igris_c::msg::dds::LowState state;
            if (!robot_io.snapshotState(state)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            std::shared_ptr<const public_inference_module::MotionDataSample> motion_data_sample;
            if (ros_motion_receiver) {
                motion_data_sample = module.readMotionDataSample();
            } else if (motion_source) {
                public_inference_module::MotionFrame motion_frame;
                if (motion_source->getLatest(motion_frame)) {
                    module.getMotionDataBuffer()->write(
                        public_inference_module::EncodeMotionFrameAsMotionData(motion_frame, config.motion_source.layout));
                    motion_data_sample = module.readMotionDataSample();
                }
            }
            const auto now       = std::chrono::steady_clock::now();

            bool motion_fresh = false;
            if (motion_data_sample && motion_data_sample->valid) {
                if (!motion_seen || motion_data_sample->seq != last_motion_seq) {
                    motion_seen           = true;
                    last_motion_seq       = motion_data_sample->seq;
                    last_motion_wall_time = now;
                }
                const auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_motion_wall_time).count();
                motion_fresh      = static_cast<double>(age_ms) <= config.motion_frame_timeout_ms;
            }

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
            } else {
                module.buildHoldCommand(state, command);
                module_ready = false;
            }

            if (!robot_io.publish(command)) {
                std::cerr << "Failed to publish LowCmd\n";
                return 1;
            }

            if (now - last_status_print >= std::chrono::seconds(1)) {
                std::cout << "control_tick=" << (module_ready ? "policy" : "hold") << " motion_fresh=" << (motion_fresh ? "Y" : "N") << std::endl;
                last_status_print = now;
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
