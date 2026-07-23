#include "public_inference_module/config.hpp"
#include "public_inference_module/inference_module.hpp"
#include "public_inference_module/motion_data.hpp"
#include "public_inference_module/motion_frame_source.hpp"
#include "public_inference_module/robot_io.hpp"
#include "public_inference_module/ros_motion_receiver.hpp"

#include "igris_c_sdk/namespace_resolver.hpp"

#include <rclcpp/parameter_client.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

std::atomic<bool> g_running{true};
constexpr const char *kDefaultPlotRoot = "/home/robros/workspace/src/public_inference_module/log/plot";

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

struct StateSample {
    igris_c::msg::dds::LowState state;
    uint64_t sequence = 0;
    bool valid        = false;
};

class StateBuffer {
  public:
    void write(const igris_c::msg::dds::LowState &state, uint64_t sequence) {
        std::lock_guard<std::mutex> lock(mutex_);
        sample_.state    = state;
        sample_.sequence = sequence;
        sample_.valid    = true;
    }

    bool readLatest(StateSample &sample) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!sample_.valid) {
            return false;
        }
        sample = sample_;
        return true;
    }

  private:
    mutable std::mutex mutex_;
    StateSample sample_;
};

struct ActionSample {
    public_inference_module::InferenceCommand command;
    uint64_t sequence    = 0;
    uint64_t motion_step = 0;
    double worker_ms     = 0.0;
    double onnx_ms       = 0.0;
    std::string motion_mode;
    bool valid = false;
};

class ActionBuffer {
  public:
    void write(public_inference_module::InferenceCommand command, uint64_t motion_step, double worker_ms, double onnx_ms,
               std::string motion_mode) {
        std::lock_guard<std::mutex> lock(mutex_);
        sample_.command     = std::move(command);
        sample_.sequence    = ++sequence_;
        sample_.motion_step = motion_step;
        sample_.worker_ms   = worker_ms;
        sample_.onnx_ms     = onnx_ms;
        sample_.motion_mode = std::move(motion_mode);
        sample_.valid       = true;
    }

    bool consume(uint64_t &last_consumed_sequence, ActionSample &sample) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!sample_.valid || sample_.sequence == last_consumed_sequence) {
            return false;
        }
        sample = sample_;
        last_consumed_sequence = sample_.sequence;
        return true;
    }

    uint64_t sequence() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sequence_;
    }

  private:
    mutable std::mutex mutex_;
    ActionSample sample_;
    uint64_t sequence_ = 0;
};

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
            if (!ros_motion_receiver->start(ros_config, config.motion_source.layout)) {
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

        StateBuffer state_buffer;
        ActionBuffer action_buffer;
        std::atomic<bool> worker_running{true};
        std::atomic<bool> worker_failed{false};
        std::mutex worker_error_mutex;
        std::string worker_error;
        auto set_worker_error = [&](const std::string &error) {
            {
                std::lock_guard<std::mutex> lock(worker_error_mutex);
                worker_error = error;
            }
            worker_failed.store(true, std::memory_order_release);
            g_running.store(false, std::memory_order_relaxed);
        };

        {
            igris_c::msg::dds::LowState state;
            uint64_t state_sequence = 0;
            if (robot_io.snapshotState(state, &state_sequence)) {
                state_buffer.write(state, state_sequence);
            }
        }

        const auto policy_period = std::chrono::microseconds(
            std::max<int64_t>(1, static_cast<int64_t>(1000000.0 / static_cast<double>(config.policy_hz))));
        std::thread inference_worker([&]() {
            using Clock = std::chrono::steady_clock;
            try {
                auto next_policy_time = Clock::now();
                bool module_ready = false;
                bool live_motion_seen = false;
                uint64_t last_live_motion_seq = 0;
                auto last_live_motion_wall_time = Clock::now();
                uint64_t last_consumed_state_seq = 0;
                uint64_t motion_step = 0;
                std::string active_motion_mode = "waiting";

                auto transition_motion_mode = [&](const std::string &motion_mode) {
                    if (motion_mode != active_motion_mode) {
                        module_ready = false;
                        active_motion_mode = motion_mode;
                    }
                };

                auto read_replay_motion_sample =
                    [&](public_inference_module::MotionFrameSource &source,
                        uint64_t step) -> std::shared_ptr<const public_inference_module::MotionDataSample> {
                    std::vector<public_inference_module::MotionFrame> motion_frames;
                    if (!source.getFrameStackAtStep(static_cast<std::size_t>(step), motion_frames,
                                                    public_inference_module::kRlReferenceFrameStackLength)) {
                        return nullptr;
                    }
                    module.getMotionDataBuffer()->write(
                        public_inference_module::EncodeMotionFrameStackAsMotionData(motion_frames, config.motion_source.layout));
                    return module.readMotionDataSample();
                };

                while (g_running.load(std::memory_order_relaxed) && worker_running.load(std::memory_order_relaxed)) {
                    const auto now = Clock::now();
                    if (now < next_policy_time) {
                        const auto sleep_duration =
                            std::min(std::chrono::duration_cast<std::chrono::microseconds>(next_policy_time - now),
                                     std::chrono::microseconds(500));
                        std::this_thread::sleep_for(sleep_duration);
                        continue;
                    }

                    StateSample state_sample;
                    if (state_buffer.readLatest(state_sample) && state_sample.sequence != last_consumed_state_seq) {
                        last_consumed_state_seq = state_sample.sequence;

                        std::shared_ptr<const public_inference_module::MotionDataSample> motion_data_sample;
                        bool motion_fresh = false;
                        std::string motion_mode = "waiting";

                        if (ros_motion_receiver) {
                            if (recorded_motion_source && ros_motion_receiver->useRecordedReference()) {
                                if (active_motion_mode != "recorded") {
                                    motion_step = 0;
                                    recorded_motion_source->reset();
                                    transition_motion_mode("recorded");
                                }
                                motion_data_sample = read_replay_motion_sample(*recorded_motion_source, motion_step);
                                motion_fresh = motion_data_sample && motion_data_sample->valid;
                                motion_mode = motion_fresh ? "recorded" : "waiting";
                            } else {
                                motion_data_sample = ros_motion_receiver->readLatest();
                                if (motion_data_sample && motion_data_sample->valid) {
                                    if (!live_motion_seen || motion_data_sample->seq != last_live_motion_seq) {
                                        live_motion_seen = true;
                                        last_live_motion_seq = motion_data_sample->seq;
                                        last_live_motion_wall_time = now;
                                    }
                                    const auto live_age_ms =
                                        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_live_motion_wall_time).count();
                                    motion_fresh = static_cast<double>(live_age_ms) <= config.motion_frame_timeout_ms;
                                    motion_mode = motion_fresh ? "live" : "waiting";
                                }
                            }
                        } else if (motion_source) {
                            if (active_motion_mode != config.motion_source.type) {
                                motion_step = 0;
                                motion_source->reset();
                                transition_motion_mode(config.motion_source.type);
                            }
                            motion_data_sample = read_replay_motion_sample(*motion_source, motion_step);
                            motion_fresh = motion_data_sample && motion_data_sample->valid;
                            motion_mode = motion_fresh ? config.motion_source.type : "waiting";
                        }

                        if (!motion_fresh) {
                            transition_motion_mode("waiting");
                        } else {
                            transition_motion_mode(motion_mode);
                            if (!module_ready && module.reset(state_sample.state, motion_data_sample.get()) != 0) {
                                transition_motion_mode("waiting");
                            } else {
                                module_ready = true;
                                public_inference_module::InferenceCommand command;
                                const auto worker_start = Clock::now();
                                const int rc = module.runPolicyStep(state_sample.state, motion_data_sample.get(), command);
                                const auto worker_end = Clock::now();
                                if (rc == 0) {
                                    const double worker_ms = std::chrono::duration<double, std::milli>(worker_end - worker_start).count();
                                    action_buffer.write(std::move(command), motion_step, worker_ms, module.lastPolicyInferenceMs(), motion_mode);
                                    ++motion_step;
                                } else {
                                    module_ready = false;
                                }
                            }
                        }
                    }

                    next_policy_time += policy_period;
                    const auto after_step = Clock::now();
                    if (next_policy_time <= after_step) {
                        next_policy_time = after_step + policy_period;
                    }
                }
            } catch (const std::exception &exception) {
                set_worker_error(std::string("Inference worker failed: ") + exception.what());
            } catch (...) {
                set_worker_error("Inference worker failed with an unknown exception");
            }
        });

        const auto control_period = std::chrono::microseconds(
            std::max<int64_t>(1, static_cast<int64_t>(1000000.0 / static_cast<double>(config.control_hz))));
        auto next_time = std::chrono::steady_clock::now();
        auto last_status_print = std::chrono::steady_clock::now();
        uint64_t last_state_sequence = 0;
        uint64_t last_action_sequence = 0;
        uint64_t published_action_count = 0;
        uint64_t published_action_window_count = 0;
        uint64_t last_motion_step = 0;
        std::string last_motion_mode = "waiting";
        double last_worker_ms = 0.0;
        double worker_sum_ms = 0.0;
        double worker_max_ms = 0.0;
        double last_onnx_ms = 0.0;
        double onnx_sum_ms = 0.0;
        double onnx_max_ms = 0.0;
        int exit_code = 0;

        while (g_running.load(std::memory_order_relaxed)) {
            if (worker_failed.load(std::memory_order_acquire)) {
                exit_code = 1;
                break;
            }

            igris_c::msg::dds::LowState state;
            uint64_t state_sequence = 0;
            if (robot_io.snapshotState(state, &state_sequence) && state_sequence != last_state_sequence) {
                state_buffer.write(state, state_sequence);
                last_state_sequence = state_sequence;
            }

            ActionSample action_sample;
            if (action_buffer.consume(last_action_sequence, action_sample)) {
                if (!robot_io.publish(action_sample.command)) {
                    std::cerr << "Failed to publish LowCmd\n";
                    exit_code = 1;
                    g_running.store(false, std::memory_order_relaxed);
                    break;
                }
                ++published_action_count;
                ++published_action_window_count;
                last_motion_step = action_sample.motion_step;
                last_motion_mode = action_sample.motion_mode;
                last_worker_ms = action_sample.worker_ms;
                worker_sum_ms += action_sample.worker_ms;
                worker_max_ms = std::max(worker_max_ms, action_sample.worker_ms);
                last_onnx_ms = action_sample.onnx_ms;
                onnx_sum_ms += action_sample.onnx_ms;
                onnx_max_ms = std::max(onnx_max_ms, action_sample.onnx_ms);
            }

            const auto now = std::chrono::steady_clock::now();
            if (now - last_status_print >= std::chrono::seconds(1)) {
                const double avg_worker_ms = published_action_window_count > 0
                                                 ? worker_sum_ms / static_cast<double>(published_action_window_count)
                                                 : 0.0;
                const double avg_onnx_ms = published_action_window_count > 0
                                               ? onnx_sum_ms / static_cast<double>(published_action_window_count)
                                               : 0.0;
                std::cout << std::fixed << std::setprecision(3)
                          << "state_seq=" << last_state_sequence
                          << " action_seq=" << last_action_sequence
                          << " published_actions=" << published_action_count
                          << " motion_mode=" << last_motion_mode
                          << " motion_step=" << last_motion_step;
                if (published_action_window_count > 0) {
                    std::cout << " worker_ms=" << last_worker_ms
                              << " avg_worker_ms=" << avg_worker_ms
                              << " max_worker_ms=" << worker_max_ms
                              << " onnx_inference_ms=" << last_onnx_ms
                              << " avg_onnx_inference_ms=" << avg_onnx_ms
                              << " max_onnx_inference_ms=" << onnx_max_ms
                              << " action_hz=" << published_action_window_count;
                } else {
                    std::cout << " worker_ms=N/A"
                              << " avg_worker_ms=N/A"
                              << " max_worker_ms=N/A"
                              << " onnx_inference_ms=N/A"
                              << " avg_onnx_inference_ms=N/A"
                              << " max_onnx_inference_ms=N/A"
                              << " action_hz=0";
                }
                std::cout << std::endl;
                last_status_print = now;
                published_action_window_count = 0;
                worker_sum_ms = 0.0;
                worker_max_ms = 0.0;
                onnx_sum_ms = 0.0;
                onnx_max_ms = 0.0;
            }

            next_time += control_period;
            const auto after_step = std::chrono::steady_clock::now();
            if (next_time <= after_step) {
                next_time = after_step + control_period;
            }
            std::this_thread::sleep_until(next_time);
        }

        worker_running.store(false, std::memory_order_relaxed);
        if (inference_worker.joinable()) {
            inference_worker.join();
        }

        if (worker_failed.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(worker_error_mutex);
            std::cerr << worker_error << "\n";
            exit_code = 1;
        }

        module.writePlotFiles(kDefaultPlotRoot);

        if (ros_motion_receiver) {
            ros_motion_receiver->stop();
        }
        if (rclcpp::contexts::get_global_default_context()->is_valid()) {
            rclcpp::shutdown();
        }
        if (exit_code != 0) {
            return exit_code;
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
