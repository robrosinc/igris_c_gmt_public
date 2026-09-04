#include "core/config.hpp"
#include "io/robot_io.hpp"
#include "modules/action_builder.h"
#include "modules/motion_handler.h"
#include "modules/obs_builder.h"
#include "modules/onnx_runner.h"
#include "modules/state_handler.h"

#include "igris_c_sdk/namespace_resolver.hpp"

#include <rclcpp/parameter_client.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

std::atomic<bool> g_running{true};

void HandleSignal(int) { g_running.store(false, std::memory_order_relaxed); }

void WriteCsvField(std::ostream &stream, const std::string &value) {
  stream << '"';
  for (const char ch : value) {
    if (ch == '"') {
      stream << "\"\"";
    } else {
      stream << ch;
    }
  }
  stream << '"';
}

uint64_t NowNs() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

class InferenceCsvLogger {
public:
  explicit InferenceCsvLogger(
      const igris_c_gmt_public::LoggingConfig &config) {
    if (!config.enabled) {
      return;
    }
    const std::filesystem::path path(config.csv_path);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    const bool write_header =
        !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
    file_.open(path, std::ios::out | std::ios::app);
    if (!file_) {
      throw std::runtime_error("Failed to open inference CSV log: " +
                               path.string());
    }
    path_ = path.string();
    needs_header_ = write_header;
  }

  bool enabled() const { return file_.is_open(); }
  const std::string &path() const { return path_; }

  void write(uint64_t state_sequence, const std::string &motion_mode,
             uint64_t motion_step,
             const igris_c_gmt_public::ObservationResult &observation,
             const igris_c_gmt_public::Vector23d &raw_actions) {
    if (!file_) {
      return;
    }
    if (needs_header_) {
      writeHeader(observation, raw_actions);
      needs_header_ = false;
    }

    file_ << NowNs() << ',' << state_sequence << ',';
    WriteCsvField(file_, motion_mode);
    file_ << ',' << motion_step;
    for (const auto &group : observation.groups) {
      for (Eigen::Index index = 0; index < group.second.size(); ++index) {
        file_ << ',' << group.second(index);
      }
    }
    for (Eigen::Index index = 0; index < raw_actions.size(); ++index) {
      file_ << ',' << raw_actions(index);
    }
    file_ << '\n';
    file_.flush();
  }

private:
  void writeHeader(const igris_c_gmt_public::ObservationResult &observation,
                   const igris_c_gmt_public::Vector23d &raw_actions) {
    file_ << "timestamp_ns,state_seq,motion_mode,motion_step";
    for (const auto &group : observation.groups) {
      for (Eigen::Index index = 0; index < group.second.size(); ++index) {
        file_ << ",obs_" << group.first << '_' << index;
      }
    }
    for (Eigen::Index index = 0; index < raw_actions.size(); ++index) {
      file_ << ",raw_action_" << index;
    }
    file_ << '\n';
  }

  std::ofstream file_;
  std::string path_;
  bool needs_header_ = false;
};

const char *RobotControlStateName(igris_c::msg::dds::RobotControlState state) {
  using State = igris_c::msg::dds::RobotControlState;
  switch (state) {
  case State::ROBOT_STATE_NOT_READY:
    return "ROBOT_STATE_NOT_READY";
  case State::ROBOT_STATE_IDLE:
    return "ROBOT_STATE_IDLE";
  case State::ROBOT_STATE_STOP:
    return "ROBOT_STATE_STOP";
  case State::ROBOT_STATE_LOW:
    return "ROBOT_STATE_LOW";
  case State::ROBOT_STATE_HIGH_MOTION_ACTIVE:
    return "ROBOT_STATE_HIGH_MOTION_ACTIVE";
  case State::ROBOT_STATE_HIGH_MOTION_STANDBY:
    return "ROBOT_STATE_HIGH_MOTION_STANDBY";
  case State::ROBOT_STATE_WALK_LOW:
    return "ROBOT_STATE_WALK_LOW";
  case State::ROBOT_STATE_WALK_HIGH_MOTION_ACTIVE:
    return "ROBOT_STATE_WALK_HIGH_MOTION_ACTIVE";
  case State::ROBOT_STATE_WALK_HIGH_MOTION_STANDBY:
    return "ROBOT_STATE_WALK_HIGH_MOTION_STANDBY";
  case State::ROBOT_STATE_WHOLE_BODY_CONTROL:
    return "ROBOT_STATE_WHOLE_BODY_CONTROL";
  }
  return "ROBOT_STATE_UNKNOWN";
}

const char *RobotEnvironmentName(igris_c::msg::dds::RobotEnvironment env) {
  using Env = igris_c::msg::dds::RobotEnvironment;
  switch (env) {
  case Env::ROBOT_ENV_REAL:
    return "ROBOT_ENV_REAL";
  case Env::ROBOT_ENV_SIM:
    return "ROBOT_ENV_SIM";
  }
  return "ROBOT_ENV_UNKNOWN";
}

void PrintUsage(const char *program) {
  std::cout << "Usage: " << program
            << " --config <path> [--domain-id N] [--namespace NS] "
               "[--cyclonedds-xml PATH]\n";
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

struct ActionSample {
  igris_c_gmt_public::InferenceCommand command;
  igris_c_gmt_public::Vector23d actual_joint_position =
      igris_c_gmt_public::Vector23d::Zero();
  igris_c_gmt_public::Vector23d reference_joint_position =
      igris_c_gmt_public::Vector23d::Zero();
  uint64_t sequence = 0;
  uint64_t motion_step = 0;
  double worker_ms = 0.0;
  double onnx_ms = 0.0;
  std::string motion_mode;
  bool replay_restarted = false;
  bool replay_active = false;
  bool motion_clip_end = false;
  bool valid = false;
};

class ActionBuffer {
public:
  void write(igris_c_gmt_public::InferenceCommand command, uint64_t motion_step,
             double worker_ms, double onnx_ms, std::string motion_mode,
             const igris_c_gmt_public::Vector23d &actual_joint_position,
             const igris_c_gmt_public::Vector23d &reference_joint_position,
             bool replay_restarted, bool replay_active, bool motion_clip_end) {
    std::lock_guard<std::mutex> lock(mutex_);
    sample_.command = std::move(command);
    sample_.sequence = ++sequence_;
    sample_.motion_step = motion_step;
    sample_.worker_ms = worker_ms;
    sample_.onnx_ms = onnx_ms;
    sample_.motion_mode = std::move(motion_mode);
    sample_.actual_joint_position = actual_joint_position;
    sample_.reference_joint_position = reference_joint_position;
    sample_.replay_restarted = replay_restarted;
    sample_.replay_active = replay_active;
    sample_.motion_clip_end = motion_clip_end;
    sample_.valid = true;
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

private:
  mutable std::mutex mutex_;
  ActionSample sample_;
  uint64_t sequence_ = 0;
};

class MotionMpjpeLogger {
public:
  explicit MotionMpjpeLogger(const igris_c_gmt_public::InferenceConfig &config) {
    if (!config.logging.mpjpe_enabled) {
      return;
    }
    const std::filesystem::path path(config.logging.mpjpe_txt_path);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    file_.open(path, std::ios::out | std::ios::app);
    if (!file_) {
      throw std::runtime_error("Failed to open MPJPE log: " + path.string());
    }
    path_ = path.string();
    const std::string &motion_path = config.motion_source.onnx_path.empty()
                                         ? config.motion_source.csv_path
                                         : config.motion_source.onnx_path;
    motion_name_ = std::filesystem::path(motion_path).filename().string();
  }

  bool enabled() const { return file_.is_open(); }
  const std::string &path() const { return path_; }

  void write(const ActionSample &sample, uint64_t frame_count,
             double mpjpe_rad) {
    if (!file_) {
      return;
    }
    file_ << "timestamp_ns=" << NowNs() << " motion_mode="
          << sample.motion_mode << " motion=" << motion_name_
          << " motion_end_step=" << sample.motion_step
          << " frames=" << frame_count
          << " joints=" << igris_c_gmt_public::kRlNumJointActions
          << " joint_position_mpjpe_rad=" << std::fixed
          << std::setprecision(8) << mpjpe_rad << '\n';
    file_.flush();
  }

private:
  std::ofstream file_;
  std::string path_;
  std::string motion_name_;
};

class MotionMpjpeTracker {
public:
  explicit MotionMpjpeTracker(MotionMpjpeLogger &logger) : logger_(logger) {}

  void start() {
    active_ = true;
    accumulated_joint_error_rad_ = 0.0;
    frame_count_ = 0;
  }

  void stop() { active_ = false; }

  void add(const ActionSample &sample) {
    if (!active_ || !logger_.enabled() || !sample.replay_active) {
      return;
    }

    accumulated_joint_error_rad_ +=
        (sample.actual_joint_position - sample.reference_joint_position)
            .cwiseAbs()
            .sum();
    ++frame_count_;

    if (sample.motion_clip_end) {
      const double mpjpe_rad =
          accumulated_joint_error_rad_ /
          static_cast<double>(frame_count_ *
                              igris_c_gmt_public::kRlNumJointActions);
      logger_.write(sample, frame_count_, mpjpe_rad);
      std::cerr << "Recorded motion MPJPE: " << mpjpe_rad << " rad over "
                << frame_count_ << " frames\n";
      active_ = false;
    }
  }

private:
  MotionMpjpeLogger &logger_;
  bool active_ = false;
  uint64_t frame_count_ = 0;
  double accumulated_joint_error_rad_ = 0.0;
};

void ResolveRobotNetworkSettings(igris_c_gmt_public::InferenceConfig &config,
                                 bool override_domain_id,
                                 bool override_namespace) {
  if (!rclcpp::contexts::get_global_default_context()->is_valid()) {
    rclcpp::init(0, nullptr);
  }

  auto node = rclcpp::Node::make_shared("igris_c_gmt_public_config_resolver");
  const std::string param_server = node->declare_parameter<std::string>(
      "param_server", "igris_c_param_server");
  auto param_client =
      std::make_shared<rclcpp::SyncParametersClient>(node, param_server);

  std::string suffix_policy = "ap_suffix";
  std::string user_suffix;
  const bool has_param_server =
      param_client->wait_for_service(std::chrono::seconds(1));
  if (has_param_server) {
    if (!override_domain_id) {
      config.domain_id = param_client->get_parameter<int>(
          "igris_c.network.cyclonedds.domain_id", config.domain_id);
    }
    suffix_policy = param_client->get_parameter<std::string>(
        "igris_c.network.cyclonedds.namespace.suffix_policy", suffix_policy);
    user_suffix = param_client->get_parameter<std::string>(
        "igris_c.network.cyclonedds.namespace.user_suffix", user_suffix);
  }

  if (!override_namespace && config.robot_namespace.empty()) {
    config.robot_namespace =
        igris_c_sdk::resolve_robot_namespace(suffix_policy, user_suffix);
  }
}

} // namespace

int main(int argc, char **argv) {
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  std::string config_path;
  bool override_domain_id = false;
  int domain_id_override = 0;
  bool override_namespace = false;
  std::string namespace_override;
  bool override_cyclonedds_xml = false;
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
    igris_c_gmt_public::InferenceConfig config =
        igris_c_gmt_public::LoadInferenceConfig(config_path);
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

    igris_c_gmt_public::RobotIo robot_io;
    if (!robot_io.initialize(config.domain_id, config.robot_namespace,
                             config.cyclonedds_xml_path)) {
      std::cerr << "Failed to initialize robot I/O\n";
      return 1;
    }

    igris_c_gmt_public::StateHandler state_handler;
    state_handler.configure(config);

    igris_c_gmt_public::ObservationBuilder observation_builder;
    if (observation_builder.configure(config) != 0) {
      std::cerr << "Failed to load observation builder config\n";
      return 1;
    }

    igris_c_gmt_public::OnnxRunner onnx_runner("IgrisCGmtPublicOnnxRunner");
    if (onnx_runner.configure(config.policy_onnx_path) != 0) {
      std::cerr << "Failed to load ONNX runner config\n";
      return 1;
    }

    igris_c_gmt_public::ActionBuilder action_builder;
    if (action_builder.configure(config) != 0) {
      std::cerr << "Failed to load action builder config\n";
      return 1;
    }
    std::cerr << "Modular inference framework loaded\n";

    igris_c_gmt_public::MotionHandler motion_handler;
    if (motion_handler.configure(config) != 0) {
      std::cerr << "Failed to configure motion handler\n";
      return 1;
    }

    InferenceCsvLogger csv_logger(config.logging);
    if (csv_logger.enabled()) {
      std::cerr << "Inference CSV logging enabled path='"
                << csv_logger.path()
                << "' only_low_command_mode="
                << (config.logging.only_low_command_mode ? "true" : "false")
                << "\n";
    }
    MotionMpjpeLogger mpjpe_logger(config);
    if (mpjpe_logger.enabled()) {
      std::cerr << "Motion MPJPE logging enabled path='"
                << mpjpe_logger.path() << "'\n";
    }

    std::cout << "Waiting for first LowState on DDS topic rt/lowstate..."
              << std::endl;
    while (g_running.load(std::memory_order_relaxed) &&
           !robot_io.waitForFirstState(std::chrono::milliseconds(100))) {
    }
    if (!g_running.load(std::memory_order_relaxed)) {
      return 0;
    }

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
        state_handler.update(state, state_sequence);
      }
    }

    const auto policy_period = std::chrono::microseconds(std::max<int64_t>(
        1, static_cast<int64_t>(1000000.0 /
                                static_cast<double>(config.policy_hz))));
    std::thread inference_worker([&]() {
      using Clock = std::chrono::steady_clock;
      try {
        auto next_policy_time = Clock::now();
        bool pipeline_ready = false;
        uint64_t last_consumed_state_seq = 0;

        while (g_running.load(std::memory_order_relaxed) &&
               worker_running.load(std::memory_order_relaxed)) {
          const auto now = Clock::now();
          if (now < next_policy_time) {
            const auto sleep_duration =
                std::min(std::chrono::duration_cast<std::chrono::microseconds>(
                             next_policy_time - now),
                         std::chrono::microseconds(500));
            std::this_thread::sleep_for(sleep_duration);
            continue;
          }

          igris_c_gmt_public::ProcessedState state_sample;
          if (state_handler.readLatest(state_sample) &&
              state_sample.sequence != last_consumed_state_seq) {
            last_consumed_state_seq = state_sample.sequence;

            igris_c_gmt_public::MotionHandlerOutput motion_sample;
            if (!motion_handler.read(motion_sample, now)) {
              if (motion_sample.mode_changed) {
                pipeline_ready = false;
                observation_builder.reset();
                action_builder.reset();
              }
            } else {
              if (!pipeline_ready || motion_sample.mode_changed) {
                observation_builder.reset();
                action_builder.reset();
                pipeline_ready = true;
              }

              igris_c_gmt_public::ObservationInput observation_input;
              observation_input.state = state_sample;
              observation_input.motion_frames = std::move(motion_sample.frames);
              observation_input.motion_frame_offsets =
                  std::move(motion_sample.frame_offsets);
              observation_input.q_default = observation_builder.qDefault();
              observation_input.last_actions = action_builder.lastActions();

              igris_c_gmt_public::ObservationResult observation;
              igris_c_gmt_public::Vector23d raw_actions =
                  igris_c_gmt_public::Vector23d::Zero();
              igris_c_gmt_public::InferenceCommand command;
              const auto worker_start = Clock::now();
              const int obs_rc =
                  observation_builder.build(observation_input, observation);
              const int onnx_rc =
                  (obs_rc == 0)
                      ? onnx_runner.run(observation.groups, raw_actions)
                      : -1;
              const int action_rc =
                  (onnx_rc == 0)
                      ? action_builder.build(state_sample, observation,
                                             raw_actions, command)
                      : -1;
              const auto worker_end = Clock::now();
              if (obs_rc == 0 && onnx_rc == 0 && action_rc == 0) {
                const double worker_ms =
                    std::chrono::duration<double, std::milli>(worker_end -
                                                              worker_start)
                        .count();
                const auto robot_mode = robot_io.snapshotRobotMode();
                if (!config.logging.only_low_command_mode ||
                    (robot_mode.valid && robot_mode.low_command_mode)) {
                  csv_logger.write(state_sample.sequence, motion_sample.mode,
                                   motion_sample.step, observation,
                                   raw_actions);
                }
                action_buffer.write(std::move(command), motion_sample.step,
                                    worker_ms, onnx_runner.lastInferenceMs(),
                                    motion_sample.mode,
                                    state_sample.rl_joint_position,
                                    observation.motion_joint_position,
                                    motion_sample.replay_restarted,
                                    motion_sample.replay_active,
                                    motion_sample.clip_end);
                motion_handler.advance();
              } else {
                pipeline_ready = false;
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
        set_worker_error(std::string("Inference worker failed: ") +
                         exception.what());
      } catch (...) {
        set_worker_error("Inference worker failed with an unknown exception");
      }
    });

    const auto control_period = std::chrono::microseconds(std::max<int64_t>(
        1, static_cast<int64_t>(1000000.0 /
                                static_cast<double>(config.control_hz))));
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
    bool was_low_command_mode = false;
    bool awaiting_replay_restart = false;
    MotionMpjpeTracker mpjpe_tracker(mpjpe_logger);

    while (g_running.load(std::memory_order_relaxed)) {
      if (worker_failed.load(std::memory_order_acquire)) {
        exit_code = 1;
        break;
      }

      igris_c::msg::dds::LowState state;
      uint64_t state_sequence = 0;
      if (robot_io.snapshotState(state, &state_sequence) &&
          state_sequence != last_state_sequence) {
        state_handler.update(state, state_sequence);
        last_state_sequence = state_sequence;
      }

      const auto robot_mode = robot_io.snapshotRobotMode();
      const bool low_command_mode =
          robot_mode.valid && robot_mode.low_command_mode;
      if (low_command_mode && !was_low_command_mode) {
        motion_handler.requestReplayReset();
        mpjpe_tracker.start();
        awaiting_replay_restart = true;
      } else if (!low_command_mode && was_low_command_mode) {
        mpjpe_tracker.stop();
        awaiting_replay_restart = false;
      }
      was_low_command_mode = low_command_mode;

      ActionSample action_sample;
      if (action_buffer.consume(last_action_sequence, action_sample)) {
        if (awaiting_replay_restart && action_sample.replay_active) {
          if (!action_sample.replay_restarted) {
            continue;
          }
          awaiting_replay_restart = false;
        }
        if (!robot_io.publish(action_sample.command)) {
          std::cerr << "Failed to publish LowCmd\n";
          exit_code = 1;
          g_running.store(false, std::memory_order_relaxed);
          break;
        }
        mpjpe_tracker.add(action_sample);
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
        const double avg_worker_ms =
            published_action_window_count > 0
                ? worker_sum_ms /
                      static_cast<double>(published_action_window_count)
                : 0.0;
        const double avg_onnx_ms =
            published_action_window_count > 0
                ? onnx_sum_ms /
                      static_cast<double>(published_action_window_count)
                : 0.0;
        const auto robot_mode = robot_io.snapshotRobotMode();
        std::cout << std::fixed << std::setprecision(3)
                  << "state_seq=" << last_state_sequence
                  << " action_seq=" << last_action_sequence
                  << " published_actions=" << published_action_count
                  << " motion_mode=" << last_motion_mode
                  << " motion_step=" << last_motion_step;
        if (robot_mode.valid) {
          std::cout << " robot_state="
                    << RobotControlStateName(robot_mode.state)
                    << " robot_env="
                    << RobotEnvironmentName(robot_mode.environment)
                    << " low_command_mode="
                    << (robot_mode.low_command_mode ? "true" : "false");
        } else {
          std::cout << " robot_state=N/A robot_env=N/A"
                    << " low_command_mode=false";
        }
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

    motion_handler.stop();
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
