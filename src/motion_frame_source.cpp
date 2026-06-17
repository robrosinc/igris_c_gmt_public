#include "public_inference_module/motion_frame_source.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace public_inference_module {
namespace {

class NullMotionFrameSource : public MotionFrameSource {
  public:
    bool getLatest(MotionFrame &frame) override {
        frame = MotionFrame{};
        return false;
    }

    void reset() override {}
};

std::vector<double> ParseCsvLine(const std::string &line) {
    std::vector<double> values;
    std::stringstream stream(line);
    std::string token;
    while (std::getline(stream, token, ',')) {
        char *end   = nullptr;
        const auto value = std::strtod(token.c_str(), &end);
        if (end == token.c_str()) {
            return {};
        }
        values.push_back(value);
    }
    return values;
}

class CsvReplayMotionFrameSource : public MotionFrameSource {
  public:
    explicit CsvReplayMotionFrameSource(const MotionSourceConfig &config) : config_(config) {
        if (config_.csv_path.empty()) {
            throw std::runtime_error("motion_source.csv_path must be set when motion_source.type=csv_replay");
        }
        if (config_.fps <= 0.0) {
            throw std::runtime_error("motion_source.fps must be positive");
        }

        std::ifstream file(config_.csv_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open motion CSV: " + config_.csv_path);
        }

        std::string line;
        while (std::getline(file, line)) {
            const auto first_non_space = std::find_if_not(line.begin(), line.end(), [](unsigned char ch) { return std::isspace(ch) != 0; });
            if (first_non_space == line.end() || *first_non_space == '#') {
                continue;
            }

            const std::vector<double> values = ParseCsvLine(line);
            if (values.empty()) {
                continue;
            }
            if (values.size() != 52 && values.size() != 103) {
                throw std::runtime_error("Motion CSV row must have 52 legacy columns or 103 reference-tracking columns: " + line);
            }

            MotionFrame frame;
            frame.seq      = static_cast<uint64_t>(values[0]);
            frame.stamp_ns = static_cast<uint64_t>(values[1]);
            frame.valid    = true;

            std::size_t offset = 2;
            for (std::size_t i = 0; i < frame.joint_position.size(); ++i) {
                frame.joint_position[i] = values[offset + i];
            }
            offset += frame.joint_position.size();
            for (std::size_t i = 0; i < frame.joint_velocity.size(); ++i) {
                frame.joint_velocity[i] = values[offset + i];
            }
            offset += frame.joint_velocity.size();
            if (values.size() == 52) {
                for (std::size_t i = 0; i < frame.anchor_quaternion_wxyz.size(); ++i) {
                    frame.anchor_quaternion_wxyz[i] = values[offset + i];
                }
            } else {
                frame.root_position_z = values[offset++];
                for (std::size_t i = 0; i < frame.root_state.size(); ++i) {
                    frame.root_state[i] = values[offset + i];
                }
                offset += frame.root_state.size();
                for (std::size_t i = 0; i < frame.body_position.size(); ++i) {
                    frame.body_position[i] = values[offset + i];
                }
                offset += frame.body_position.size();
                for (std::size_t i = 0; i < frame.root_linear_velocity.size(); ++i) {
                    frame.root_linear_velocity[i] = values[offset + i];
                }
                offset += frame.root_linear_velocity.size();
                for (std::size_t i = 0; i < frame.root_angular_velocity.size(); ++i) {
                    frame.root_angular_velocity[i] = values[offset + i];
                }
            }
            frames_.push_back(frame);
        }

        if (frames_.empty()) {
            throw std::runtime_error("Motion CSV contains no valid frames: " + config_.csv_path);
        }

        reset();
    }

    bool getLatest(MotionFrame &frame) override {
        if (frames_.empty()) {
            return false;
        }

        const auto now         = std::chrono::steady_clock::now();
        const auto elapsed_sec = std::chrono::duration<double>(now - start_time_).count();
        const auto elapsed_frames =
            static_cast<std::size_t>(std::max(0.0, elapsed_sec * config_.fps));

        std::size_t index = elapsed_frames;
        if (config_.loop) {
            index %= frames_.size();
        } else if (index >= frames_.size()) {
            index = frames_.size() - 1;
        }

        frame = frames_[index];
        return true;
    }

    void reset() override { start_time_ = std::chrono::steady_clock::now(); }

  private:
    MotionSourceConfig config_;
    std::vector<MotionFrame> frames_;
    std::chrono::steady_clock::time_point start_time_ = std::chrono::steady_clock::now();
};

}  // namespace

std::unique_ptr<MotionFrameSource> CreateMotionFrameSource(const InferenceConfig &config) {
    if (config.motion_source.type == "null" || config.motion_source.type == "hold") {
        return std::make_unique<NullMotionFrameSource>();
    }
    if (config.motion_source.type == "csv_replay") {
        return std::make_unique<CsvReplayMotionFrameSource>(config.motion_source);
    }

    throw std::runtime_error("Unsupported motion_source.type: " + config.motion_source.type);
}

}  // namespace public_inference_module
