#pragma once

#include "core/config.hpp"
#include "core/pipeline_types.hpp"
#include "utils/obs_functions.h"
#include "utils/rolling_history.hpp"

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace igris_c_gmt_public {

class ObservationBuilder {
public:
  int configure(const InferenceConfig &config);
  void reset();
  int build(const ObservationInput &input, ObservationResult &result);
  bool readLatest(ObservationResult &result) const;
  const Vector23d &qDefault() const { return q_default_; }

private:
  struct TermState {
    std::string function_name;
    obs_functions::ObsFunction function;
    std::size_t history_length = 1;
    RollingHistory history;
    bool initialized = false;
  };

  struct GroupState {
    std::string name;
    std::vector<TermState> terms;
  };

  void resetHistories();

private:
  std::string obs_config_path_;
  Vector23d q_default_ = Vector23d::Zero();
  std::vector<GroupState> groups_;
  mutable std::mutex mutex_;
  ObservationResult latest_result_;
  bool has_result_ = false;
};

} // namespace igris_c_gmt_public
