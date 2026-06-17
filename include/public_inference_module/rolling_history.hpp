#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <deque>
#include <stdexcept>

namespace public_inference_module {

class RollingHistory {
  public:
    void init(std::size_t capacity, const Eigen::VectorXd &value) {
        if (capacity == 0) {
            throw std::invalid_argument("RollingHistory capacity must be positive");
        }
        capacity_ = capacity;
        values_.clear();
        values_.assign(capacity_, value);
    }

    void push_back(const Eigen::VectorXd &value) {
        if (capacity_ == 0) {
            throw std::logic_error("RollingHistory must be initialized before use");
        }
        if (values_.size() == capacity_) {
            values_.pop_front();
        }
        values_.push_back(value);
    }

    Eigen::VectorXd toEigenVector() const {
        if (values_.empty()) {
            return Eigen::VectorXd();
        }

        const Eigen::Index element_size = values_.front().size();
        Eigen::VectorXd flattened(static_cast<Eigen::Index>(values_.size()) * element_size);
        Eigen::Index offset = 0;
        for (const auto &value : values_) {
            flattened.segment(offset, element_size) = value;
            offset += element_size;
        }
        return flattened;
    }

  private:
    std::size_t capacity_ = 0;
    std::deque<Eigen::VectorXd> values_;
};

}  // namespace public_inference_module
