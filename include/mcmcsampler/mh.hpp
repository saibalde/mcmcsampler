#ifndef MCMCSAMPLER_MH_HPP
#define MCMCSAMPLER_MH_HPP

#include <Eigen/Core>
#include <functional>
#include <random>

#include "mcmcsampler/dram.hpp"

namespace mcmcsampler {

inline Eigen::MatrixXd MH(
    unsigned long num_dim,
    const std::function<double(const Eigen::VectorXd &)> &negative_log_pdf,
    const Eigen::MatrixXd &proposal_covariance,
    const Eigen::VectorXd &initial_sample, unsigned long num_sample,
    std::mt19937 &rng) {
  return DRAM(num_dim, negative_log_pdf, proposal_covariance, initial_sample,
              num_sample, num_sample, 1.0, 1.0, rng);
}

}  // namespace mcmcsampler

#endif
