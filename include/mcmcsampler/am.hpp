#ifndef MCMCSAMPLER_AM_HPP
#define MCMCSAMPLER_AM_HPP

#include <Eigen/Core>
#include <functional>
#include <random>

#include "mcmcsampler/dram.hpp"

namespace mcmcsampler {

inline Eigen::MatrixXd AM(
    unsigned long num_dim,
    const std::function<double(const Eigen::VectorXd &)> &negative_log_pdf,
    const Eigen::MatrixXd &initial_covariance,
    const Eigen::VectorXd &initial_sample, unsigned long num_sample,
    unsigned long adaptive_metropolis_threshold,
    double adaptive_metropolis_regularization, std::mt19937 &rng) {
  return DRAM(num_dim, negative_log_pdf, initial_covariance, initial_sample,
              num_sample, adaptive_metropolis_threshold,
              adaptive_metropolis_regularization, 1.0, rng);
}

}  // namespace mcmcsampler

#endif
