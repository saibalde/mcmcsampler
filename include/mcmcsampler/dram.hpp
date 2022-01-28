#ifndef MCMCSAMPLER_DRAM_HPP
#define MCMCSAMPLER_DRAM_HPP

#include <Eigen/Core>
#include <functional>
#include <random>

namespace mcmcsampler {

Eigen::MatrixXd DRAM(
    unsigned long num_dim,
    const std::function<double(const Eigen::VectorXd &)> &negative_log_pdf,
    const Eigen::MatrixXd &initial_covariance,
    const Eigen::VectorXd &initial_sample, unsigned long num_sample,
    unsigned long adaptive_metropolis_threshold,
    double adaptive_metropolis_regularization, double delayed_rejection_scaling,
    std::mt19937 &rng);

}

#endif
