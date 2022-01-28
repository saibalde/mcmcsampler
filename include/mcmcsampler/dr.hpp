#ifndef MCMCSAMPLER_DR_HPP
#define MCMCSAMPLER_DR_HPP

#include <Eigen/Core>
#include <functional>
#include <random>

namespace mcmcsampler {

Eigen::MatrixXd DR(
    unsigned long num_dim,
    const std::function<double(const Eigen::VectorXd &)> &negative_log_pdf,
    const Eigen::MatrixXd &proposal_covariance,
    const Eigen::VectorXd &initial_sample, unsigned long num_sample,
    double delayed_rejection_scaling, std::mt19937 &rng);

}

#endif
