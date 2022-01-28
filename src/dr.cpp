#include "mcmcsampler/dr.hpp"

#include <stdexcept>

#include "mcmcsampler/dram.hpp"

Eigen::MatrixXd mcmcsampler::DR(
    unsigned long num_dim,
    const std::function<double(const Eigen::VectorXd &)> &negative_log_pdf,
    const Eigen::MatrixXd &proposal_covariance,
    const Eigen::VectorXd &initial_sample, unsigned long num_sample,
    double delayed_rejection_scaling, std::mt19937 &rng) {
  if (delayed_rejection_scaling <= 0.0 || delayed_rejection_scaling >= 1.0) {
    throw std::invalid_argument(
        "Delayed rejection scaling must be in (0, 1) interval");
  }

  return DRAM(num_dim, negative_log_pdf, proposal_covariance, initial_sample,
              num_sample, num_sample, 1.0, delayed_rejection_scaling, rng);
}
