#include "mcmcsampler/dram.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "gaussian_proposal.hpp"

Eigen::MatrixXd mcmcsampler::DRAM(
    unsigned long num_dim,
    const std::function<double(const Eigen::VectorXd &)> &negative_log_pdf,
    const Eigen::MatrixXd &initial_covariance,
    const Eigen::VectorXd &initial_sample, unsigned long num_sample,
    unsigned long adaptive_metropolis_threshold,
    double adaptive_metropolis_regularization, double delayed_rejection_scaling,
    std::mt19937 &rng) {
  // sanity check
  if (num_dim < 1) {
    throw std::invalid_argument("Number of dimensions is not positive");
  }

  try {
    negative_log_pdf(initial_sample);
  } catch (...) {
    throw std::invalid_argument(
        "Cannot evaluate negative log pdf on initial sample");
  }

  if (initial_covariance.rows() != num_dim ||
      initial_covariance.cols() != num_dim) {
    throw std::invalid_argument(
        "Size of initial covariance matrix is incompatible with number of "
        "dimensions");
  }

  if (initial_sample.rows() != num_dim) {
    throw std::invalid_argument(
        "Size of initial sample is incompatible with number of dimensions");
  }

  if (num_sample < 1) {
    throw std::invalid_argument("Number of samples must be positive");
  }

  if (adaptive_metropolis_threshold < 1) {
    throw std::invalid_argument(
        "Adaptive Metropolis threshold must be positive");
  }

  if (adaptive_metropolis_regularization <= 0.0) {
    throw std::invalid_argument(
        "Adaptive Metropolis regularization must be positive");
  }

  if (delayed_rejection_scaling <= 0.0 || delayed_rejection_scaling > 1.0) {
    throw std::invalid_argument(
        "Delayed rejection scaling must be in (0, 1] interval");
  }

  // allocate memory for storing samples and copy over initial sample
  Eigen::MatrixXd samples(num_dim, num_sample + 1);
  Eigen::VectorXd x(num_dim);
  for (unsigned long d = 0; d < num_dim; ++d) {
    samples(d, 0) = initial_sample(d);
    x(d) = initial_sample(d);
  }

  // initialize variables to keep track of running sums of x and x * x^T
  Eigen::VectorXd sum_x(num_dim);
  Eigen::MatrixXd sum_x_xt(num_dim, num_dim);
  sum_x.fill(0.0);
  sum_x_xt.fill(0.0);

  // prepare for loops
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  const double s_d = 5.6644 / num_dim;
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(num_dim, num_dim);

  // main loop
  for (unsigned long n = 0; n < num_sample; ++n) {
    Eigen::MatrixXd S;
    if (n < adaptive_metropolis_threshold) {
      S = s_d * initial_covariance;
    } else {
      S = s_d * (sum_x_xt / n - sum_x * sum_x.transpose() / (n * (n + 1))) +
          adaptive_metropolis_regularization * I;
    }

    GaussianProposal proposal_1(S);
    Eigen::VectorXd y1 = proposal_1.Sample(x, rng);

    const double acceptance_1_x_y1 =
        std::exp(-std::max(0.0, negative_log_pdf(y1) - negative_log_pdf(x) +
                                    proposal_1.NegativeLogPdf(x, y1) -
                                    proposal_1.NegativeLogPdf(y1, x)));

    if (uniform(rng) < acceptance_1_x_y1) {
      x = y1;
    } else if (0.0 < delayed_rejection_scaling &&
               delayed_rejection_scaling < 1.0) {
      GaussianProposal proposal_2(delayed_rejection_scaling * S);
      Eigen::VectorXd y2 = proposal_2.Sample(x, rng);

      const double acceptance_1_y2_y1 =
          std::exp(-std::max(0.0, negative_log_pdf(y1) - negative_log_pdf(y2) +
                                      proposal_1.NegativeLogPdf(y2, y1) -
                                      proposal_1.NegativeLogPdf(y1, y2)));
      const double acceptance_2_x_y1_y2 =
          exp(-std::max(0.0, negative_log_pdf(y2) - negative_log_pdf(x) +
                                 proposal_1.NegativeLogPdf(y1, y2) -
                                 proposal_1.NegativeLogPdf(y1, x) +
                                 proposal_2.NegativeLogPdf(x, y2) -
                                 proposal_2.NegativeLogPdf(y2, x) -
                                 std::log(1.0 - acceptance_1_y2_y1) +
                                 std::log(1.0 - acceptance_1_x_y1)));

      if (uniform(rng) < acceptance_2_x_y1_y2) {
        x = y2;
      }
    }

    sum_x += x;
    sum_x_xt += x * x.transpose();

    for (unsigned long d = 0; d < num_dim; ++d) {
      samples(d, n + 1) = x(d);
    }
  }

  return samples;
}
