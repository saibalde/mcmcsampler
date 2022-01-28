#include "mcmcsampler/am.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>

void WriteMatrix(const Eigen::MatrixXd &matrix, const std::string &file_name) {
  std::ofstream file(file_name);
  file << std::scientific;
  for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
      file << std::setw(13) << std::setprecision(6) << matrix(i, j) << " ";
    }
    file << std::endl;
  }
  file << std::defaultfloat;
}

TEST(McmcSampler, AdaptiveMetropolis) {
  unsigned long num_dim = 2;
  std::function<double(const Eigen::VectorXd &)> negative_log_pdf =
      [](const Eigen::VectorXd &x) {
        const double y1 = x[0];
        const double y2 = x[1] + x[0] * x[0] + 1.0;
        return (y1 * y1 + y2 * y2 - 1.8 * y1 * y2) / 0.38;
      };
  Eigen::MatrixXd initial_covariance(num_dim, num_dim);
  initial_covariance << 100.0 / 19.0, -90.0 / 19.0, -90.0 / 19.0, 100.0 / 19.0;
  Eigen::VectorXd initial_sample(num_dim);
  initial_sample << 0.0, -1.0;
  unsigned long num_sample = 100000;
  unsigned long adaptive_metropolis_threshold = 1000;
  double adaptive_metropolis_regularization = 1.0e-02;
  std::mt19937 rng(0);

  Eigen::MatrixXd samples = mcmcsampler::AM(
      num_dim, negative_log_pdf, initial_covariance, initial_sample, num_sample,
      adaptive_metropolis_threshold, adaptive_metropolis_regularization, rng);

  WriteMatrix(samples, "am_test.dat");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
