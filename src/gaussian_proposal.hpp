#ifndef GAUSSIAN_PROPOSAL_HPP
#define GAUSSIAN_PROPOSAL_HPP

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <random>

class GaussianProposal {
public:
  GaussianProposal(const Eigen::MatrixXd &S) : S_(S) {}

  ~GaussianProposal() = default;

  GaussianProposal(const GaussianProposal &) = delete;

  GaussianProposal &operator=(const GaussianProposal &) = delete;

  GaussianProposal(GaussianProposal &&) = delete;

  GaussianProposal &operator=(GaussianProposal &&) = delete;

  Eigen::VectorXd Sample(const Eigen::VectorXd &x, std::mt19937 &rng) const {
    std::normal_distribution<double> normal(0.0, 1.0);

    const Eigen::Index n = x.rows();
    Eigen::VectorXd z(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      z(i) = normal(rng);
    }

    return x + S_.matrixL() * z;
  }

  double NegativeLogPdf(const Eigen::VectorXd &y,
                        const Eigen::VectorXd &x) const {
    Eigen::VectorXd z = y - x;
    return 0.5 * z.dot(S_.solve(z));
  }

private:
  Eigen::LLT<Eigen::MatrixXd> S_;
};

#endif
