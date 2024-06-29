#ifndef MULTDIRICHLETBOOT_H
#define MULTDIRICHLETBOOT_H

namespace MultDirichletBoot{
  
  Eigen::MatrixXd alrInv_default(const Eigen::Ref<const Eigen::MatrixXd> &eta){
    int D = eta.rows()+1;
    int N = eta.cols();
    Eigen::MatrixXd pi = Eigen::MatrixXd::Zero(D, N);
    pi.topRows(D-1) = eta;
    pi.array() = pi.array().exp();
    pi.array().rowwise() /= pi.colwise().sum().array();
    return pi;
  }
  
  Eigen::MatrixXd alr_default(Eigen::MatrixXd& pi){
    int D = pi.rows();
    int N = pi.cols();
    Eigen::MatrixXd eta(D-1,N);
    eta = pi.topRows(D-1);
    eta.array().rowwise() /= pi.row(D-1).array();
    return eta.array().log();
  }
  
  // Sample dirichlet - alpha must be a vector
  Eigen::MatrixXd rDirichlet(int n_samples, Eigen::VectorXd& alpha){
    int D = alpha.rows();
    int p = alpha.cols();
    if (p > 1) Rcpp::stop("rDirichlet must only be passed alpha as a vector");
    Rcpp::NumericVector r(n_samples);
    Eigen::MatrixXd s(D, n_samples);
    for (int i=0; i<D; i++){
      r = Rcpp::rgamma(n_samples, alpha(i), 1);
      Eigen::Map<Eigen::VectorXd> rvec(r.begin(), n_samples);
      s.row(i) = rvec; // Directly assign without transpose
    }
    s.array().rowwise() /= s.colwise().sum().array();
    return s;
  }
  
  
  Eigen::MatrixXd MultDirichletBoot(int n_samples, const Eigen::Ref<const Eigen::MatrixXd> &eta, 
                             Eigen::ArrayXXd Y, double pseudocount){
    int D = eta.rows()+1;
    int N = eta.cols();
    Eigen::MatrixXd alpha = alrInv_default(eta);
    alpha.array().rowwise() *= Y.colwise().sum();
    alpha.array() += pseudocount; 
    Eigen::MatrixXd samp(N*(D-1), n_samples);
    Eigen::MatrixXd s(D, n_samples);
    Eigen::VectorXd a;
    for (int i=0; i<N; i++){
      a = alpha.col(i);
      s = rDirichlet(n_samples, a);
      // transform to eta
      samp.middleRows(i*(D-1), D-1) = alr_default(s);
    }
    return samp;
  }
}


#endif