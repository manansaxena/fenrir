#include "fenrirs.h"

//[[Rcpp::export]]
Rcpp::List fenrir_optim(
    Eigen::MatrixXd Y_obs, Eigen::VectorXd observed_TT, Eigen::VectorXd N_total_list,
    Eigen::MatrixXd F, Rcpp::List G, Eigen::VectorXd gamma, Rcpp::List W, Rcpp::List M0,
    Rcpp::List C0, Eigen::MatrixXd Xi0, double v0, Eigen::MatrixXd init,  Rcpp::CharacterVector log_probs_path,
    int num_dirsamples = 2000, double pseudocount = 0.5,
    double eps_f = 1e-8, double eps_g = 1e-5, int max_iter = 10000
    )
{
    std::vector<Eigen::MatrixXd> G_cpp = RcppListToMatrixVector(G);
    std::vector<Eigen::MatrixXd> W_cpp = RcppListToMatrixVector(W);
    std::vector<Eigen::MatrixXd> M0_cpp = RcppListToMatrixVector(M0);
    std::vector<Eigen::MatrixXd> C0_cpp = RcppListToMatrixVector(C0);

    Eigen::VectorXd etavec = Eigen::Map<Eigen::VectorXd>(init.data(), init.size());
    std::string log_probs_path_cpp = Rcpp::as<std::string>(log_probs_path);

    FenrirGrad fenrir_grad(Y_obs, observed_TT, N_total_list, observed_TT.cols(), Y_obs.cols(), F, G_cpp, gamma, W_cpp, M0_cpp, C0_cpp, Xi0, v0, log_probs_path_cpp);

    double nllopt = 0.0;
    auto optim_start = std::chrono::high_resolution_clock::now();
    int status = Numer::optim_lbfgs(fenrir_grad, etavec, nllopt, max_iter, eps_f, eps_g);
    auto optim_stop = std::chrono::high_resolution_clock::now();
    int optim_duration = std::chrono::duration_cast<std::chrono::milliseconds>(optim_stop - optim_start).count();

    Rcpp::Rcout << "optimizer status:" << status << std::endl;
    if (status < 0)
        Rcpp::stop("fail to converge");

    Eigen::MatrixXd etamat = Eigen::Map<Eigen::MatrixXd>(etavec.data(), Y_obs.rows() - 1, Y_obs.cols());

    auto dirichlet_start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd multdirsamp = MultDirichletBoot::MultDirichletBoot(num_dirsamples, etamat, Y_obs.array(), pseudocount);
    Rcpp::IntegerVector d = Rcpp::IntegerVector::create(Y_obs.rows() - 1, Y_obs.cols(), num_dirsamples);
    Rcpp::NumericVector samples = Rcpp::wrap(multdirsamp);
    samples.attr("dim") = d;
    auto dirichlet_stop = std::chrono::high_resolution_clock::now();
    int dirichlet_duration = std::chrono::duration_cast<std::chrono::milliseconds>(dirichlet_stop - dirichlet_start).count();

    return Rcpp::List::create(
        Rcpp::Named("final_ll") = -nllopt,
        Rcpp::Named("optim_eta") = etamat,
        Rcpp::Named("optim_time") = optim_duration,
        Rcpp::Named("mult_dir_samples") = samples,
        Rcpp::Named("mult_dir_time") = dirichlet_duration);
}

//[[Rcpp::export]]
Rcpp::List fenrir_smooth(Eigen::MatrixXd eta, Eigen::MatrixXd F, Rcpp::List G,
                        Eigen::VectorXd gamma, Rcpp::List W, Rcpp::List M0,
                        Rcpp::List C0, Eigen::MatrixXd Xi0, double v0, Eigen::VectorXd observed_TT, Eigen::VectorXd N_total_list, int seed)
{
    std::vector<Eigen::MatrixXd> G_cpp = RcppListToMatrixVector(G);
    std::vector<Eigen::MatrixXd> W_cpp = RcppListToMatrixVector(W);
    std::vector<Eigen::MatrixXd> M0_cpp = RcppListToMatrixVector(M0);
    std::vector<Eigen::MatrixXd> C0_cpp = RcppListToMatrixVector(C0);

    std::vector<std::vector<Eigen::MatrixXd>> theta_smoothed = fenrir_smoother(eta, F, G_cpp, gamma, W_cpp, M0_cpp, C0_cpp, Xi0, v0, observed_TT, N_total_list, seed);
    return Rcpp::List::create(
        Rcpp::Named("theta_smoothed") = theta_smoothed[0],
        Rcpp::Named("theta0_smoothed") = theta_smoothed[1],
        Rcpp::Named("Sigma") = theta_smoothed[2]
    );
}
