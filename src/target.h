#ifndef TARGET_H
#define TARGET_H

double target(const Eigen::VectorXd& x, FENRIR &fenrir){
    Eigen::Map<const Eigen::MatrixXd> eta(x.data(), fenrir.P, fenrir.N_obs);
    double log_prob_eta = fenrir.log_pdf(eta);
    Eigen::RowVectorXd n = fenrir.Y_obs.colwise().sum();
    Eigen::ArrayXXd O = eta.array().exp();
    Eigen::ArrayXd m = O.colwise().sum();
    m += Eigen::ArrayXd::Ones(fenrir.N_obs);
    Eigen::MatrixXd rhomat = (O.rowwise() / m.transpose()).matrix();
    Eigen::Map<Eigen::VectorXd> rhovec(rhomat.data(), rhomat.size());
    Eigen::VectorXd rho = rhovec;
    double log_prob_mult = (fenrir.Y_obs.topRows(fenrir.P).array() * eta.array()).sum() - n * m.log().matrix();
    return log_prob_eta + log_prob_mult;
}

#endif