#ifndef FENRIR_GRAD
#define FENRIR_GRAD

class FenrirGrad : public Numer::MFuncGrad
{
private:
    Eigen::MatrixXd Y_obs;
    Eigen::VectorXd observed_TT;
    int N_total;
    int N_obs;
    Eigen::VectorXd N_total_list;
    int P;
    LogProbSaver saver;

    Eigen::ArrayXXd O;
    Eigen::ArrayXd m;
    Eigen::MatrixXd rhomat;
    Eigen::VectorXd rho;
    Eigen::RowVectorXd n;

    Eigen::MatrixXd F;
    std::vector<Eigen::MatrixXd> G;
    Eigen::VectorXd gamma;
    std::vector<Eigen::MatrixXd> W;
    std::vector<Eigen::MatrixXd> M0;
    std::vector<Eigen::MatrixXd> C0;
    Eigen::MatrixXd Xi0;
    double v0;

public:
    FenrirGrad(Eigen::MatrixXd Y_obs_, Eigen::VectorXd observed_TT_, Eigen::VectorXd N_total_list_, int N_total_, int N_obs_, Eigen::MatrixXd F_, std::vector<Eigen::MatrixXd> G_, Eigen::VectorXd gamma_, std::vector<Eigen::MatrixXd> W_, std::vector<Eigen::MatrixXd> M0_, std::vector<Eigen::MatrixXd> C0_, Eigen::MatrixXd Xi0_, 
             double v0_, std::string log_prob_filename_):
             Y_obs(Y_obs_), observed_TT(observed_TT_), N_total_list(N_total_list_), N_total(N_total_), N_obs(N_obs_), F(F_), G(G_), gamma(gamma_), W(W_), M0(M0_), C0(C0_), Xi0(Xi0_), v0(v0_), saver(log_prob_filename_)
    {
        n = Y_obs.colwise().sum();
        P = Y_obs.rows() - 1;
    }
    
    ~FenrirGrad() {}

    double calc_eta_ll(const Eigen::Ref<const Eigen::VectorXd> &etavec, FENRIR &fenrir)
    {
        Eigen::Map<const Eigen::MatrixXd> eta(etavec.data(), P, N_obs);
        double log_prob = fenrir.log_pdf(eta);
        return log_prob;
    }
    double calc_mult_ll(const Eigen::Ref<const Eigen::VectorXd> &etavec)
    {
        Eigen::Map<const Eigen::MatrixXd> eta(etavec.data(), P, N_obs);

        O = eta.array().exp();
        m = O.colwise().sum();
        m += Eigen::ArrayXd::Ones(N_obs);
        rhomat = (O.rowwise() / m.transpose()).matrix();
        Eigen::Map<Eigen::VectorXd> rhovec(rhomat.data(), rhomat.size());
        rho = rhovec;
        double log_prob = (Y_obs.topRows(P).array() * eta.array()).sum() - n * m.log().matrix();
        return log_prob;
    }

    double calc_ll(const Eigen::Ref<const Eigen::VectorXd> &etavec, FENRIR &fenrir)
    {
        double log_prob_eta = calc_eta_ll(etavec, fenrir);
        double log_prob_mult = calc_mult_ll(etavec);
        double log_prob = log_prob_eta + log_prob_mult;
        saver.saveValues(log_prob_mult, log_prob_eta, log_prob);
        return log_prob;
    }

    Eigen::VectorXd calc_grad(const Eigen::Ref<const Eigen::VectorXd> &etavec, FENRIR &fenrir)
    {
        Eigen::Map<const Eigen::MatrixXd> eta(etavec.data(), P, N_obs);
        Eigen::MatrixXd g = (Y_obs.topRows(P).array() - (rhomat.array().rowwise() * n.array())).matrix();
        Eigen::MatrixXd grad_sum = fenrir.calculate_gradient(eta);
        g.noalias() += grad_sum;
        Eigen::Map<Eigen::VectorXd> grad(g.data(), g.size());
        return grad;
    }

    double f_grad(Numer::Constvec &etavec, Numer::Refvec grad)
    {   
        FENRIR fenrir(Y_obs, observed_TT, N_total_list, N_total, N_obs, F, G, gamma, W, M0, C0, Xi0, v0);
        double log_prob = calc_ll(etavec, fenrir);
        grad = -calc_grad(etavec, fenrir);
        return -log_prob;
    }
};

#endif