#ifndef FENRIR_H
#define FENRIR_H

/*
    This is the main class where log probability of the model is calculated.
*/
class FENRIR
{
public:
    int D;
    int P;
    int N_obs;
    int N_total;
    int num_timeseries;
    int Q;
    Eigen::MatrixXd Y_obs;
    Eigen::VectorXd observed_TT;
    Eigen::VectorXd N_total_list;

    std::vector<Eigen::MatrixXd> A;
    std::vector<Eigen::MatrixXd> R;
    std::vector<Eigen::MatrixXd> M;
    std::vector<Eigen::MatrixXd> C;
    Eigen::MatrixXd f;
    Eigen::VectorXd q;
    Eigen::MatrixXd e;
    Eigen::MatrixXd S;

    Eigen::VectorXd v;
    std::vector<Eigen::MatrixXd> Xi, G, W;
    std::vector<Eigen::MatrixXd> M0;
    std::vector<Eigen::MatrixXd> C0;
    Eigen::VectorXd gamma;
    Eigen::MatrixXd F;
    int F_len, G_len, W_len, gamma_len;

    FENRIR(
        Eigen::MatrixXd Y_obs, Eigen::VectorXd observed_TT, Eigen::VectorXd N_total_list, int N_total, int N_obs, Eigen::MatrixXd F, std::vector<Eigen::MatrixXd> G, Eigen::VectorXd gamma, std::vector<Eigen::MatrixXd> W, std::vector<Eigen::MatrixXd> M0, std::vector<Eigen::MatrixXd> C0, Eigen::MatrixXd Xi0, double v0)
    {
        this->Y_obs = Y_obs;
        this->N_total = observed_TT.size();
        this->N_obs = N_obs;
        this->num_timeseries = N_total_list.size();
        this->observed_TT = observed_TT;
        this->N_total_list = N_total_list;
        this->D = Y_obs.rows();
        this->P = D - 1;
        this->Q = C0[0].rows();

        this->A.resize(this->N_total);
        this->R.resize(this->N_total);
        this->M.resize(this->N_total + this->num_timeseries);
        this->C.resize(this->N_total + this->num_timeseries);
        this->Xi.resize(this->N_total + 1);
        this->v.resize(this->N_total + 1);

        this->F = F;
        this->G = G;
        this->gamma = gamma;
        this->W = W;
        this->M0 = M0;
        this->C0 = C0;
        this->Xi[0] = Xi0;
        this->v[0] = v0;
        this->F_len = F.size();
        this->G_len = G.size();
        this->W_len = W.size();
        this->gamma_len = gamma.size();

        this->f = Eigen::MatrixXd::Zero(this->P, this->N_obs);
        this->q = Eigen::VectorXd::Zero(this->N_obs);
        this->e = Eigen::MatrixXd::Zero(this->P, this->N_obs);
        this->S = Eigen::MatrixXd::Zero(this->Q, this->N_obs);
    }

    Eigen::MatrixXd calculate_gradient(Eigen::MatrixXd eta)
    {
        Eigen::VectorXd S = Eigen::VectorXd::Zero(this->N_obs);
        int pos = 0;
        for (int i = 0; i < this->N_total; i++)
        {
            if (this->observed_TT(i) == 1)
            {
                S[pos] = 1 + 1 / this->v[i] * ((eta.col(pos) - this->f.col(pos)).transpose() * invertMatrix(this->q[pos] * this->Xi[i]) * (eta.col(pos) - this->f.col(pos)))(0, 0);
                pos += 1;
            }
            else
            {
                continue;
            }
        }
        Eigen::MatrixXd grad_sum = Eigen::MatrixXd::Zero(this->P, this->N_obs);
        pos = 0;
        for (int i = 0; i < this->N_total; i++)
        {
            if (this->observed_TT(i) == 1)
            {
                grad_sum.col(pos) = -1 * (this->v[i] + this->P) / 2 * 1 / S[pos] * 2 / this->v[i] * (eta.col(pos).transpose() - this->f.col(pos).transpose()) * invertMatrix(this->q[pos] * this->Xi[i]);
                pos += 1;
            }
            else
            {
                continue;
            }
        }
        return grad_sum;
    }

    double log_pdf(const Eigen::MatrixXd &eta)
    {
        double log_prob_eta_accum = 0;
        int pos = 0;
        int t = 0;

        Eigen::MatrixXd G_t(this->Q, this->Q);
        Eigen::VectorXd F_t(this->Q);
        Eigen::MatrixXd W_t(this->Q, this->Q);
        double gamma_t;

        for (int timeseries = 0; timeseries < this->num_timeseries; timeseries++)
        {
            this->M[t + timeseries] = this->M0[timeseries];
            this->C[t + timeseries] = this->C0[timeseries];
            for (int i = 0; i < this->N_total_list[timeseries]; i++)
            {
                F_t = this->F.col(t);
                G_t = this->G[t];
                W_t = this->W[t];
                gamma_t = this->gamma[t];
                this->A[t] = G_t * this->M[t + timeseries];
                this->R[t] = G_t * this->C[t + timeseries] * G_t.transpose() + W_t;

                if (this->observed_TT(t) == 1)
                {
                    this->f.col(pos) = this->A[t].transpose() * F_t;
                    this->q[pos] = F_t.transpose() * this->R[t] * F_t + gamma_t;
                    log_prob_eta_accum = log_prob_eta_accum + mult_t_log_pdf(eta.col(pos),
                                                                             this->v[t],
                                                                             this->f.col(pos),
                                                                             (this->q[pos] * this->Xi[t]) / this->v[t]);
                    this->S.col(pos) = this->R[t] * F_t / this->q[pos];
                    this->e.col(pos) = eta.col(pos) - this->f.col(pos);
                    this->M[t + timeseries + 1] = this->A[t] + this->S.col(pos) * this->e.col(pos).transpose();
                    this->C[t + timeseries + 1] = this->R[t] - this->q[pos] * (this->S.col(pos) * this->S.col(pos).transpose());
                    this->v[t + 1] = this->v[t] + 1;
                    this->Xi[t + 1] = this->Xi[t] + (this->e.col(pos) * this->e.col(pos).transpose()) / this->q[pos];
                    pos += 1;
                }
                else
                {
                    this->M[t + timeseries + 1] = this->A[t];
                    this->C[t + timeseries + 1] = this->R[t];
                    this->v[t + 1] = this->v[t];
                    this->Xi[t + 1] = this->Xi[t];
                }
                t = t + 1;
            }
        }
        return log_prob_eta_accum;
    }
};

/*
    Function for the smoother
*/
std::vector<std::vector<Eigen::MatrixXd>> fenrir_smoother(const Eigen::MatrixXd &eta,
                                                          Eigen::MatrixXd F, std::vector<Eigen::MatrixXd> G, Eigen::VectorXd gamma, std::vector<Eigen::MatrixXd> W,
                                                          std::vector<Eigen::MatrixXd> M0, std::vector<Eigen::MatrixXd> C0, Eigen::MatrixXd Xi0,
                                                          double v0, Eigen::VectorXd observed_TT,
                                                          Eigen::VectorXd N_total_list, int seed)
{
    int TT = observed_TT.size();
    int p = eta.rows();
    int q = G[0].rows();
    int num_timeseries = N_total_list.size();

    Eigen::VectorXd F_t(q);
    Eigen::MatrixXd G_t(q, q);
    Eigen::MatrixXd W_t(q, q);
    Eigen::VectorXd f(p);
    double qq;
    Eigen::VectorXd S(q);
    Eigen::VectorXd e(p);
    double gamma_t;
    std::vector<Eigen::MatrixXd> Xi(TT + 1, Eigen::MatrixXd(p, p));
    std::vector<double> v(TT + 1, 0.0);

    std::vector<Eigen::MatrixXd> M(TT + num_timeseries, Eigen::MatrixXd(q, p));
    std::vector<Eigen::MatrixXd> C(TT + num_timeseries, Eigen::MatrixXd(q, q));
    std::vector<Eigen::MatrixXd> A(TT, Eigen::MatrixXd(q, p));
    std::vector<Eigen::MatrixXd> R(TT, Eigen::MatrixXd(q, q));
    std::vector<Eigen::MatrixXd> theta(TT, Eigen::MatrixXd(q, p));
    int pos = 0;
    int t = 0;

    Xi[0] = Xi0;
    v[0] = v0;

    for (int timeseries = 0; timeseries < num_timeseries; timeseries++)
    {
        M[t + timeseries] = M0[timeseries];
        C[t + timeseries] = C0[timeseries];
        for (int i = 0; i < N_total_list[timeseries]; i++)
        {
            F_t = F.col(t);
            G_t = G[t];
            W_t = W[t];
            gamma_t = gamma[t];
            A[t] = G_t * M[t + timeseries];
            R[t] = G_t * C[t + timeseries] * G_t.transpose() + W_t;
            if (observed_TT(t) == 1)
            {
                f = A[t].transpose() * F_t;
                qq = F_t.transpose() * R[t] * F_t + gamma_t;
                S = R[t] * F_t / qq;
                e = eta.col(pos) - f;
                M[t + timeseries + 1] = A[t] + S * e.transpose();
                C[t + timeseries + 1] = R[t] - qq * (S * S.transpose());
                v[t + 1] = v[t] + 1;
                Xi[t + 1] = Xi[t] + (e * e.transpose()) / qq;
                pos += 1;
            }
            else
            {
                M[t + timeseries + 1] = A[t];
                C[t + timeseries + 1] = R[t];
                v[t + 1] = v[t];
                Xi[t + 1] = Xi[t];
            }
            t = t + 1;
        }
    }

    Xi[TT] = (Xi[TT] + Xi[TT].transpose()) / 2.0;

    boost::random::mt19937 rng(seed);
    Eigen::MatrixXd Sigma = inv_wishart_rng(v[TT], Xi[TT], rng);
    Sigma = (Sigma + Sigma.transpose()) / 2.0;

    t = -1;
    int reset_flag = 1;
    Eigen::MatrixXd Z(q, q);

    std::vector<Eigen::MatrixXd> theta0(num_timeseries, Eigen::MatrixXd(q, p));

    for (int timeseries = num_timeseries - 1; timeseries >= 0; timeseries--)
    {
        reset_flag = 1;
        for (int i = 0; i < N_total_list[timeseries]; ++i)
        {
            if (reset_flag == 1)
            {
                C[TT + timeseries - t - 1] = (C[TT + timeseries - t - 1] + C[TT + timeseries - t - 1].transpose()) / 2.0;
                theta[TT - t - 2] = matrix_normal_rng(M[TT + timeseries - t - 1], C[TT + timeseries - t - 1], Sigma, rng);
                reset_flag = 0;
            }
            else
            {
                G_t = G[TT - t - 1];
                Z = C[TT + timeseries - t - 1] * G_t.transpose() * (R[TT - t - 1]).inverse();
                C[TT + timeseries - t - 1] = C[TT + timeseries - t - 1] - Z * R[TT - t - 1] * Z.transpose();
                C[TT + timeseries - t - 1] = (C[TT + timeseries - t - 1] + C[TT + timeseries - t - 1].transpose()) / 2.0;
                M[TT + timeseries - t - 1] = M[TT + timeseries - t - 1] + Z * (theta[TT - t - 1] - A[TT - t - 1]);
                theta[TT - t - 2] = matrix_normal_rng(M[TT + timeseries - t - 1], C[TT + timeseries - t - 1], Sigma, rng);
                if (i == N_total_list[timeseries] - 1)
                {
                    G_t = G[TT - t - 2];
                    Z = C[TT + timeseries - t - 2] * G_t.transpose() * (R[TT - t - 2]).inverse();
                    C[TT + timeseries - t - 2] = C[TT + timeseries - t - 2] - Z * R[TT - t - 2] * Z.transpose();
                    C[TT + timeseries - t - 2] = (C[TT + timeseries - t - 2] + C[TT + timeseries - t - 2].transpose()) / 2.0;
                    M[TT + timeseries - t - 2] = M[TT + timeseries - t - 2] + Z * (theta[TT - t - 2] - A[TT - t - 2]);
                    theta0[timeseries] = matrix_normal_rng(M[TT + timeseries - t - 2], C[TT + timeseries - t - 2], Sigma, rng);
                }
            }
            t = t + 1;
        }
    }

    std::vector<Eigen::MatrixXd> Sigma_wrapper(1, Sigma);
    std::vector<std::vector<Eigen::MatrixXd>> result(3);
    result[0] = theta;
    result[1] = theta0;
    result[2] = Sigma_wrapper;
    return result;
}
#endif
