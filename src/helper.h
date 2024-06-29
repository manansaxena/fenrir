#ifndef HELPER_H
#define HELPER_H

#include <random>

double mult_t_log_pdf(Eigen::VectorXd x,
                      double nu,
                      Eigen::VectorXd mean,
                      Eigen::MatrixXd sigma)
{
    int d = x.size();
    double nu_d = nu + d;

    Eigen::VectorXd diff = x - mean;
    double mahalanobis = diff.transpose() * sigma.inverse() * diff;

    double log_pdf = boost::math::lgamma(0.5 * nu_d) - boost::math::lgamma(0.5 * nu) - 0.5 * log(sigma.determinant()) - 0.5 * d * log(nu * M_PI) - 0.5 * nu_d * log(1.0 + mahalanobis / nu);
    return log_pdf;
}

Eigen::MatrixXd to_symmetric_matrix(Eigen::MatrixXd x)
{
    return x + x.transpose();
}

Eigen::MatrixXd multiply_lower_tri_self_transpose(const Eigen::MatrixXd &x)
{
    Eigen::MatrixXd L = x.triangularView<Eigen::Lower>();
    return L * L.transpose();
}

Eigen::MatrixXd invertMatrix(const Eigen::MatrixXd &X)
{
    Eigen::MatrixXd scaledMatrix = X;
    Eigen::LLT<Eigen::MatrixXd> llt(scaledMatrix);
    if (llt.info() == Eigen::NumericalIssue)
    {
        throw std::runtime_error("Numerical issue occurred during Cholesky decomposition");
    }
    Eigen::MatrixXd inverse = llt.solve(Eigen::MatrixXd::Identity(scaledMatrix.rows(), scaledMatrix.cols()));
    return inverse;
}

bool is_spd_square(const Eigen::MatrixXd &x)
{
    if (x.rows() != x.cols())
    {
        return false;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd singular_values = svd.singularValues();
    for (int i = 0; i < singular_values.size(); ++i)
    {
        if (singular_values[i] <= 0)
        {
            return false;
        }
    }
    return true;
}

Eigen::MatrixXd read_csv(const std::string &filename)
{
    std::vector<double> matrixEntries;
    std::ifstream matrixDataFile(filename);

    std::string matrixRowString;
    std::string matrixEntry;

    int matrixRowNumber = 0;

    while (getline(matrixDataFile, matrixRowString))
    {
        std::stringstream matrixRowStringStream(matrixRowString);

        while (getline(matrixRowStringStream, matrixEntry, ','))
        {
            matrixEntries.push_back(stod(matrixEntry));
        }
        ++matrixRowNumber;
    }

    if (matrixRowNumber == 1)
    {
        return Eigen::Map<Eigen::VectorXd>(matrixEntries.data(), matrixEntries.size());
    }
    else if (matrixEntries.size() == matrixRowNumber)
    {
        return Eigen::Map<Eigen::VectorXd>(matrixEntries.data(), matrixRowNumber);
    }
    else
    {
        return Eigen::Map<Eigen::MatrixXd>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
    }
}

class LogProbSaver
{
public:
    LogProbSaver(const std::string &filename)
    {
        file.open(filename, std::ios::out);
        if (!file.is_open())
        {
            std::cerr << "Failed to open file for writing" << std::endl;
        }
    }

    ~LogProbSaver()
    {
        if (file.is_open())
        {
            file.close();
        }
    }

    void saveValues(double logProbEta, double logProbNll, double logProb)
    {
        if (file.is_open())
        {
            file << std::setprecision(10) << logProbEta << ", " << logProbNll << ", " << logProb << std::endl;
        }
    }

private:
    std::ofstream file;
};

std::vector<Eigen::MatrixXd> RcppListToMatrixVector(Rcpp::List matrices)
{
    std::vector<Eigen::MatrixXd> eigenMatrices;

    for (int i = 0; i < matrices.size(); i++)
    {
        Rcpp::NumericMatrix mat = matrices[i];
        Eigen::Map<Eigen::MatrixXd> eigenMat(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(mat));
        eigenMatrices.push_back(eigenMat);
    }

    return eigenMatrices;
}

template <typename EigMat>
Eigen::Matrix<typename EigMat::Scalar, Eigen::Dynamic, Eigen::Dynamic> inverse_spd(const EigMat &m)
{
    if (m.size() == 0)
    {
        return {};
    }
    if (!m.isApprox(m.transpose()))
    {
        throw std::domain_error("Matrix must be symmetric");
    }

    Eigen::Matrix<typename EigMat::Scalar, Eigen::Dynamic, Eigen::Dynamic> mmt = 0.5 * (m + m.transpose());
    Eigen::LDLT<Eigen::Matrix<typename EigMat::Scalar, Eigen::Dynamic, Eigen::Dynamic>> ldlt(mmt);

    if (ldlt.info() != Eigen::Success)
    {
        throw std::domain_error("LDLT decomposition failed");
    }
    if (ldlt.isNegative())
    {
        throw std::domain_error("Matrix is not positive definite");
    }

    return ldlt.solve(Eigen::Matrix<typename EigMat::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(m.rows(), m.cols()));
}

template <typename RNG>
Eigen::VectorXd multi_normal_rng(const Eigen::VectorXd& mu,
                                 const Eigen::MatrixXd& S,
                                 RNG& rng) {
    if (S.rows() <= 0) {
        throw std::domain_error("Covariance matrix must have a positive number of rows.");
    }
    if (S.rows() != S.cols()) {
        throw std::domain_error("Covariance matrix must be square.");
    }
    if (!mu.allFinite()) {
        throw std::domain_error("Location parameter must be finite.");
    }
    if (!S.isApprox(S.transpose())) {
        throw std::domain_error("Covariance matrix must be symmetric.");
    }

    Eigen::LLT<Eigen::MatrixXd> llt_of_S = S.llt();
    if (llt_of_S.info() != Eigen::Success) {
        throw std::domain_error("Cholesky decomposition failed.");
    }

    boost::normal_distribution<> norm(0, 1);
    boost::variate_generator<RNG&, boost::normal_distribution<> > randn(rng, norm);
    Eigen::VectorXd z(S.cols());
    for (int i = 0; i < S.cols(); ++i) {
        z(i) = randn();
    }

    return mu + llt_of_S.matrixL() * z;
}

Eigen::MatrixXd kroniker(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    int m = A.rows();
    int n = A.cols();
    int p = B.rows();
    int q = B.cols();
    Eigen::MatrixXd C(m * p, n * q);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < p; ++k) {
                for (int l = 0; l < q; ++l) {
                    C(p * i + k, q * j + l) = A(i, j) * B(k, l);
                }
            }
        }
    }
    return C;
}

template <class RNG>
Eigen::MatrixXd wishart_rng(double nu, const Eigen::MatrixXd &S, RNG &rng){
    int dim = S.rows();
    Eigen::LLT<Eigen::MatrixXd> lltOfS(S);  
    Eigen::MatrixXd chol = lltOfS.matrixL(); 

    boost::random::normal_distribution<> normal_dist(0, 1);
    Eigen::MatrixXd foo = Eigen::MatrixXd::Zero(dim, dim);

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (i == j) {
                boost::random::chi_squared_distribution<> chi_squared_dist(nu - i);
                foo(i, j) = std::sqrt(chi_squared_dist(rng));
            } else {
                foo(i, j) = normal_dist(rng);
            }
        }
    }

    Eigen::MatrixXd result = chol * (foo * (foo.transpose() * chol.transpose()));
    return result;
}


template <class RNG>
Eigen::MatrixXd inv_wishart_rng(double nu, const Eigen::MatrixXd &S, RNG &rng)
{
    int k = S.rows();
    if (S.rows() != S.cols())
    {
        throw std::domain_error("Scale matrix must be square");
    }
    if (!S.isApprox(S.transpose()))
    {
        throw std::domain_error("Scale matrix must be symmetric");
    }
    if (nu <= k - 1)
    {
        throw std::domain_error("Degrees of freedom must be greater than dimension of scale matrix minus 1");
    }

    return wishart_rng(nu,S.inverse(),rng).inverse();
}


template <class RNG>
Eigen::MatrixXd matrix_normal_rng(const Eigen::MatrixXd& M, const Eigen::MatrixXd& C, const Eigen::MatrixXd& Sigma, RNG& rng) {
    int p = M.cols();
    int q = M.rows();

    boost::random::normal_distribution<> norm(0, 1);

    Eigen::MatrixXd X(q, p);
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < p; ++j) {
            X(i, j) = norm(rng);
        }
    }

    Eigen::LLT<Eigen::MatrixXd> llt_of_Sigma = Sigma.llt();
    Eigen::LLT<Eigen::MatrixXd> llt_of_C = C.llt();
    
    Eigen::MatrixXd theta = M + llt_of_C.matrixL() * X * llt_of_Sigma.matrixL().transpose();
    return theta;
}

#endif