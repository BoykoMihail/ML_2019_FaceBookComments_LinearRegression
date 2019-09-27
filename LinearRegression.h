/* 
 * File:   LinearRegression.h
 * Author: boyko_mihail
 *
 * Created on 8 сентября 2019 г., 21:34
 */

#include <vector>
#include <eigen3/Eigen/Core>

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;


#ifndef LINEARREGRESSION_H
#define	LINEARREGRESSION_H

enum Regularization {
    NONE,
    L2_REGULARIZATION,
    L1_REGULARIZATION
};

class LinearRegression {
private:
    VectorXd W;

    Regularization regul;
    double learning_rate;
    int numEpoh;
    int bach_size;

    VectorXd predict_value(const VectorXd &ntheta, const MatrixXd &features);
    void normVectro(MatrixXd &v);
    VectorXd gradientDescent(MatrixXd &X, VectorXd &Y);

public:

    LinearRegression(double alpha, int numEpoh, int bach_size, Regularization regul);

    void fit(const std::vector<std::vector<double>> &X, const std::vector<double> &Y);
    std::vector<double> predict(const std::vector< std::vector<double>> X_test);

    void setAlpha(double newAlpha) {
        this->learning_rate = learning_rate;
    }

    void setRegul(Regularization newRegul) {
        this->regul = newRegul;
    }

    void setNumEpoh(double newNumEpoh) {
        this->numEpoh = newNumEpoh;
    }

    std::vector<double> getW() {
        std::vector<double> w(0);
        for (int i = 0; i < W.size(); ++i) {
            w.push_back(W(i));
        }
        return w;
    }
};

#endif	/* LINEARREGRESSION_H */

