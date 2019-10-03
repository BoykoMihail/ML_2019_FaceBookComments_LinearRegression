
#include <cassert>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>
#include "LinearRegression.h"
#include "Statistic.h"
#include "RMSE_metric.h"
#include "R2_metric.h"
#include <eigen3/Eigen/Core>

using namespace std;
using namespace Eigen;

//using Eigen::MatrixXd;
//using Eigen::VectorXd;

LinearRegression::LinearRegression(double learning_rate, int numEpoh, int bach_size, Regularization regul) {

    this->regul = regul;
    this->learning_rate = learning_rate;
    this->numEpoh = numEpoh;
    this->bach_size = bach_size;
}

void LinearRegression::normVectro(MatrixXd &v) {

    for (int i = 0; i < v.cols() - 1; ++i) {
        double m = 0;
        double sig = 0;
        Statistic::findeStatistic(v.col(i), m, sig);

        for (int j = 0; j < v.rows(); ++j) {
            if (sig != 0) {
                v(j, i) = (v(j, i) - m) / sig;
            } else {
                v(j, i) = 1;
            }
//            if (sig > 0 && fabs(v(j, i) - m) > 3 * sig) {
//                if (v(j, i) - m > 3 * sig) {
//                    v(j, i) = m + 3 * sig;
//                } else if (v(j, i) - m < -(3 * sig)) {
//                    v(j, i) = m - 3 * sig;
//                }
//            }
        }
    }
}

VectorXd LinearRegression::predict_value(const VectorXd &ntheta, const MatrixXd &features) {

    VectorXd v(features * ntheta);
    return v;
}

VectorXd LinearRegression::gradientDescent(MatrixXd &X, VectorXd &Y) {

    normVectro(X);
    int k = 0;
    VectorXd lastDiff = VectorXd();

    while (k < numEpoh) {
        VectorXd newW = this->W;
        lastDiff.setZero(bach_size);
        //        PermutationMatrix<Dynamic, Dynamic> perm(Y.size());
        //        perm.setIdentity();
        //        std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
        //        // A_perm = X * perm; // permute columns
        //        MatrixXd X_perm = perm * X; // permute rows
        //        VectorXd Y_perm = perm * Y;
//        if (k % 100 == 0)
//            cout << "learning_rate = " << learning_rate << endl << endl;
        for (int i = 0; i < X.rows(); i += this->bach_size) {

            MatrixXd bachX;
            VectorXd bachY;
            if (i + bach_size < X.rows()) {

                bachX = X.block(i, 0, bach_size, X.cols());
                bachY = Y.block(i, 0, bach_size, 1);
            } else {
                bachX = X.block(i, 0, X.rows() - i, X.cols());
                bachY = Y.block(i, 0, X.rows() - i, 1);
            }


            VectorXd current_predict_value = predict_value(newW, bachX);
            VectorXd diff = (bachY - current_predict_value).array() / (((bachY - current_predict_value).array()*(bachY - current_predict_value).array()).sqrt()).array();
            diff = diff.array() * learning_rate;
            //            cout<<"diff size = "<<diff.size()<<endl;
            //            cout<<"learning_rate = "<<learning_rate<<endl<<endl;
            //                double tete = 0 * newW(j);
            //                switch (regul) {
            //                    case Regularization::NONE:
            //                        tete = 0;
            //                        break;
            //                    case Regularization::L1_REGULARIZATION:
            //                        if (newW(j) > 0) {
            //                            tete = 0.009;
            //                        } else if (newW(j) < 0) {
            //                            tete = -0.009;
            //                        }
            //                        break;
            //                    case Regularization::L2_REGULARIZATION:
            //                        tete = 0.6 * newW(j);
            //                        break;
            //                }
            newW = (newW.array() + ((diff.transpose() * bachX) / bachY.size()).transpose().array()).transpose();
            W = newW;

            srand(time(NULL));
            int indexCurrent = rand() % diff.size();

//            if (lastDiff[indexCurrent] * diff[indexCurrent] <= 0) {
              //  if (learning_rate > 0.0015){
                    learning_rate *= 0.99999999;
               // } else {
                   // learning_rate *= 0.9999999;
               // }
                lastDiff = diff;
//            } else {
//                learning_rate *= 1.00000001;
//                lastDiff = diff;
//            }
        }
        ++k;
    }
    //             for (int i = 0; i < W.size(); i++)
    //                {
    //                    cout << " W[i] = " << W[i] << " i = " << i ;
    //              
    //                }
    //            cout<<endl;
    return W;
}

void LinearRegression::fit(const std::vector<std::vector<double>> &X, const std::vector<double> &Y) {

    srand(time(NULL));

    MatrixXd X_Matrix(X.size(), X[0].size());
    for (int i = 0; i < X.size(); ++i) {
        X_Matrix.row(i) = VectorXd::Map(X[i].data(), X[i].size());
    }

    VectorXd Y_Vector = VectorXd::Map(Y.data(), Y.size());

    this->W = VectorXd();
    W.setRandom(X[0].size());

    this->gradientDescent(X_Matrix, Y_Vector);

}

std::vector<double> LinearRegression::predict(const std::vector< std::vector<double>> X_test) {

    MatrixXd X_Matrix(X_test.size(), X_test[0].size());
    for (int i = 0; i < X_test.size(); ++i) {
        X_Matrix.row(i) = VectorXd::Map(X_test[i].data(), X_test[i].size());
    }

    normVectro(X_Matrix);

    VectorXd vecPred = predict_value(W, X_Matrix);
    std::vector<double> predict_Y(vecPred.data(), vecPred.data() + vecPred.rows() * vecPred.cols());
    return predict_Y;
}



