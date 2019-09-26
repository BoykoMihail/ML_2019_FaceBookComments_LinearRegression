
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

using namespace std;

LinearRegression::LinearRegression(double alpha, int numEpoh, Regularization regul) {

    this->regul = regul;
    this->alpha = alpha;
    this->numEpoh = numEpoh;
}

void LinearRegression::normVectro(std::vector<std::vector<double>> &v) {

    for (int i = 0; i < v[0].size() - 1; ++i) {
        std::vector<double> ex(0);
        for (int j = 0; j < v.size(); ++j) {
            ex.push_back(v[j][i]);
        }

        double m = 0;
        double sig = 0;

        Statistic::findeStatistic(ex, m, sig);

        for (int j = 0; j < v.size(); ++j) {
            if (sig != 0) {
                v[j][i] = (v[j][i] - m) / sig;
            } else {
                v[j][i] = (v[j][i] - m);
            }
            if (sig > 0 && fabs(v[j][i] - m) > 3 * sig) {
                v[j][i] = m;
            }
        }
    }
}

double LinearRegression::predict_value(const vector<double>& ntheta, const vector<double>& features) {
    double sum = 0.0;
    for (int i = 0; i < features.size(); ++i) {
        sum += ntheta[i] * features[i];
    }

    return sum;
}

vector<double> LinearRegression::gradientDescent(std::vector<std::vector<double>> &X, const std::vector<int> &Y) {

    normVectro(X);

    int k = 0;
    double lastDiff = 0;
    while (k < numEpoh) {
        vector<double> newW = this->W;

        for (int i = 0; i < X.size(); ++i) {
            //alpha/=(i+1);
            double current_predict_value = predict_value(newW, X[i]);
            double diff = (Y[i] - current_predict_value) / sqrt((Y[i] - current_predict_value)*(Y[i] - current_predict_value));
            diff *= alpha;
            for (unsigned j = 0; j < X[0].size(); ++j) {
                double tete = 0 * newW[j];
                switch (regul) {
                    case Regularization::NONE:
                        tete = 0;
                        break;
                    case Regularization::L1_REGULARIZATION:
                        if (newW[j] > 0) {
                            tete = 0.009;
                        } else if (newW[j] < 0) {
                            tete = -0.009;
                        }
                        break;
                    case Regularization::L2_REGULARIZATION:
                        tete = 0.6 * newW[j];
                        break;
                }
                newW[j] += ((diff * X[i][j]) / Y.size() - tete);
            }
            W = newW;
//            auto Y_pred = this->predict(X);
//            double currentLost = RMSE_metric::calculateMetric(Y_pred, Y);
            if (lastDiff*diff <= 0){
                alpha *= 0.9999999 ;
                lastDiff = diff;
            } else {
                alpha *=1.0000001;
            }
            
//                            if (i%2500 == 0) {
//                                cout<<" alpha = "<<alpha<<endl;
//                                auto Y_pred = this->predict(X);
//                                double result_RMSE_test = RMSE_metric::calculateMetric(Y_pred, Y); 
//                                double result_R2_test = R2_metric::calculateMetric(Y_pred, Y);  
//                                cout<< "result Rmse  " << i << " = " << result_RMSE_test << endl; 
//                                 cout<< "result R2 trening " << i << " = " << result_R2_test << endl << endl; 
//                            }
        }
        ++k;
    }
    //         for (int i = 0; i < W.size(); i++)
    //            {
    //                cout << " W[i] = " << W[i] << " i = " << i ;
    //          
    //            }
    //        cout<<endl;
    return W;
}

void LinearRegression::fit(const std::vector<std::vector<double>> &X, const std::vector<int> &Y) {

    srand(time(NULL));

    std::vector<std::vector<double>> new_X(0);
    for (int i = 0; i < X.size(); ++i) {
        std::vector<double> t(0);
        for (int j = 0; j < X[i].size(); ++j) {
            t.push_back(X[i][j]);
        }
        new_X.push_back(t);
    }
    //        
    //        std::vector<std::vector<double>> new_X(0);
    //        for(int i = 0; i<X.size(); ++i ){
    //            std::vector<double> t(0);
    //            for(int j = 0; j<X[i].size(); ++j){
    //                if(X[i][j] > 0){
    //                    t.push_back(log(X[i][j]));
    //                } else if(X[i][j] < 0) {
    //                    t.push_back(-log(-X[i][j]));
    //                }else {
    //                    t.push_back(X[i][j]);
    //                }
    //            }
    //            new_X.push_back(t);
    //        }
    //        std::vector<int> new_Y(Y.size());
    //        for(int i = 0; i<Y.size(); ++i ){
    //            new_Y[i] = log(Y[i]);
    //        }

    this->W = vector<double>(X[0].size());
    for (int i = 0; i < X[0].size(); i++) {
        double temp = ((double) rand() / (RAND_MAX));
        //if(temp<0.5){temp+=0.5;}
        W[i] = (temp);
    }

    this->gradientDescent(new_X, Y);


}

std::vector<int> LinearRegression::predict(const std::vector< std::vector<double>> X_test) {

    std::vector<std::vector<double>> new_X(0);
    for (int i = 0; i < X_test.size(); ++i) {
        std::vector<double> t(0);
        for (int j = 0; j < X_test[i].size(); ++j) {
            t.push_back(X_test[i][j]);
        }
        new_X.push_back(t);
    }

    normVectro(new_X);

    std::vector<int> predict_Y(0);
    for (int i = 0; i < new_X.size(); i++) {
        predict_Y.push_back(predict_value(W, new_X[i]));
    }
    return predict_Y;
}



