/* 
 * File:   LinearRegression.h
 * Author: boyko_mihail
 *
 * Created on 8 сентября 2019 г., 21:34
 */

#include <vector>
#ifndef LINEARREGRESSION_H
#define	LINEARREGRESSION_H

enum Regularization
{
    NONE,
    L2_REGULARIZATION,
    L1_REGULARIZATION
};

class LinearRegression {
private:
    std::vector<std::vector<double>> X_data;
    std::vector<int> Y_label;
    std::vector<double> W;
    unsigned mExamples, nFeatures;
    
    Regularization regul;
    double alpha;
    double epsilon;
    int numEpoh;
public:
    LinearRegression(std::vector<std::vector<double>> &x, std::vector<int> &label);
    void normVectro(std::vector<std::vector<double>> &v);
    double findeMean(const std::vector<double> &v);
    double findeMean(const std::vector<int> &v);
    double findeSigma(const std::vector<double> &v, double m1);
    double findeSigma(const std::vector<int> &v, double m1);
    void fit();
    std::vector<int> predict(const std::vector< std::vector<double>> X_test);
    double calRMSE(std::vector<int> Y_pred, const std::vector<int> Y_test);
    double calR2(std::vector<int> Y_pred, const std::vector<int> Y_test);
    double H(std::vector<double>& ntheta, std::vector<double>& features);
    std::vector<double> gradientDescent();
    
    void setAlpha(double newAlpha){
        this->alpha = newAlpha;
    }
    void setRegul(Regularization newRegul){
        this->regul = newRegul;
    }
    void setEpsilon(double newEpsilon){
        this->epsilon = newEpsilon;
    }
    void setNumEpoh(double newNumEpoh){
        this->numEpoh = newNumEpoh;
    }
};

#endif	/* LINEARREGRESSION_H */

