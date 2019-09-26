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
    std::vector<double> W;
    
    Regularization regul;
    double alpha;
    int numEpoh;
    
    double predict_value(const std::vector<double>& ntheta,const std::vector<double>& features);
    void normVectro(std::vector<std::vector<double>> &v);
    std::vector<double> gradientDescent(std::vector<std::vector<double>> &X,const std::vector<int> &Y);
    
public:
    
    LinearRegression(double alpha, int numEpoh, Regularization regul );
   
    void fit(const std::vector<std::vector<double>> &X,const std::vector<int> &Y);
    std::vector<int> predict(const std::vector< std::vector<double>> X_test);
    
    void setAlpha(double newAlpha){
        this->alpha = newAlpha;
    }
    void setRegul(Regularization newRegul){
        this->regul = newRegul;
    }
    void setNumEpoh(double newNumEpoh){
        this->numEpoh = newNumEpoh;
    }
    
    std::vector<double> getW(){
        return this->W;
    }
};

#endif	/* LINEARREGRESSION_H */

