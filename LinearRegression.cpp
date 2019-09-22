
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

using namespace std;  

    LinearRegression::LinearRegression(std::vector<std::vector<double>> &x, std::vector<int> &label){
        
        
        this->regul = Regularization::NONE ;
        this->alpha = 0.8;
        this->epsilon = 0.00000000001;
        this->numEpoh = 10;
    
        srand(time(NULL));
        this->mExamples = label.size();
        this->nFeatures = x[0].size();
        this->X_data = x;
        
        this->Y_label = label;
        double labelMean = this->findeMean(Y_label);
        double labelSig = this->findeSigma(Y_label, labelMean);
        for (int i=0; i<Y_label.size(); ++i){
            if(labelSig > 0 && fabs(Y_label[i] - labelMean) > 3*labelSig) {
                Y_label[i] = labelMean;
            }
        }
        this->W = vector<double>(nFeatures);
        for(int i = 0; i<nFeatures; i++){
            double temp = ((double) rand() / (RAND_MAX)) ;
            W[i] = (temp);
        }
        normVectro(X_data);
        
    }

    
    double LinearRegression::findeMean(const std::vector<double> &v){
        
        float summOfElements = 0;
        for (int i = 0; i < v.size(); ++i){
            summOfElements += v[i];
        }
        return summOfElements/v.size();
    }
    
    double LinearRegression::findeMean(const std::vector<int> &v){
        
        float summOfElements = 0;
        for (int i = 0; i < v.size(); ++i){
            summOfElements += v[i];
        }
        return summOfElements/v.size();
    }

    double LinearRegression::findeSigma(const std::vector<double> &v, double m1){

        double sum = 0;
        for (int i = 0; i < v.size(); i++){
            sum += (v[i] - m1)*(v[i] - m1);
        }
        sum /= v.size();
        return sqrt(sum);

    }
    
    double LinearRegression::findeSigma(const std::vector<int> &v, double m1){

        double sum = 0;
        for (int i = 0; i < v.size(); i++){
            sum += (v[i] - m1)*(v[i] - m1);
        }
        sum /= v.size();
        return sqrt(sum);
    }
    

    void LinearRegression::normVectro(std::vector<std::vector<double>> &v){

        for(int i = 0; i<v[0].size()-1; ++i){
            std::vector<double> ex(0);
            for(int j=0; j<v.size(); ++j){
                ex.push_back(v[j][i]);
            }
            float m = findeMean(ex);
            float sig = findeSigma(ex, m);
            for (int j=0; j<v.size(); ++j){
                if (sig != 0){
                    v[j][i] = (v[j][i] - m)/sig;
                } else {
                    v[j][i] = (v[j][i] - m);
                }
                if(sig > 0 && fabs(v[j][i] - m) > 3*sig) {
                    v[j][i] = m;
                }
            }
        }
    }
    
    double LinearRegression::H(vector<double>& ntheta, vector<double>& features){
        double sum = 0.0;
            for (int i = 0; i < features.size(); i++){
                sum += ntheta[i]*features[i];
            }
            
            return sum;
    }
    
    
    
    vector<double> LinearRegression::gradientDescent(){
        bool converge = true;
        int debug = 0;
        int k = 0;
        while (converge) {
            
            vector<double> newW = this->W;
            k++;
            
            for (int i = 0; i < mExamples; i++){ 
                
                alpha/=(i+1);
                double HH = H(newW, X_data[i]);
                double diff = ( Y_label[i] - HH) / sqrt((Y_label[i] - HH)*(Y_label[i] - HH)) ;
                diff *= alpha;
                for (unsigned j = 0; j < nFeatures; j++){
                    double tete = 0*newW[j];
                    newW[j] += ((diff * this->X_data[j][i])/Y_label.size() + tete);
                }
                W = newW;
//                if (i%2500 == 0) {
//                    auto Y_pred = this->predict(X_data);
//                    double result = this->calRMSE(Y_pred, Y_label); 
//                    double result2 = this->calR2(Y_pred, Y_label); 
//                    cout<< "result Rmse  " << i << " = " << result << endl; 
//                     cout<< "result R2 trening " << i << " = " << result2 << endl << endl; 
//                }
            }
            converge = true;
            for (unsigned i = 0; i < W.size(); i++){
                converge = (fabs(W[i]-newW[i]) > epsilon) && k<numEpoh;
            }
            W = newW;
            
        }
//         for (int i = 0; i < W.size(); i++)
//            {
//                cout << " W[i] = " << W[i] << " i = " << i ;
//          
//            }
//        cout<<endl;
        return W;
    }
    
    
    
    void LinearRegression::fit(){
        this->gradientDescent();
    }
    
    
    
    std::vector<int> LinearRegression::predict(std::vector< std::vector<double>> X_test){

        normVectro(X_test);
        std::vector<int> predict_Y(0);
        for(int i = 0; i<X_test.size(); i++){
            predict_Y.push_back(H(W, X_test[i]));
        }
        return predict_Y;
    }
    
    
    double LinearRegression::calRMSE(const std::vector<int> Y_pred, const std::vector<int> Y_test){
        
        double sum = 0;  
        for(int i = 0; i<Y_pred.size(); i++){
            sum += (Y_test[i] - Y_pred[i])*(Y_test[i] - Y_pred[i]);
        }
        return sqrt(sum/Y_pred.size());
    }
  
    
    double LinearRegression::calR2(const std::vector<int> Y_pred, const std::vector<int> Y_test){
        
        double sum_Up = 0;    
        for(int i = 0; i<Y_pred.size(); i++){
            sum_Up += ( Y_test[i] - Y_pred[i])*( Y_test[i] - Y_pred[i]);
        }
        double sum_down = 0;
        double Y_mean =  this->findeMean(Y_test);
        for(int i = 0; i<Y_pred.size(); i++){
            sum_down += ( Y_test[i] - Y_mean)*( Y_test[i] - Y_mean);
        }
        return 1 - (sum_Up/sum_down);
    }
   
    
