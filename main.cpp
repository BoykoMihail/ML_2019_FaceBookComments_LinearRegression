/* 
 * File:   main.cpp
 * Author: boyko_mihail
 *
 * Created on 8 сентября 2019 г., 11:34
 */

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <random>
#include "rapidcsv.h"
#include "LinearRegression.h"

using namespace std;

/*
 * 
 */




int main(int argc, char** argv) {

    cout << "Hellow World!" << endl;
    
    rapidcsv::Document doc("/home/boyko_mihail/NetBeansProjects/FaceBookFirstHW/Dataset/Dataset/Training/Features_Variant_1.csv");
    
    int crossValCount = doc.GetRowCount()/5;
    std::vector<double> RMSE_results(0);
    std::vector<double> R2_results(0);
    
    std::vector<std::vector<double>> All(0);
    for (int i = 0; i<doc.GetRowCount(); ++i){
        std::vector<double> newX = doc.GetRow<double>(i);
//        for(int k=0; k<newX.size(); ++k){
//            if(newX[k] != 0){
//                newX[k] = log(newX[k]);
//            }
//        }
        auto y_t = newX.back();
        newX.pop_back();
        newX.push_back(1);
        newX.push_back(y_t);
        All.push_back(newX);
    } 
    auto rng = std::default_random_engine{};
    shuffle(All.begin(), All.end(), rng);
    
    std::vector<std::vector<double>> X(0);
    std::vector<double> Y(0);
    for (int i = 0; i<All.size(); ++i){
        std::vector<double> newX = All[i];
        Y.push_back(newX.back());
        newX.pop_back();
        X.push_back(newX);
    } 
    std::vector<int> vv(0);
    for(int i=0; i<Y.size(); ++i){
        vv.push_back(i);
    }
    
    for(int i=0; i<Y.size(); ++i){
        if(vv[i] != i ){
            swap(X[i], X[vv[i]]);
            swap(Y[i], Y[vv[i]]);
            i=0;
        }
    }
    
    std::vector<std::vector<double>> X_train(0);
    std::vector<std::vector<double>> X_test(0);

    std::vector<int> Y_train(0);
    std::vector<int>Y_test(0);
    
        
    for(int i = 0; i<5; i ++){
        
        X_train.clear();
        X_test.clear();
        Y_train.clear();
        Y_test.clear();
        
        std::vector<int> Y_train(0);
        std::vector<int>Y_test(0);
        cout<<"x[0].size() = "<< X[0].size()<<endl;
        cout<<"X.size() = "<<X.size()<<endl;
        for(int j=0;j<X.size(); j++){
            if(j < crossValCount*i || j >= crossValCount*(i+1)) {
                X_train.push_back(X[j]);
                Y_train.push_back(Y[j]);
            } else {
                X_test.push_back(X[j]);
                Y_test.push_back(Y[j]);
            }
        }

        cout<<"X_train.size() = "<<X_train.size()<<endl;
        
        LinearRegression model(X_train,Y_train);   
        model.fit();

        auto Y_pred = model.predict(X_train);
        double result_RMSE = model.calRMSE(Y_pred, Y_train); 
        double result_R2 = model.calR2(Y_pred, Y_train); 

        cout<< "result RMSE trening iteretion #"<<i<<" = " << result_RMSE << endl; 
        cout<< "result R^2 trening iteretion #"<<i<<" = " << result_R2 << endl; 
        
        auto Y_pred_test = model.predict(X_test);
        double result_RMSE_test = model.calRMSE(Y_pred_test, Y_test); 
        double result_R2_test = model.calR2(Y_pred_test, Y_test); 

        cout<< "result RMSE test iteretion #"<<i<<" = " << result_RMSE_test << endl; 
        cout<< "result R^2 test iteretion #"<<i<<" = " << result_R2_test << endl << endl << endl; 
        
        RMSE_results.push_back(result_RMSE_test);
        R2_results.push_back(result_R2_test);
    }
    
    double R2_sum = 0;
    double RMSE_sum = 0;
    for(int i=0; i<R2_results.size(); ++i){
        R2_sum += R2_results[i];
        RMSE_sum += RMSE_results[i];
    }
    cout << "M RMSE = " << RMSE_sum/5;
    cout << "M R2 = " << R2_sum/5;
    
    
    double R2_sum2 = 0;
    double RMSE_sum2 = 0;
    for(int i=0; i<R2_results.size(); ++i){
        R2_sum2 += (R2_results[i]-R2_sum/5)*(R2_results[i]-R2_sum/5);
        RMSE_sum2 += (RMSE_results[i]-RMSE_sum/5)*(RMSE_results[i]-RMSE_sum/5);
    }
    
    cout << " Sig RMSE = " << sqrt(RMSE_sum2/5);
    cout << " Sig R2 = " << sqrt(R2_sum2/5);

   
    return 0;
}



