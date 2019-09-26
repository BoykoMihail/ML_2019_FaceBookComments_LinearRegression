/* 
 * File:   main.cpp
 * Author: boyko_mihail
 *
 * Created on 8 сентября 2019 г., 11:34
 */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <random>
#include "rapidcsv.h"
#include "LinearRegression.h"
#include "RMSE_metric.h"
#include "R2_metric.h"
#include "Statistic.h"

using namespace std;

/*
 * 
 */


double rountFor(double a, int count=6){
    double ten = 1.0;
    for(int i=0;i<count;++i)
        ten*=10;
    return round(a*ten)/ten;
}


int main(int argc, char** argv) {

    cout << "Hellow World!" << endl;
    
    rapidcsv::Document doc("/home/boyko_mihail/NetBeansProjects/FaceBookFirstHW/Dataset/Dataset/Training/Features_Variant_1.csv");
    
    int crossValCount = doc.GetRowCount()/5;
    std::vector<double> RMSE_results(0);
    std::vector<double> R2_results(0);
    std::vector<std::vector<double>> all_W(0);
//    
    std::vector<std::vector<double>> All(0);
    for (int i = 0; i<doc.GetRowCount(); ++i){
        std::vector<double> newX = doc.GetRow<double>(i);
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
    
    std::vector<std::vector<double>> X_train(0);
    std::vector<std::vector<double>> X_test(0);

    std::vector<int> Y_train(0);
    std::vector<int>Y_test(0);
    
        
    for(int i = 0; i<5; i ++){
       
        X_train.clear();
        X_test.clear();
        Y_train.clear();
        Y_test.clear();
        
        for(int j=0;j<X.size(); j++){
            if(j < crossValCount*i || j >= crossValCount*(i+1)) {
                X_train.push_back(X[j]);
                Y_train.push_back(Y[j]);
            } else {
                X_test.push_back(X[j]);
                Y_test.push_back(Y[j]);
            }
        }
         
        LinearRegression model(500.1, 160, Regularization::NONE);  
        model.fit(X_train, Y_train);

        auto Y_pred = model.predict(X_train);
        double result_RMSE = RMSE_metric::calculateMetric(Y_pred, Y_train );
        double result_R2 = R2_metric::calculateMetric(Y_pred, Y_train);

        cout<< "result RMSE trening iteretion #"<<i<<" = " << result_RMSE << endl; 
        cout<< "result R^2 trening iteretion #"<<i<<" = " << result_R2 << endl; 
        
        auto Y_pred_test = model.predict(X_test);
        double result_RMSE_test = RMSE_metric::calculateMetric(Y_pred_test, Y_test); 
        double result_R2_test = R2_metric::calculateMetric(Y_pred_test, Y_test); 

        cout<< "result RMSE test iteretion #"<<i<<" = " << result_RMSE_test << endl; 
        cout<< "result R^2 test iteretion #"<<i<<" = " << result_R2_test << endl << endl << endl; 
        
        RMSE_results.push_back(result_RMSE_test);
        R2_results.push_back(result_R2_test);
        all_W.push_back(model.getW());
    }
    
    double R2_M = 0;
    double RMSE_M = 0;
    
    double R2_sig = 0;
    double RMSE_sig = 0;
    
    
    Statistic::findeStatistic(R2_results, R2_M, R2_sig );
    Statistic::findeStatistic(RMSE_results, RMSE_M, RMSE_sig );
    
    std::ofstream outFile;
    
    cout<< "all RMSE Mean = " << RMSE_M << endl; 
    cout<< "all RMSE Sigma = " << RMSE_sig << endl;
    cout<< "all R2 Mean = " << R2_M << endl; 
    cout<< "all R2 Sigma = " << R2_sig << endl; 
        
        
    std::ofstream myfile;
    myfile.open("/home/boyko_mihail/NetBeansProjects/ML_Facebook_LinearRegression/ML_2019_FaceBookComments_LinearRegression/Result_Table.csv");
    myfile <<",1,2,3,4,5,E,SD,\n";
    myfile <<"RMSE," << (RMSE_results[0]) << "," << (RMSE_results[1]) << "," << (RMSE_results[2]) << "," << (RMSE_results[3]) << "," << (RMSE_results[4]) << "," << RMSE_M << "," << RMSE_sig << ",\n";
    myfile <<"R^2," << (R2_results[0]) << "," << (R2_results[1]) << "," << (R2_results[2]) << "," << (R2_results[3]) << "," << (R2_results[4]) << "," << R2_M << "," << R2_sig << ",\n";
    
    
    for(int i = 0; i<all_W[0].size(); i++){
       double W_i_M = 0;
       double W_i_Sig = 0;
       for(int k=0; k<all_W.size(); ++k){
           W_i_M += all_W[k][i];
           W_i_Sig += all_W[k][i]*all_W[k][i];
       } 
       W_i_M = W_i_M/all_W.size();
       W_i_Sig = sqrt(W_i_Sig/all_W.size() - W_i_M*W_i_M);
       myfile << "W["<<i<<"]," <<  all_W[0][i] << "," << all_W[1][i] << "," << all_W[2][i] << "," << all_W[3][i] << "," << all_W[4][i] << "," << W_i_M << "," << W_i_Sig <<",\n";
    }
    
    myfile.close();
   
    return 0;
}






//    outFile.open("/home/boyko_mihail/NetBeansProjects/ML_Facebook_LinearRegression/ML_2019_FaceBookComments_LinearRegression/README.md", std::ios_base::app);
//    outFile<<"                     |           1        |         2          |          3         |         4          |        5           |       E            |          SD        |\n";
//    outFile<<"| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |\n";
//    outFile<<"RMSE                 |      "<<rountFor(RMSE_results[0])<<"      |      "<<rountFor(RMSE_results[1])<<"      |      "<<rountFor(RMSE_results[2])<<"      |      "<<rountFor(RMSE_results[3])<<"      |      "<<rountFor(RMSE_results[4])<<"      |      "<<rountFor(RMSE_M)<<"      |      "<<rountFor(RMSE_sig)<<"      |\n";
//    outFile<<"| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |\n";
//    outFile<<"R^2                  |      "<<rountFor(R2_results[0])<<"      |      "<<rountFor(R2_results[1])<<"      |      "<<rountFor(R2_results[2])<<"      |      "<<rountFor(R2_results[3])<<"      |      "<<rountFor(R2_results[4])<<"      |      "<<rountFor(R2_M)<<"      |      "<<rountFor(R2_sig)<<"      |\n";
//    outFile<<"| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |\n";
//    for(int i = 0; i<all_W[0].size(); i++){
//        
//        double W_i_M = 0;
//        double W_i_Sig = 0;
//        for(int k=0; k<all_W.size(); ++k){
//            W_i_M += all_W[k][i];
//            W_i_Sig += all_W[k][i]*all_W[k][i];
//        } 
//        W_i_M = W_i_M/all_W.size();
//        W_i_Sig = sqrt(W_i_Sig/all_W.size() - W_i_M*W_i_M);
//        
//        auto c = i>9 ? "" : " ";
//        outFile<<"W["<<i<<c<<"]                |      "<<rountFor(all_W[0][i])<<"      |      "<<rountFor(all_W[1][i])<<"      |      "<<rountFor(all_W[2][i])<<"      |      "<<rountFor(all_W[3][i])<<"      |      "<<rountFor(all_W[4][i])<<"      |      "<<rountFor(W_i_M)<<"      |      "<<rountFor(W_i_Sig)<<"      |\n";
//        outFile<<"| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |\n";
//    }
