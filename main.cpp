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
#include <time.h>
#include <map>
#include <sstream>

using namespace std;

double rountFor(double a, int count = 6) {
    double ten = 1.0;
    for (int i = 0; i < count; ++i)
        ten *= 10;
    return round(a * ten) / ten;
}

int main(int argc, char** argv) {

    cout << "Hellow World!" << endl;



    //rapidcsv::Document doc("/home/boyko_mihail/NetBeansProjects/ML_Facebook_LinearRegression/ML_2019_FaceBookComments_LinearRegression/Dataset/Dataset/Training/Features_Variant_1.csv");
    rapidcsv::Document doc("/home/boyko_mihail/NetBeansProjects/ML_Facebook_LinearRegression/ML_2019_FaceBookComments_LinearRegression/Features_new2.csv");

    int crossValCount = doc.GetRowCount() / 5;
    std::vector<double> RMSE_results(0);
    std::vector<double> R2_results(0);
    std::vector<std::vector<double>> all_W(0); 
    map <int, string> labelFeatures;

    std::vector<string> newXlabel = doc.GetRow<string>(0);

   
    for (int i = 0; i < newXlabel.size(); ++i) {
        labelFeatures[i] = newXlabel[i];
    }

    std::vector<std::vector<double>> All(0);
    for (int i = 1; i < doc.GetRowCount(); ++i) {
        std::vector<double> newX = doc.GetRow<double>(i);
        All.push_back(newX);
    }
    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(All.begin(), All.end(), g);


    std::vector<std::vector<double>> X(0);
    std::vector<std::vector<double>> XUpdateFeatures(0);
    std::vector<double> Y(0);

   
    //
    //    for (int i = 0; i < All[0].size() - 1; ++i) {
    //        std::ostringstream oss;
    //        oss << "X[" << i << "]";
    //        labelFeatures[i] = oss.str();
    //    }
    for (int i = 0; i < All.size(); ++i) {
        std::vector<double> newX = All[i];
        Y.push_back(newX.back());
        newX.pop_back();
        //        int sizeX = newX.size();
        //        for (int q = 0; q < sizeX; ++q) {
        //            for (int q2 = q; q2 < sizeX; ++q2) {
        //                newX.push_back(newX[q] * newX[q2]);
        //                std::ostringstream oss;
        //                oss << labelFeatures[q] << " * " << labelFeatures[q2] << "";
        //                labelFeatures[newX.size() - 1] = oss.str();
        //            }
        //        }
        //        for (int w = 0; w < 24; ++w) {
        //            if (w != newX[38]) {
        //                newX.push_back(0);
        //            } else {
        //                newX.push_back(1);
        //            }
        //        }
        //        newX.push_back(1);
        //labelFeatures[newX.size() - 1] = "X_0";
        X.push_back(newX);
    }

    std::vector<std::vector<double>> X_train(0);
    std::vector<std::vector<double>> X_test(0);

    std::vector<double> Y_train(0);
    std::vector<double>Y_test(0);

    //    map <double, double> pXi;
    //    map <pair<double, double>, double> pXiYi;
    //    map <int, double> HYX;
    //    map <double, double> pYi;
    //    double HY = 0;
    //    map < int, double> IG_YX;


    //    for (int j = 0; j < Y.size(); ++j) {
    //        pYi[Y[j]] += 1.0;
    //    }
    //    for (auto it = pYi.begin(); it != pYi.end(); ++it) {
    //        HY += ((*it).second / Y.size()) * log2(((*it).second / Y.size()));
    //    }
    //    HY *= -1;
    //
    //    for (int j = 0; j < X[0].size(); ++j) {
    //        for (int q = 0; q < X.size(); ++q) {
    //            pXi[X[q][j]] += 1.0;
    //            pXiYi[pair<double, double>(X[q][j], Y[q])] += 1.0;
    //
    //        }
    //
    //        for (auto it = pXi.begin(); it != pXi.end(); ++it) {
    //            double tempSumm = 0;
    //            for (auto ity = pXiYi.begin(); ity != pXiYi.end(); ++ity) {
    //                if ((*ity).first.first == (*it).first) {
    //                    tempSumm += ((*ity).second / (*it).second) * log2((*ity).second / (*it).second);
    //                }
    //            }
    //            tempSumm *= -1;
    //            HYX[j] += ((*it).second / X.size()) * tempSumm;
    //        }
    //        //HYX[j] *= -1;
    //        pXi.clear();
    //        pXiYi.clear();
    //    }
    //    for (auto it = HYX.begin(); it != HYX.end(); ++it) {
    //        IG_YX[(*it).first] = HY - (*it).second;
    //    }
    //
    //    //    auto cmp = [](std::pair<int, double> const & a, std::pair<int, double> const & b){
    //    //        return a.second != b.second? a.second < b.second : a.first < b.first;
    //    //    };
    //    //    std::sort(IG_YX.begin(), IG_YX.end(), cmp);
    //
    //    for (auto it = IG_YX.begin(); it != IG_YX.end(); ++it) {
    //        cout << "IG_YX[" << (*it).first << "] = " << (*it).second << endl;
    //    }
    //    cout << "HY = " << HY << endl;
    //
    //    map <int, string> newlabelFeatures;
    //    int indexnewLabel = 0;
    //    for (int j = 0; j < X[0].size(); ++j) {
    //        if (IG_YX[j] > 0.7 || j == X[0].size() - 1) {
    //            newlabelFeatures[indexnewLabel] = labelFeatures[j];
    //            ++indexnewLabel;
    //        }
    //    }
    //    labelFeatures = newlabelFeatures;
    //
    //    for (int i = 0; i < X.size(); ++i) {
    //        std::vector<double> newX(0);
    //        for (int j = 0; j < X[i].size(); ++j) {
    //            if (IG_YX[j] > 0.7 || j == X[i].size() - 1) {
    //                newX.push_back(X[i][j]);
    //            }
    //        }
    //        XUpdateFeatures.push_back(newX);
    //    }
    //
    //    X = XUpdateFeatures;

    //    std::ofstream myfileFeauters;
    //    myfileFeauters.open("/home/boyko_mihail/NetBeansProjects/ML_Facebook_LinearRegression/ML_2019_FaceBookComments_LinearRegression/Features_new2.csv");
    //
    //    for (int j = 0; j < X[0].size(); j++) {
    //        myfileFeauters << labelFeatures[j] << ",";
    //    }
    //    myfileFeauters << ",";
    //    myfileFeauters << "\n";

    //    for (int i = 0; i < X.size(); i++) {
    //        for (int j = 0; j < X[i].size(); j++) {
    //            myfileFeauters << X[i][j] << ",";
    //        }
    //        myfileFeauters << Y[i];
    //        myfileFeauters << "\n";
    //    }

    //    myfileFeauters.close();


    clock_t start = clock();
    for (int i = 0; i < 5; i++) {

        X_train.clear();
        X_test.clear();
        Y_train.clear();
        Y_test.clear();

        for (int j = 0; j < X.size(); j++) {

            if (j < crossValCount * i || j >= crossValCount * (i + 1)) {
                X_train.push_back(X[j]);
                Y_train.push_back(Y[j]);
            } else {
                X_test.push_back(X[j]);
                Y_test.push_back(Y[j]);
            }
        }


        // BATCH SIZE 60 PARAM = 0.1, 110, 60
        // BACH SIZE 1000 PARAM = 0.2, 220, 1000
        // BACH SIZE 1000 PARAM = 0.5, 400, 10000

        // new features param = 0.38, 70, 1000
        // old features param without ingen = 0.7, 140, 1000
        // old feature param with ingen = 2.7, 220, 1000 (0.6)
        LinearRegression model(0.35, 520, 1000, Regularization::NONE);
        clock_t start2 = clock();
        model.fit(X_train, Y_train);
        clock_t end2 = clock();

        double seconds2 = (double) (end2 - start2) / CLOCKS_PER_SEC;
        cout << " time to fit = " << seconds2 << " seconds" << endl << endl;


        auto Y_pred = model.predict(X_train);
        double result_RMSE = RMSE_metric::calculateMetric(Y_pred, Y_train);
        double result_R2 = R2_metric::calculateMetric(Y_pred, Y_train);

        cout << "result RMSE trening iteretion #" << i << " = " << result_RMSE << endl;
        cout << "result R^2 trening iteretion #" << i << " = " << result_R2 << endl;

        auto Y_pred_test = model.predict(X_test);
        double result_RMSE_test = RMSE_metric::calculateMetric(Y_pred_test, Y_test);
        double result_R2_test = R2_metric::calculateMetric(Y_pred_test, Y_test);

        cout << "result RMSE test iteretion #" << i << " = " << result_RMSE_test << endl;
        cout << "result R^2 test iteretion #" << i << " = " << result_R2_test << endl << endl << endl;

        RMSE_results.push_back(result_RMSE_test);
        R2_results.push_back(result_R2_test);
        all_W.push_back(model.getW());
    }
    clock_t end = clock();

    double seconds = (double) (end - start) / CLOCKS_PER_SEC;
    cout << " time = " << seconds << " seconds" << endl << endl;

    double R2_M = 0;
    double RMSE_M = 0;

    double R2_sig = 0;
    double RMSE_sig = 0;


    Statistic::findeStatistic(R2_results, R2_M, R2_sig);
    Statistic::findeStatistic(RMSE_results, RMSE_M, RMSE_sig);

    std::ofstream outFile;

    cout << "all RMSE Mean = " << RMSE_M << endl;
    cout << "all RMSE Sigma = " << RMSE_sig << endl;
    cout << "all R2 Mean = " << R2_M << endl;
    cout << "all R2 Sigma = " << R2_sig << endl;


    std::ofstream myfile;
    myfile.open("/home/boyko_mihail/NetBeansProjects/ML_Facebook_LinearRegression/ML_2019_FaceBookComments_LinearRegression/Features_enginering_8.csv");
    myfile << ",1,2,3,4,5,E,SD,\n";
    myfile << "RMSE," << (RMSE_results[0]) << "," << (RMSE_results[1]) << "," << (RMSE_results[2]) << "," << (RMSE_results[3]) << "," << (RMSE_results[4]) << "," << RMSE_M << "," << RMSE_sig << ",\n";
    myfile << "R^2," << (R2_results[0]) << "," << (R2_results[1]) << "," << (R2_results[2]) << "," << (R2_results[3]) << "," << (R2_results[4]) << "," << R2_M << "," << R2_sig << ",\n";


    for (int i = 0; i < all_W[0].size(); i++) {
        double W_i_M = 0;
        double W_i_Sig = 0;
        for (int k = 0; k < all_W.size(); ++k) {
            W_i_M += all_W[k][i];
            W_i_Sig += all_W[k][i] * all_W[k][i];
        }
        W_i_M = W_i_M / all_W.size();
        W_i_Sig = sqrt(W_i_Sig / all_W.size() - W_i_M * W_i_M);
        myfile << labelFeatures[i] << "," << all_W[0][i] << "," << all_W[1][i] << "," << all_W[2][i] << "," << all_W[3][i] << "," << all_W[4][i] << "," << W_i_M << "," << W_i_Sig << ",\n";
    }

    myfile.close();



    return 0;
}
