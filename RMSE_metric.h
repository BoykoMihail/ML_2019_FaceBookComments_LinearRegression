/* 
 * File:   RMSE_metric.h
 * Author: boyko_mihail
 *
 * Created on 24 сентября 2019 г., 12:28
 */
#include "Metric.h"

#ifndef RMSE_METRIC_H
#define	RMSE_METRIC_H

class RMSE_metric : public Metric {
    public:
        static double calculateMetric(const std::vector<int> Y_pred, const std::vector<int> Y_test){
        
            double sum = 0;  
            for(int i = 0; i<Y_pred.size(); i++){
                sum += (Y_test[i] - Y_pred[i])*(Y_test[i] - Y_pred[i]);
            }
            return sqrt(sum/Y_pred.size());
        }
};

#endif	/* RMSE_METRIC_H */

