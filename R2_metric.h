/* 
 * File:   R2_metric.h
 * Author: boyko_mihail
 *
 * Created on 24 сентября 2019 г., 12:28
 */
#include "Metric.h"
#include "Statistic.h"

#ifndef R2_METRIC_H
#define	R2_METRIC_H

class R2_metric : public Metric {
    public:
        static double calculateMetric(const std::vector<int> Y_pred, const std::vector<int> Y_test){

            double sum_Up = 0;  
            double sum_down = 0;
            double Y_mean = 0;
            double Y_sig = 0;

            Statistic::findeStatistic(Y_test, Y_mean, Y_sig );

            for(int i = 0; i<Y_pred.size(); i++){
                sum_Up += ( Y_test[i] - Y_pred[i])*( Y_test[i] - Y_pred[i]);
                sum_down += ( Y_test[i] - Y_mean)*( Y_test[i] - Y_mean);
            }
            return 1 - (sum_Up/sum_down);
        }
};

#endif	/* R2_METRIC_H */

