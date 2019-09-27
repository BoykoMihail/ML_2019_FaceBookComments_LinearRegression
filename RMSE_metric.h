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

    static double calculateMetric(const std::vector<double> Y_pred, const std::vector<double> Y_test) {

        double sum = 0;

        VectorXd Y_test_vector = VectorXd::Map(Y_test.data(), Y_test.size());
        VectorXd Y_pred_vector = VectorXd::Map(Y_pred.data(), Y_pred.size());

        sum += (Y_test_vector - Y_pred_vector).dot(Y_test_vector - Y_pred_vector);

        return sqrt(sum / Y_pred.size());
    }
};

#endif	/* RMSE_METRIC_H */

