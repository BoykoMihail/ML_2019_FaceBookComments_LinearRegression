/* 
 * File:   Statistic.h
 * Author: boyko_mihail
 *
 * Created on 24 сентября 2019 г., 12:42
 */
#include <vector>

#ifndef STATISTIC_H
#define	STATISTIC_H

class Statistic {
public:
    template <typename T>
    static void findeStatistic(const std::vector<T> &v,  double &mean, double &sig ){

        float summOfElements = 0;
        float summOfSquareElements = 0;
        for (int i = 0; i < v.size(); ++i){
            summOfElements += v[i];
            summOfSquareElements += v[i]*v[i];
        }
        mean = summOfElements/v.size();
        sig = sqrt(summOfSquareElements/v.size() - mean*mean);
    }

};

#endif	/* STATISTIC_H */

