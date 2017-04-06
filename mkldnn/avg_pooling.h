#pragma once
#ifndef _AVG_POOLING_H_
#define _AVG_POOLING_H_

#include <mkldnn.hpp>
#include <vector>
#include "pooling.h"

template <typename T>
class AvgPooling:public Pooling<T> {
private:
    static Pooling<T>* get_forward_object(
                      T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                      int s_y, int s_x,
                      int p_u, int p_d, int p_l, int p_r,
                      int ker_h, int ker_w) {
        return Pooling<T>::get_forward_object(x, x_d1, x_d2, x_d3, x_d4,
                                              s_y, s_x, p_u, p_d, p_l, p_r,
                                              ker_h, ker_w,
                                              mkldnn::pooling_avg);
    };

    static Pooling<T>* get_backward_object(
                      T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                      int s_y, int s_x,
                      int p_u, int p_d, int p_l, int p_r,
                      int ker_h, int ker_w) {
        return Pooling<T>::get_backward_object(x, x_d1, x_d2, x_d3, x_d4,
                                              s_y, s_x, p_u, p_d, p_l, p_r,
                                              ker_h, ker_w,
                                              mkldnn::pooling_avg);
    };

public:
    static void do_forward(
                T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                T* y, int y_d1, int y_d2, int y_d3, int y_d4,
                int s_y, int s_x,
                int p_u, int p_d, int p_l, int p_r,
                int ker_h, int ker_w) {
        Pooling<T>::do_forward(x, x_d1, x_d2, x_d3, x_d4,
                               y, y_d1, y_d2, y_d3, y_d4,
                               s_y, s_x, p_u, p_d, p_l, p_r, ker_h, ker_w,
                               mkldnn::pooling_avg);
    }

    static void do_backward(
                T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
                T* x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
                T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
                int s_y, int s_x,
                int p_u, int p_d, int p_l, int p_r,
                int ker_h, int ker_w) {
        Pooling<T>::do_backward(gy, gy_d1, gy_d2, gy_d3, gy_d4,
                                x,  x_d1,  x_d2,  x_d3,  x_d4,
                                gx, gx_d1, gx_d2, gx_d3, gx_d4,
                                s_y, s_x, p_u, p_d, p_l, p_r, ker_h, ker_w,
                                mkldnn::pooling_avg);
    }
};

#endif // _AVG_POOLING_H_
