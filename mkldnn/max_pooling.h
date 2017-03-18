#pragma once
#ifndef _MAX_POOLING_H_
#define _MAX_POOLING_H_

#include <mkldnn.hpp>
#include <vector>
#include "pooling.h"

template <typename T>
class MaxPooling:public Pooling<T> {
public:
    static Pooling<T>* get_forward_object(
                      T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                      int s_y, int s_x,
                      int p_h, int p_w,
                      int ker_h, int ker_w) {
        return Pooling<T>::get_forward_object(x, x_d1, x_d2, x_d3, x_d4,
                                              s_y, s_x, p_h, p_w, ker_h, ker_w,
                                              mkldnn::pooling_max);
    };

    static Pooling<T>* get_backward_object(
                      T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                      int s_y, int s_x,
                      int p_h, int p_w,
                      int ker_h, int ker_w) {
        return Pooling<T>::get_backward_object(x, x_d1, x_d2, x_d3, x_d4,
                                              s_y, s_x, p_h, p_w, ker_h, ker_w,
                                              mkldnn::pooling_max);
    };
};

#endif // _MAX_POOLING_H_
