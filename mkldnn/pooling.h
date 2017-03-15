#pragma once
#ifndef _POOLING_H_
#define _POOLING_H_

#include <mkldnn.hpp>
#include <vector>
#include "layer.h"

template <typename T>
class Pooling:public Layer<T> {
public:
    Pooling();

    int forward();
    int backward();

    int forward_setup(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                      T* y, int y_d1, int y_d2, int y_d3, int y_d4,
                      int s_y, int s_x,
                      int p_h, int p_w,
                      int ker_h, int ker_w,
                      mkldnn::algorithm alg_kind);  // alg_kind = pooling_max
                                                    // or         pooling_avg
    static Pooling<T>* get_forward_object(
                      T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                      T* y, int y_d1, int y_d2, int y_d3, int y_d4,
                      int s_y, int s_x,
                      int p_h, int p_w,
                      int ker_h, int ker_w,
                      mkldnn::algorithm alg_kind) {
        Pooling<T>* pooling_forward = new Pooling<T>();
        pooling_forward->forward_setup(x, x_d1, x_d2, x_d3, x_d4,
                                       y, y_d1, y_d2, y_d3, y_d4,
                                       s_y, s_x, p_h, p_w, ker_h, ker_w,
                                       alg_kind);
    }

#if 0 //TODO not defined
    int backward_setup(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                       T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
                       T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
                       int s_y, int s_x,
                       int p_h, int p_w,
                       int ker_h, int ker_w,
                       mkldnn::algorithm alg_kind);  // alg_kind = pooling_max
                                                     // or         pooling_avg
#endif

private:
    mkldnn::stream* stream_;
    std::vector<mkldnn::primitive> primitives_;
};

#endif // _POOLING_H_
