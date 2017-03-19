#pragma once
#ifndef _POOLING_H_
#define _POOLING_H_

#include <glog/logging.h>
#include <iostream>
#include <mkldnn.hpp>
#include <vector>
#include "layer.h"
#include "layer_factory.h"

template <typename T>
class Pooling: public Layer<T>{
public:
    //Pooling();

    int forward(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                T* y, int y_d1, int y_d2, int y_d3, int y_d4);

    int backward(T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
                 T* x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
                 T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4);

    int forward_setup(int x_d1, int x_d2, int x_d3, int x_d4,
                      int s_y, int s_x,
                      int p_h, int p_w,
                      int ker_h, int ker_w,
                      mkldnn::algorithm alg_kind);  // alg_kind = pooling_max
                                                    // or         pooling_avg
    int backward_setup(int x_d1, int x_d2, int x_d3, int x_d4,
                       int s_y, int s_x,
                       int p_h, int p_w,
                       int ker_h, int ker_w,
                       mkldnn::algorithm alg_kind);  // alg_kind = pooling_max
                                                    // or         pooling_avg
    static Pooling<T>* get_forward_object(
                      T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                      int s_y, int s_x,
                      int p_h, int p_w,
                      int ker_h, int ker_w,
                      mkldnn::algorithm alg_kind) {
        Pooling<T>* pooling_forward = NULL;
        assert (alg_kind == pooling_max || alg_kind == pooling_avg);
        if (alg_kind == mkldnn::pooling_max) {
            pooling_forward = dynamic_cast<Pooling<T>*>(
                                LayerFactory::getInstance().getMaxPoolLayer
                                (x_d1, x_d2, x_d3, x_d4,
                                 s_y, s_x, ker_h, ker_w, p_h, p_w, p_h, p_w));
        } else {
            pooling_forward = dynamic_cast<Pooling<float>*>(
                                LayerFactory::getInstance().getAvgPoolLayer
                                (x_d1, x_d2, x_d3, x_d4,
                                 s_y, s_x, ker_h, ker_w, p_h, p_w, p_h, p_w));
        }
        if (pooling_forward == NULL) {
            pooling_forward = new Pooling<T>();
            pooling_forward->forward_setup(x_d1, x_d2, x_d3, x_d4,
                                       s_y, s_x, p_h, p_w, ker_h, ker_w,
                                       alg_kind);
            if (alg_kind == mkldnn::pooling_max) {
                LayerFactory::getInstance().setMaxPoolLayer(
                                    x_d1, x_d2, x_d3, x_d4,
                                    s_y, s_x, ker_h, ker_w, p_h, p_w, p_h, p_w,
                                    pooling_forward);
            } else {
                LayerFactory::getInstance().setAvgPoolLayer(
                                    x_d1, x_d2, x_d3, x_d4,
                                    s_y, s_x, ker_h, ker_w, p_h, p_w, p_h, p_w,
                                    pooling_forward);
            }
        }
        return pooling_forward;
    }

    static Pooling<T>* get_backward_object(
                      T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                      int s_y, int s_x,
                      int p_h, int p_w,
                      int ker_h, int ker_w,
                      mkldnn::algorithm alg_kind) {
        Pooling<T>* pooling_backward = NULL;
        if (alg_kind == mkldnn::pooling_max) {
            pooling_backward = dynamic_cast<Pooling<T>*>(
                                LayerFactory::getInstance().getMaxPoolLayer
                                (x_d1, x_d2, x_d3, x_d4,
                                 s_y, s_x, ker_h, ker_w, p_h, p_w, p_h, p_w));
        } else {
            pooling_backward = dynamic_cast<Pooling<T>*>(
                                LayerFactory::getInstance().getAvgPoolLayer
                                (x_d1, x_d2, x_d3, x_d4,
                                 s_y, s_x, ker_h, ker_w, p_h, p_w, p_h, p_w));
        }
        assert (pooling_backward != NULL);  // we must have already done forward
                                            // before
        if (pooling_backward->backward_first_setup_ == true) {
            pooling_backward->backward_setup(x_d1, x_d2, x_d3, x_d4,
                                       s_y, s_x, p_h, p_w, ker_h, ker_w,
                                       alg_kind);
            pooling_backward->backward_first_setup_ = false;
        }
        return pooling_backward;
    }
private:
    //mkldnn::stream* stream_;
    //std::vector<mkldnn::primitive> primitives_;
    int x_d1_, x_d2_, x_d3_, x_d4_;
    int y_d1_, y_d2_, y_d3_, y_d4_;
    int s_y_, s_x_, p_h_, p_w_, ker_h_, ker_w_;
    mkldnn::algorithm alg_kind_;

    std::shared_ptr<mkldnn::memory>                           user_x_mem_;
    std::shared_ptr<mkldnn::memory>                           user_y_mem_;
    std::shared_ptr<mkldnn::memory>                           user_gx_mem_;
    std::shared_ptr<mkldnn::memory>                           user_gy_mem_;
    std::shared_ptr<mkldnn::memory>                           workspace_memory_;
    std::shared_ptr<mkldnn::memory>                           x_mem_;
    std::shared_ptr<mkldnn::memory>                           y_mem_;
    std::shared_ptr<mkldnn::memory>                           gx_mem_;
    std::shared_ptr<mkldnn::memory>                           gy_mem_;
    std::shared_ptr<mkldnn::memory::desc>                     x_md_;
    std::shared_ptr<mkldnn::memory::desc>                     y_md_;
    std::shared_ptr<mkldnn::memory::desc>                     gx_md_;
    std::shared_ptr<mkldnn::memory::desc>                     gy_md_;
    std::shared_ptr<mkldnn::pooling_forward::desc>            fwd_desc_;
    std::shared_ptr<mkldnn::pooling_forward::primitive_desc>  fwd_prim_desc_;
    std::shared_ptr<mkldnn::pooling_forward>                  fwd_;
    std::shared_ptr<mkldnn::pooling_backward::desc>           bwd_desc_;
    std::shared_ptr<mkldnn::pooling_backward::primitive_desc> bwd_prim_desc_;
    std::shared_ptr<mkldnn::pooling_backward>                 bwd_;

    // reordered related
    mkldnn::primitive                         reorder_x_;
    mkldnn::primitive                         reorder_y_;
    mkldnn::primitive                         reorder_gx_;
    mkldnn::primitive                         reorder_gy_;
};

#endif // _POOLING_H_
