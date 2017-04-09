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

    int forward(T*   x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
                T*   y,  int y_d1,  int y_d2,  int y_d3,  int y_d4,
                int* ws, int ws_d1, int ws_d2, int ws_d3, int ws_d4);

    int backward(T*   gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
                 T*   x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
                 T*   gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
                 int* ws, int ws_d1, int ws_d2, int ws_d3, int ws_d4);

protected:
    int forward_setup(int x_d1, int x_d2, int x_d3, int x_d4,
                      int s_y, int s_x,
                      int p_u, int p_d, int p_l, int p_r,
                      int ker_h, int ker_w,
                      mkldnn::algorithm alg_kind);  // alg_kind = pooling_max
                                                    // or         pooling_avg
    int backward_setup(int x_d1, int x_d2, int x_d3, int x_d4,
                       int s_y, int s_x,
                       int p_u, int p_d, int p_l, int p_r,
                       int ker_h, int ker_w,
                       mkldnn::algorithm alg_kind);  // alg_kind = pooling_max
                                                    // or         pooling_avg
    static Pooling<T>* get_forward_object(
                      T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                      int s_y, int s_x,
                      int p_u, int p_d, int p_l, int p_r,
                      int ker_h, int ker_w,
                      mkldnn::algorithm alg_kind) {
        Pooling<T>* pooling_forward = NULL;
        assert (alg_kind == pooling_max || alg_kind == pooling_avg);
        if (alg_kind == mkldnn::pooling_max) {
            pooling_forward = dynamic_cast<Pooling<T>*>(
                                LayerFactory<T>::get_instance().get_max_pool_layer
                                (x_d1, x_d2, x_d3, x_d4,
                                 s_y, s_x, ker_h, ker_w, p_u, p_d, p_l, p_r));
        } else {
            pooling_forward = dynamic_cast<Pooling<float>*>(
                                LayerFactory<T>::get_instance().get_avg_pool_layer
                                (x_d1, x_d2, x_d3, x_d4,
                                 s_y, s_x, ker_h, ker_w, p_u, p_d, p_l, p_r));
        }
        if (pooling_forward == NULL) {
            pooling_forward = new Pooling<T>();
            pooling_forward->forward_setup(x_d1, x_d2, x_d3, x_d4,
                                       s_y, s_x, p_u, p_d, p_l, p_r,
                                       ker_h, ker_w,
                                       alg_kind);
            if (alg_kind == mkldnn::pooling_max) {
                LayerFactory<T>::get_instance().set_max_pool_layer(
                                    x_d1, x_d2, x_d3, x_d4,
                                    s_y, s_x, ker_h, ker_w, p_u, p_d, p_l, p_r,
                                    pooling_forward);
            } else {
                LayerFactory<T>::get_instance().set_avg_pool_layer(
                                    x_d1, x_d2, x_d3, x_d4,
                                    s_y, s_x, ker_h, ker_w, p_u, p_d, p_l, p_r,
                                    pooling_forward);
            }
        }
        return pooling_forward;
    }

    static Pooling<T>* get_backward_object(
                      T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                      int s_y, int s_x,
                      int p_u, int p_d, int p_l, int p_r,
                      int ker_h, int ker_w,
                      mkldnn::algorithm alg_kind) {
        Pooling<T>* pooling_backward = NULL;
        if (alg_kind == mkldnn::pooling_max) {
            pooling_backward = dynamic_cast<Pooling<T>*>(
                                LayerFactory<T>::get_instance().get_max_pool_layer
                                (x_d1, x_d2, x_d3, x_d4,
                                 s_y, s_x, ker_h, ker_w, p_u, p_d, p_l, p_r));
        } else {
            pooling_backward = dynamic_cast<Pooling<T>*>(
                                LayerFactory<T>::get_instance().get_avg_pool_layer
                                (x_d1, x_d2, x_d3, x_d4,
                                 s_y, s_x, ker_h, ker_w, p_u, p_d, p_l, p_r));
        }
        assert (pooling_backward != NULL);  // we must have already done forward
                                            // before
        if (pooling_backward->backward_first_setup_ == true) {
            pooling_backward->backward_setup(x_d1, x_d2, x_d3, x_d4,
                                       s_y, s_x, p_u, p_d, p_l, p_r,
                                       ker_h, ker_w,
                                       alg_kind);
            pooling_backward->backward_first_setup_ = false;
        }
        return pooling_backward;
    }

public:
    static void do_forward(
                T*   x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
                T*   y,  int y_d1,  int y_d2,  int y_d3,  int y_d4,
                int* ws, int ws_d1, int ws_d2, int ws_d3, int ws_d4,
                int  s_y, int s_x,
                int  p_u, int p_d, int p_l, int p_r,
                int  ker_h, int ker_w,
                mkldnn::algorithm alg_kind) {
        Pooling<T> *forward_object = get_forward_object(
                                        x, x_d1, x_d2, x_d3, x_d4,
                                        s_y, s_x, p_u, p_d, p_l, p_r,
                                        ker_h, ker_w,
                                        alg_kind);
        forward_object->forward(x,  x_d1,  x_d2,  x_d3,  x_d4,
                                y,  y_d1,  y_d2,  y_d3,  y_d4,
                                ws, ws_d1, ws_d2, ws_d3, ws_d4);
    }

    static void do_forward(
                T* x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
                T* y,  int y_d1,  int y_d2,  int y_d3,  int y_d4,
                int s_y, int s_x,
                int p_u, int p_d, int p_l, int p_r,
                int ker_h, int ker_w,
                mkldnn::algorithm alg_kind) {
        do_forward(x, x_d1, x_d2, x_d3, x_d4,
                   y, y_d1, y_d2, y_d3, y_d4,
                   NULL, 0, 0, 0, 0,
                   s_y, s_x, p_u, p_d, p_l, p_r, ker_h, ker_w,
                   alg_kind);
    }

    static void do_backward(
                T*   gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
                T*   x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
                T*   gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
                int* ws, int ws_d1, int ws_d2, int ws_d3, int ws_d4,
                int  s_y, int s_x,
                int  p_u, int p_d, int p_l, int p_r,
                int  ker_h, int ker_w,
                mkldnn::algorithm alg_kind) {
        Pooling<T> *backward_object = get_backward_object(
                                        x, x_d1, x_d2, x_d3, x_d4,
                                        s_y, s_x, p_u, p_d, p_l, p_r,
                                        ker_h, ker_w,
                                        alg_kind);
        backward_object->backward(gy, gy_d1, gy_d2, gy_d3, gy_d4,
                                  x,  x_d1,  x_d2,  x_d3,  x_d4,
                                  gx, gx_d1, gx_d2, gx_d3, gx_d4,
                                  ws, ws_d1, ws_d2, ws_d3, ws_d4);
    }

    static void do_backward(
                T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
                T* x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
                T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
                int s_y, int s_x,
                int p_u, int p_d, int p_l, int p_r,
                int ker_h, int ker_w,
                mkldnn::algorithm alg_kind) {
        do_backward(gy, gy_d1, gy_d2, gy_d3, gy_d4,
                    x,  x_d1,  x_d2,  x_d3,  x_d4,
                    gx, gx_d1, gx_d2, gx_d3, gx_d4,
                    NULL, 0, 0, 0, 0,
                    s_y, s_x, p_u, p_d, p_l, p_r, ker_h, ker_w,
                    alg_kind);
    }
private:
    int x_d1_, x_d2_, x_d3_, x_d4_;
    int y_d1_, y_d2_, y_d3_, y_d4_;
    int s_y_, s_x_, p_u_, p_d_, p_l_, p_r_, ker_h_, ker_w_;
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
    std::shared_ptr<mkldnn::pooling_forward::primitive_desc>  fwd_pd_;
    std::shared_ptr<mkldnn::pooling_forward>                  fwd_;
    std::shared_ptr<mkldnn::pooling_backward::desc>           bwd_desc_;
    std::shared_ptr<mkldnn::pooling_backward::primitive_desc> bwd_pd_;
    std::shared_ptr<mkldnn::pooling_backward>                 bwd_;

    // reordered related
    mkldnn::primitive                         reorder_x_;
    mkldnn::primitive                         reorder_y_;
    mkldnn::primitive                         reorder_gx_;
    mkldnn::primitive                         reorder_gy_;
};

#endif // _POOLING_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s