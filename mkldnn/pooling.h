#pragma once
#ifndef _POOLING_H_
#define _POOLING_H_

#include <mkldnn.hpp>
#include <vector>
#include "layer.h"
#include "stream_factory.h"

template <typename T>
class Pooling: public Layer<T>{
public:
    //Pooling();

    int forward(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                T* y, int y_d1, int y_d2, int y_d3, int y_d4);

    //int backward();

    int forward_setup(int x_d1, int x_d2, int x_d3, int x_d4,
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
        if (alg_kind == mkldnn::pooling_max) {
            pooling_forward = dynamic_cast<Pooling<T>*>(
                                StreamFactory::getInstance().getMaxPoolFwdStream
                                (x_d1, x_d2, x_d3, x_d4,
                                 s_y, s_x, ker_h, ker_w, p_h, p_w, p_h, p_w));
        } else {
            // TODO
            //pooling_forward = dynamic_cast<Pooling<float>*>(
                                //StreamFactory::getInstance().getAvgPoolFwdStream
                                //(x_d1, x_d2, x_d3, x_d4,
                                 //s_y, s_x, p_h, p_w, p_h, p_w, ker_h, ker_w));
        }
        if (pooling_forward == NULL) {
            pooling_forward = new Pooling<T>();
            pooling_forward->forward_setup(x_d1, x_d2, x_d3, x_d4,
                                       s_y, s_x, p_h, p_w, ker_h, ker_w,
                                       alg_kind);
            if (alg_kind == mkldnn::pooling_max) {
                StreamFactory::getInstance().setMaxPoolFwdStream(
                                    x_d1, x_d2, x_d3, x_d4,
                                    s_y, s_x, ker_h, ker_w, p_h, p_w, p_h, p_w,
                                    pooling_forward);
            } else {
                // TODO
                // check avg pool
            }
        }
        return pooling_forward;
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
    //mkldnn::stream* stream_;
    //std::vector<mkldnn::primitive> primitives_;
    int x_d1_, x_d2_, x_d3_, x_d4_;
    int y_d1_, y_d2_, y_d3_, y_d4_;
    int s_y_, s_x_, p_h_, p_w_, ker_h_, ker_w_;
    T *x_internal_, *y_internal_;
    mkldnn::algorithm alg_kind_;

    std::shared_ptr<mkldnn::memory>                            user_src_memory_;
    std::shared_ptr<mkldnn::memory>                            user_dst_memory_;
    std::shared_ptr<mkldnn::memory>                            indice_memory_;
    std::shared_ptr<mkldnn::memory>                            src_memory_;
    std::shared_ptr<mkldnn::memory>                            dst_memory_;
    std::shared_ptr<mkldnn::memory::desc>                      src_md_;
    std::shared_ptr<mkldnn::memory::desc>                      dst_md_;
    std::shared_ptr<mkldnn::pooling_forward::desc>             fwd_desc_;
    std::shared_ptr<mkldnn::pooling_forward::primitive_desc>   fwd_prim_desc_;
    std::shared_ptr<mkldnn::pooling_forward>                   fwd_;

    // reordered related
    mkldnn::primitive                         reorder_src_;
    mkldnn::primitive                         reorder_dst_;
    bool                                      reorder_src_p_;
    bool                                      reorder_dst_p_;
};

#endif // _POOLING_H_
