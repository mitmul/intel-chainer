#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "pooling.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
int Pooling<T>::forward_setup(int x_d1, int x_d2, int x_d3, int x_d4,
                              int s_y, int s_x,
                              int p_h, int p_w,
                              int ker_h, int ker_w,
                              mkldnn::algorithm alg_kind)
{
    int y_d1, y_d2, y_d3, y_d4;
    // prepare y according to x, s, p, ker
    y_d1 = x_d1;
    y_d2 = x_d2;
    y_d3 = x_d3-ker_h+p_h*2+1;
    y_d4 = x_d4-ker_w+p_w*2+1;

    LOG(INFO) << "Pooling forward_setup";

    LOG(INFO) << "x =(" << x_d1 << "," << x_d2 << "," << x_d3 << "," << x_d4 << ")";
    LOG(INFO) << "y =(" << y_d1 << "," << y_d2 << "," << y_d3 << "," << y_d4 << ")";
    LOG(INFO) << "strides =(" << s_y << "," << s_x << ")";
    LOG(INFO) << "padding =(" << p_h << "," << p_w << ")";
    LOG(INFO) << "kernel =(" << ker_h << "," << ker_w << ")";
    LOG(INFO) << "alg_kind =" << (alg_kind == pooling_max ? "max" :
                                 (alg_kind == pooling_avg ? "avg" :
                                              /* else */    "unknown"));

    if (alg_kind != pooling_max && alg_kind != pooling_avg) {
        LOG(ERROR) << "alg_kind must be either pooling_max or "
                   << "pooling_avg";
    }

    memory::dims src_tz     = {x_d1, x_d2, x_d3, x_d4};
    memory::dims dst_tz     = {y_d1, y_d2, y_d3, y_d4};
    memory::dims strides    = {s_y, s_x};
    memory::dims padding   = {p_h, p_w};
    memory::dims kernel     = {ker_h, ker_w};

    /* create memory for user data */
    auto user_src_memory = new memory({{{src_tz}, memory_data_type<T>(),
                                      memory::format::nchw}, cpu_engine});
    auto src_md = new memory::desc({src_tz}, memory_data_type<T>(),
                                   memory::format::any);

    auto user_dst_memory = new memory({{{dst_tz}, memory_data_type<T>(),
                                      memory::format::nchw}, cpu_engine});
    auto dst_md = new memory::desc({dst_tz}, memory_data_type<T>(),
                                   memory::format::any);

    // create a pooling descriptor
    pooling_forward::desc* fwd_desc = NULL;
    fwd_desc = new pooling_forward::desc(prop_kind::forward, alg_kind,
                                         *src_md, *dst_md,
                                         strides, kernel, padding, padding,
                                         padding_kind::zero);

    auto fwd_prim_desc = new pooling_forward::primitive_desc(
                                *fwd_desc, cpu_engine);


    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    #if 0 // reorder does not seem needed
    auto src_memory = user_src_memory;
    auto dst_memory = user_dst_memory;
    bool reorder_pool_src = false;
    bool reorder_pool_dst = false;

    if (memory::primitive_desc(fwd_prim_desc.dst_primitive_desc())
        != user_dst_memory.get_primitive_desc()) {
        dst_memory = memory(fwd_prim_desc.dst_primitive_desc());
        pool_reorder_dst = reorder(pool_dst_memory, pool_user_dst_memory);
        reorder_pool_dst = true;
    }
    #endif

    auto pool_workspace_memory = new memory(
                                    fwd_prim_desc->workspace_primitive_desc());

    auto fwd = new pooling_forward(
            *fwd_prim_desc, *user_src_memory, *pool_workspace_memory, *user_dst_memory);
    this->primitives_.push_back(*fwd);
    this->stream_ = new stream(stream::kind::eager);

    x_d1_     = x_d1;
    x_d2_     = x_d2;
    x_d3_     = x_d3;
    x_d4_     = x_d4;

    s_y_      = s_y;
    s_x_      = s_x;
    p_h_      = p_h;
    p_w_      = p_w;
    ker_h_    = ker_h;
    ker_w_    = ker_w;

    alg_kind_ = alg_kind;

    return 0;
}

template<typename T>
int Pooling<T>::forward(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                        T* y, int y_d1, int y_d2, int y_d3, int y_d4)
{
    if (this->first_use) {
        this->stream_->submit(this->primitives_).wait();
        this->first_use = false;
    } else {
        this->stream_->rerun().wait();
    }
    return 0;
}

#if 0
template<typename T>
Convolution2D<T>::Convolution2D(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                                T* W, int W_d1, int W_d2, int W_d3, int W_d4,
                                T* y, int y_d1, int y_d2, int y_d3, int y_d4,
                                int s1, int s2,
                                int p1, int p2)
{
    LOG(INFO) << "Convolution CTOR, without b";

    Convolution2D(x, x_d1, x_d2, x_d3, x_d4,
                  W, W_d1, W_d2, W_d3, W_d4,
                  NULL, 0, // no bias
                  y, y_d1, y_d2, y_d3, y_d4,
                  s1, s2,
                  p1, p2);
}

template<typename T>
int Convolution2D<T>::forward()
{
    LOG(INFO) << "Convolution forward";
    stream_->submit(primitives_);
    return 0;
}

template<typename T>
int Convolution2D<T>::backward()
{
    LOG(INFO) << "Convolution backward";
    return 0;
}

#endif

template class Pooling<float>;
