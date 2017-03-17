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
    y_d3 = (x_d3-ker_h+p_h*2)/s_y+1;
    y_d4 = (x_d4-ker_w+p_w*2)/s_x+1;

    LOG(INFO) << "Pooling forward_setup";

    LOG(INFO) << "    xdim=(" << x_d1 << "," << x_d2 << "," << x_d3 << "," << x_d4 << ")";
    LOG(INFO) << "    ydim=(" << y_d1 << "," << y_d2 << "," << y_d3 << "," << y_d4 << ")";
    LOG(INFO) << "    strides =(" << s_y << "," << s_x << ")";
    LOG(INFO) << "    padding =(" << p_h << "," << p_w << ")";
    LOG(INFO) << "    kernel =(" << ker_h << "," << ker_w << ")";
    LOG(INFO) << "    alg_kind =" << (alg_kind == pooling_max ? "max" :
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
    user_src_memory_.reset(new memory({{{src_tz}, memory_data_type<T>(),
                                      memory::format::nchw}, cpu_engine}));
    x_internal_ = (T*)user_src_memory_->get_data_handle();
    src_md_.reset(new memory::desc({src_tz}, memory_data_type<T>(),
                                   memory::format::any));

    user_dst_memory_.reset(new memory({{{dst_tz}, memory_data_type<T>(),
                                      memory::format::nchw}, cpu_engine}));
    y_internal_ = (T*)user_dst_memory_->get_data_handle();
    dst_md_.reset(new memory::desc({dst_tz}, memory_data_type<T>(),
                                   memory::format::any));


    // create a pooling descriptor
    fwd_desc_.reset(new pooling_forward::desc(prop_kind::forward, alg_kind,
                                         *src_md_, *dst_md_,
                                         strides, kernel, padding, padding,
                                         padding_kind::zero));

    fwd_prim_desc_.reset(new pooling_forward::primitive_desc(
                                *fwd_desc_, cpu_engine));


    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    src_memory_ = user_src_memory_;
    dst_memory_ = user_dst_memory_;
    bool reorder_src_p_ = false;
    bool reorder_dst_p_ = false;

    #if 0
    if (memory::primitive_desc(fwd_prim_desc_.get()->src_primitive_desc())
        != user_src_memory_->get_primitive_desc()) {
        src_memory_.reset(new memory(fwd_prim_desc_.get()->src_primitive_desc()));
        reorder_src_ = reorder(*user_src_memory_, *src_memory_);
        reorder_src_p_ = true;
    }
    #endif

    if (memory::primitive_desc(fwd_prim_desc_->dst_primitive_desc())
        != user_dst_memory_->get_primitive_desc()) {
        dst_memory_.reset(new memory(fwd_prim_desc_.get()->dst_primitive_desc()));
        reorder_dst_ = reorder(*dst_memory_, *user_dst_memory_);
        reorder_dst_p_ = true;
    }

    indice_memory_.reset(new memory(dst_memory_->get_primitive_desc()));
    fwd_.reset(new pooling_forward(
            *fwd_prim_desc_, *src_memory_, *dst_memory_, *indice_memory_));

    LOG(INFO) << "    reorder_src: " << reorder_src_p_;
    LOG(INFO) << "    reorder_dst: " << reorder_src_p_;
    if (reorder_src_p_) this->primitives_.push_back(reorder_src_);
    this->primitives_.push_back(*fwd_);
    if (reorder_dst_p_) this->primitives_.push_back(reorder_dst_);
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
    this->first_use_ = true;

    return 0;
}

template<typename T>
int Pooling<T>::forward(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                        T* y, int y_d1, int y_d2, int y_d3, int y_d4)
{
    LOG(INFO) << "Pooling forward";
    LOG(INFO) << "    xdim=(" << x_d1 << "," << x_d2 <<","<< x_d3 <<","<< x_d4 <<")";
    LOG(INFO) << "    ydim=(" << y_d1 << "," << y_d2 <<","<< y_d3 <<","<< y_d4 <<")";
    LOG(INFO) << "    x={" << x[0] << "," << x[1] << "," << x[2] << "," << x[3] << "}";

    size_t size_x = x_d1*x_d2*x_d3*x_d4*sizeof(T);
    size_t size_y = y_d1*y_d2*y_d3*y_d4*sizeof(T);
    LOG(INFO) << "    copying " << size_x << " bytes from x";
    memcpy(this->x_internal_, x, size_x);
    if (this->first_use_) {
        this->stream_->submit(this->primitives_).wait();
        this->first_use_ = false;
    } else {
        this->stream_->rerun().wait();
    }
    LOG(INFO) << "    copying " << size_y << " bytes to y";
    memcpy(y, this->y_internal_, size_y);
    LOG(INFO) << "    y={" << y[0] << "," << y[1] << "," << y[2] << "," << y[3] << "}";
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
