#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "relu4d.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
Relu4D<T>::Relu4D()
: relu_fwd_user_src_mem_(NULL), relu_fwd_dst_mem_(NULL)
               , relu_fwd_src_md_(NULL), relu_fwd_desc_(NULL), relu_fwd_pd_(NULL)
               , relu_fwd_(NULL), fwd_stream_(NULL)
               , relu_diff_dst_mem_(NULL), relu_diff_dst_md_(NULL)
               , relu_bwd_desc_(NULL), relu_bwd_pd_(NULL)
               , relu_bwd_(NULL), bwd_stream_(NULL)
{

}

template<typename T>
int Relu4D<T>::forward_setup(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                             T* y, int y_d1, int y_d2, int y_d3, int y_d4)
{
    memory::dims relu_src_tz = {x_d1, x_d2, x_d3, x_d4};
    memory::dims relu_dst_tz = {y_d1, y_d2, y_d3, y_d4};

    /* create memory for user data */
    relu_fwd_user_src_mem_.reset(new memory({{{relu_src_tz}, memory::data_type::f32,
        memory::format::nchw}, cpu_engine}, x));
    relu_fwd_dst_mem_.reset(new memory({{{relu_src_tz}, memory::data_type::f32,
        memory::format::nchw}, cpu_engine}, y));

    /* create memory descriptors for relu data w/ no specified format */
    relu_fwd_src_md_.reset(new memory::desc({relu_src_tz}, memory::data_type::f32,
        memory::format::nchw));

    /* no reorder for relu, since there is no interface src_primitive_desc() of relu pd*/
    auto relu_src_mem = relu_fwd_user_src_mem_;

    const double negative_slope = 0.0;//1.0;

    /* create relu primitive and add it to net_ */
    relu_fwd_desc_.reset(new relu_forward::desc(prop_kind::forward,
            *relu_fwd_src_md_, negative_slope));
    relu_fwd_pd_.reset(new relu_forward::primitive_desc(*relu_fwd_desc_, cpu_engine));

    relu_fwd_.reset(new relu_forward(*relu_fwd_pd_, *relu_src_mem,
            *relu_fwd_dst_mem_));

    fwd_primitives_.push_back(*relu_fwd_);
    fwd_stream_.reset(new stream(stream::kind::eager));

    return 0;
}

template<typename T>
void Relu4D<T>::fwd_reset_mem(T* x,
                              T* y)
{
    relu_fwd_user_src_mem_->set_data_handle(x);
    relu_fwd_dst_mem_->set_data_handle(y);
}

template<typename T>
int Relu4D<T>::forward(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                       T* y, int y_d1, int y_d2, int y_d3, int y_d4)
{
    //LOG(INFO) << "forward: " << x << " : " << x_size << " : " << y << " : " << y_size;
    //LOG(INFO) << "Convolution forward";
    if (!fwd_stream_) {
        forward_setup(x, x_d1, x_d2, x_d3, x_d4,
                      y, y_d1, y_d2, y_d3, y_d4);
        fwd_reset_mem(x, y);
        fwd_stream_->submit(fwd_primitives_).wait();
    } else {
        fwd_reset_mem(x, y);
        fwd_stream_->rerun().wait();
    }
    return 0;
}

template<typename T>
int Relu4D<T>::backward_setup(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                              T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
                              T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4)
{
    const double negative_slope = 0.0;//1.0;

    /* Backward relu */
    memory::dims relu_src_tz = {x_d1, x_d2, x_d3, x_d4};
    memory::dims relu_diff_dst_tz = {gy_d1, gy_d2, gy_d3, gy_d4};
    memory::dims relu_diff_src_tz = {gx_d1, gx_d2, gx_d3, gx_d4};

    relu_diff_src_mem_.reset(new memory({{{relu_diff_src_tz}, memory::data_type::f32,
        memory::format::nchw}, cpu_engine}, gx));
    relu_diff_dst_mem_.reset(new memory({{{relu_diff_dst_tz}, memory::data_type::f32,
        memory::format::nchw}, cpu_engine}, gy));

    /* create memory descriptors for relu data with user memory format */
    relu_diff_dst_md_.reset(new memory::desc({relu_diff_dst_tz}, memory::data_type::f32,
        memory::format::nchw));

    /* create backward relu primitive_descriptor */
    relu_bwd_desc_.reset(new relu_backward::desc(*relu_diff_dst_md_, *relu_fwd_src_md_, negative_slope));
    relu_bwd_pd_.reset(new relu_backward::primitive_desc(*relu_bwd_desc_, cpu_engine, *relu_fwd_pd_));

    /* finally create a backward relu primitive */
    relu_bwd_.reset(new relu_backward(*relu_bwd_pd_, *relu_fwd_user_src_mem_,
                                  *relu_diff_dst_mem_, *relu_diff_src_mem_));
    bwd_primitives_.push_back(*relu_bwd_);
    bwd_stream_.reset(new stream(stream::kind::eager));

    //LOG(INFO) << "Convolution backward";
    return 0;
}

template<typename T>
void Relu4D<T>::bwd_reset_mem(T* x,
                              T* gy,
                              T* gx)
{
    relu_fwd_user_src_mem_->set_data_handle(x);
    relu_diff_dst_mem_->set_data_handle(gy);
    relu_diff_src_mem_->set_data_handle(gx);
}

template<typename T>
int Relu4D<T>::backward(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
                        T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4)
{
    //LOG(INFO) << "backward: " << x << " : " << x_size << " : " << gy << " : " << gy_size << " : " << gx << " : " << gx_size;
    if (!bwd_stream_) {
        backward_setup(x, x_d1, x_d2, x_d3, x_d4,
                       gy, gy_d1, gy_d2, gy_d3, gy_d4,
                       gx, gx_d1, gx_d2, gx_d3, gx_d4);
        bwd_reset_mem(x, gy, gx);
        bwd_stream_->submit(bwd_primitives_).wait();
    } else {
        bwd_reset_mem(x, gy, gx);
        bwd_stream_->rerun().wait();
    }
    return 0;
}

template class Relu4D<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s