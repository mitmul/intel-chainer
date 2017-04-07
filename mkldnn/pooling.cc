#include <glog/logging.h>
#include <iostream>
#include "common.h"
#include "mkldnn.hpp"
#include "pooling.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
int Pooling<T>::forward_setup(int x_d1, int x_d2, int x_d3, int x_d4,
                              int s_y, int s_x,
                              int p_u, int p_d, int p_l, int p_r,
                              int ker_h, int ker_w,
                              mkldnn::algorithm alg_kind)
{
    memory::format format;
    // we check AVX512 first then AVX2
    if (cpu_support_avx512_p() && (x_d2%16)==0) {
        format = memory::format::nChw16c;
    } else if (cpu_support_avx2_p() && (x_d2%8)==0) {
        format = memory::format::nChw8c;
    } else {
        format = memory::format::nchw;
    }

    int y_d1, y_d2, y_d3, y_d4;
    // prepare y according to x, s, p, ker
    y_d1 = x_d1;
    y_d2 = x_d2;
    y_d3 = (x_d3-ker_h+p_u+p_d)/s_y+1;
    y_d4 = (x_d4-ker_w+p_l+p_r)/s_x+1;

    LOG(INFO) << "Pooling forward_setup";

    LOG(INFO) << "    xdim=(" << x_d1 << "," << x_d2<< ","
                              << x_d3 << "," << x_d4 << ")";
    LOG(INFO) << "    ydim=(" << y_d1 << "," << y_d2 << ","
                              << y_d3 << "," << y_d4 << ")";
    LOG(INFO) << "    strides =(" << s_y << "," << s_x << ")";
    LOG(INFO) << "    padding =(" << p_u << "," << p_d << ","
                                  << p_l << "," << p_r << ")";
    LOG(INFO) << "    kernel =(" << ker_h << "," << ker_w << ")";
    LOG(INFO) << "    alg_kind =" << (alg_kind == pooling_max ? "max" :
                                     (alg_kind == pooling_avg ? "avg" :
                                                  /* else */    "unknown"));

    if (alg_kind != pooling_max && alg_kind != pooling_avg) {
        LOG(ERROR) << "alg_kind must be either pooling_max or "
                   << "pooling_avg";
    }

    memory::dims x_tz      = {x_d1, x_d2, x_d3, x_d4};
    memory::dims y_tz      = {y_d1, y_d2, y_d3, y_d4};
    memory::dims strides   = {s_y, s_x};
    memory::dims padding_l = {p_u, p_l};
    memory::dims padding_r = {p_d, p_r};
    memory::dims kernel    = {ker_h, ker_w};

    /* create memory for user data */
    user_x_mem_.reset(new memory({{{x_tz}, memory_data_type<T>(),
                            memory::format::nchw}, cpu_engine}, dummy));
    // TODO here we let mkldnn allocate a piece of internal memory but its not
    // used and will soon be replaced.  An alt. way is pass data pointer of
    // first run to forward setup and use that data for first run but it makes
    // interface not clean
    x_md_.reset(new memory::desc({x_tz}, memory_data_type<T>(),
                              format));

    user_y_mem_.reset(new memory({{{y_tz}, memory_data_type<T>(),
                            memory::format::nchw}, cpu_engine}, dummy));
    y_md_.reset(new memory::desc({y_tz}, memory_data_type<T>(),
                            memory::format::any));


    // create a pooling descriptor
    fwd_desc_.reset(new pooling_forward::desc(prop_kind::forward_training, alg_kind,
                                         *x_md_, *y_md_,
                                         strides, kernel, padding_l, padding_r,
                                         padding_kind::zero));

    fwd_prim_desc_.reset(new pooling_forward::primitive_desc(
                                *fwd_desc_, cpu_engine));


    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    x_mem_ = user_x_mem_;
    y_mem_ = user_y_mem_;
    bool reorder_x_p = false;
    bool reorder_y_p = false;

    if (format != memory::format::nchw) {
        x_mem_.reset(new memory({{{x_tz}, memory_data_type<T>(),
                            format}, cpu_engine}));
        reorder_x_ = reorder(*user_x_mem_, *x_mem_);
        reorder_x_p = true;
    }

    if (memory::primitive_desc(fwd_prim_desc_->dst_primitive_desc())
        != user_y_mem_->get_primitive_desc()) {
        y_mem_.reset(new memory(fwd_prim_desc_.get()->dst_primitive_desc()));
        reorder_y_ = reorder(*y_mem_, *user_y_mem_);
        reorder_y_p = true;
    }

    workspace_memory_.reset(new memory(y_mem_->get_primitive_desc()));
    fwd_.reset(new pooling_forward(
            *fwd_prim_desc_, *x_mem_, *y_mem_, *workspace_memory_));

    LOG(INFO) << "    reorder_src: " << reorder_x_p;
    LOG(INFO) << "    reorder_dst: " << reorder_y_p;
    if (reorder_x_p) this->forward_primitives_.push_back(reorder_x_);
    this->forward_primitives_.push_back(*fwd_);
    if (reorder_y_p) this->forward_primitives_.push_back(reorder_y_);
    this->forward_stream_ = new stream(stream::kind::eager);

    x_d1_     = x_d1;
    x_d2_     = x_d2;
    x_d3_     = x_d3;
    x_d4_     = x_d4;

    s_y_      = s_y;
    s_x_      = s_x;
    p_u_      = p_u;
    p_d_      = p_d;
    p_l_      = p_l;
    p_r_      = p_r;
    ker_h_    = ker_h;
    ker_w_    = ker_w;

    alg_kind_ = alg_kind;
    this->forward_first_use_ = true;

    return 0;
}

template<typename T>
int Pooling<T>::backward_setup(int x_d1, int x_d2, int x_d3, int x_d4,
                              int s_y, int s_x,
                              int p_u, int p_d, int p_l, int p_r,
                              int ker_h, int ker_w,
                              mkldnn::algorithm alg_kind)
{
    memory::format format;
    // we check AVX512 first then AVX2
    if (cpu_support_avx512_p() && (x_d2%16)==0) {
        format = memory::format::nChw16c;
    } else if (cpu_support_avx2_p() && (x_d2%8)==0) {
        format = memory::format::nChw8c;
    } else {
        format = memory::format::nchw;
    }

    int y_d1, y_d2, y_d3, y_d4;
    // prepare y according to x, s, p, ker
    y_d1 = x_d1;
    y_d2 = x_d2;
    y_d3 = (x_d3-ker_h+p_u+p_d)/s_y+1;
    y_d4 = (x_d4-ker_w+p_l+p_r)/s_x+1;

    LOG(INFO) << "Pooling backward_setup";

    LOG(INFO) << "    xdim=(" << x_d1 << "," << x_d2 << ","
                              << x_d3 << "," << x_d4 << ")";
    LOG(INFO) << "    ydim=(" << y_d1 << "," << y_d2 << ","
                              << y_d3 << "," << y_d4 << ")";
    LOG(INFO) << "    strides =(" << s_y << "," << s_x << ")";
    LOG(INFO) << "    padding =(" << p_u << "," << p_d << ","
                                  << p_l << "," << p_r << ")";
    LOG(INFO) << "    kernel =(" << ker_h << "," << ker_w << ")";
    LOG(INFO) << "    alg_kind =" << (alg_kind == pooling_max ? "max" :
                                     (alg_kind == pooling_avg ? "avg" :
                                                  /* else */    "unknown"));

    if (alg_kind != pooling_max && alg_kind != pooling_avg) {
        LOG(ERROR) << "alg_kind must be either pooling_max or "
                   << "pooling_avg";
    }

    memory::dims x_tz      = {x_d1, x_d2, x_d3, x_d4};
    memory::dims y_tz      = {y_d1, y_d2, y_d3, y_d4};
    memory::dims strides   = {s_y, s_x};
    memory::dims padding_l = {p_u, p_l};
    memory::dims padding_r = {p_d, p_r};
    memory::dims kernel    = {ker_h, ker_w};

    /* create memory for user data */
    user_gx_mem_.reset(new memory({{{x_tz}, memory_data_type<T>(),
                                    memory::format::nchw}, cpu_engine}, dummy));
    gx_md_.reset(new memory::desc({x_tz}, memory_data_type<T>(),
                                    memory::format::any));

    user_gy_mem_.reset(new memory({{{y_tz}, memory_data_type<T>(),
                                    memory::format::nchw}, cpu_engine}, dummy));
    gy_md_.reset(new memory::desc({y_tz}, memory_data_type<T>(),
                                      format));

    // create a pooling descriptor
    bwd_desc_.reset(new pooling_backward::desc(
                                        alg_kind,
                                        *gx_md_, *gy_md_,
                                        strides, kernel, padding_l, padding_r,
                                        padding_kind::zero));

    bwd_prim_desc_.reset(new pooling_backward::primitive_desc(
                                *bwd_desc_, cpu_engine, *fwd_prim_desc_));
    // TODO: we don't know its safe until we know what mkldnn do with fwd_desc_


    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    gx_mem_ = user_gx_mem_;
    gy_mem_ = user_gy_mem_;
    bool reorder_x_p = false;
    bool reorder_y_p = false;

    if (format != memory::format::nchw) {
        gy_mem_.reset(new memory({{{y_tz}, memory_data_type<T>(),
                                    format}, cpu_engine}));
        reorder_gy_ = reorder(*user_gy_mem_, *gy_mem_);
        reorder_y_p = true;
    }

    if (memory::primitive_desc(bwd_prim_desc_.get()->diff_src_primitive_desc())
        != user_gx_mem_->get_primitive_desc()) {
        gx_mem_.reset(new memory(
                            bwd_prim_desc_.get()->diff_src_primitive_desc()));
        reorder_gx_ = reorder(*gx_mem_, *user_gx_mem_);
        reorder_x_p = true;
    }

    bwd_.reset(new pooling_backward(
            *bwd_prim_desc_, *gy_mem_, *workspace_memory_, *gx_mem_));

    LOG(INFO) << "    reorder_dst_diff: " << reorder_y_p;
    LOG(INFO) << "    reorder_src_diff: " << reorder_x_p;
    if (reorder_y_p) this->backward_primitives_.push_back(reorder_gy_);
    this->backward_primitives_.push_back(*bwd_);
    if (reorder_x_p) this->backward_primitives_.push_back(reorder_gx_);
    this->backward_stream_ = new stream(stream::kind::eager);

    x_d1_     = x_d1;
    x_d2_     = x_d2;
    x_d3_     = x_d3;
    x_d4_     = x_d4;

    s_y_      = s_y;
    s_x_      = s_x;
    p_u_      = p_u;
    p_d_      = p_d;
    p_l_      = p_l;
    p_r_      = p_r;
    ker_h_    = ker_h;
    ker_w_    = ker_w;

    alg_kind_ = alg_kind;
    this->backward_first_use_ = true;

    return 0;
}

template<typename T>
int Pooling<T>::forward(T*   x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
                        T*   y,  int y_d1,  int y_d2,  int y_d3,  int y_d4,
                        int* ws, int ws_d1, int ws_d2, int ws_d3, int ws_d4)
{
    LOG(INFO) << "Pooling forward";
    LOG(INFO) << "    xdim=(" << x_d1 << "," << x_d2 << ","
                              << x_d3 << "," << x_d4 << ")";
    LOG(INFO) << "    ydim=(" << y_d1 << "," << y_d2 << ","
                              << y_d3 << "," << y_d4 << ")";
    LOG(INFO) << "    x={"    << x[0] << "," << x[1] << ","
                              << x[2] << "," << x[3] << "}";

    user_x_mem_->set_data_handle(x);
    user_y_mem_->set_data_handle(y);
    if (ws != NULL)
        workspace_memory_->set_data_handle(ws);
    if (this->forward_first_use_) {
        this->forward_stream_->submit(this->forward_primitives_).wait();
        this->forward_first_use_ = false;
    } else {
        this->forward_stream_->rerun().wait();
    }
    LOG(INFO) << "    y={" << y[0] << "," << y[1] << ","
                           << y[2] << "," << y[3] << "}";
    return 0;
}

template<typename T>
int Pooling<T>::backward(T*   gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
                         T*   x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
                         T*   gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
                         int* ws, int ws_d1, int ws_d2, int ws_d3, int ws_d4)
{
    LOG(INFO) << "Pooling backward";
    LOG(INFO) << "    xdim=(" << x_d1  << "," << x_d2  << ","
                              << x_d3  << "," << x_d4  << ")";
    LOG(INFO) << "    ydim=(" << gy_d1 << "," << gy_d2 << ","
                              << gy_d3 << "," << gy_d4 << ")";
    LOG(INFO) << "    gy={"   << gy[0] << "," << gy[1] << ","
                              << gy[2] << "," << gy[3] << "}";
    LOG(INFO) << "    x={"    << x[0]  << "," << x[1]  << ","
                              << x[2]  << "," << x[3]  << "}";

    user_x_mem_ ->set_data_handle(x);
    user_gx_mem_->set_data_handle(gx);
    user_gy_mem_->set_data_handle(gy);
    if (ws != NULL)
        workspace_memory_->set_data_handle(ws);
    if (this->backward_first_use_) {
        this->backward_stream_->submit(this->backward_primitives_).wait();
        this->backward_first_use_ = false;
    } else {
        this->backward_stream_->rerun().wait();
    }
    LOG(INFO) << "    gx={" << gx[0] << "," << gx[1] << ","
                            << gx[2] << "," << gx[3] << "}";
    return 0;
}

template class Pooling<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s