#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "convolution.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
Convolution2D<T>::Convolution2D(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                                T* W, int W_d1, int W_d2, int W_d3, int W_d4,
                                T* b, int b_d1,
                                T* y, int y_d1, int y_d2, int y_d3, int y_d4,
                                int s1, int s2,
                                int p1, int p2)
{
    LOG(INFO) << "Convolution CTOR, with b";

    LOG(INFO) << "x =(" << x_d1 << "," << x_d2 << "," << x_d3 << "," << x_d4 << ")";
    LOG(INFO) << "W =(" << W_d1 << "," << W_d2 << "," << W_d3 << "," << W_d4 << ")";
    LOG(INFO) << "b =(" << b_d1 << ")";
    LOG(INFO) << "y =(" << y_d1 << "," << y_d2 << "," << y_d3 << "," << y_d4 << ")";

    memory::dims src_tz = {x_d1, x_d2, x_d3, x_d4};
    memory::dims weights_tz = {W_d1, W_d2, W_d3, W_d4};
    memory::dims dst_tz = {y_d1, y_d2, y_d3, y_d4};
    memory::dims strides = {s1, s2};
    memory::dims bias_tz = {b_d1};

    auto padding = {p1, p2};

    /* create memory for user data */
    auto user_src_memory = new memory({{{src_tz}, memory_data_type<T>(),
                                      memory::format::nchw}, cpu_engine}, x);
    auto user_weights_memory = new memory({{{weights_tz},
                                          memory_data_type<T>(), memory::format::oihw}, cpu_engine}, W);

    primitive* user_bias_memory = NULL;
    if (b != NULL)
        user_bias_memory = new memory({{{bias_tz},
                                      memory_data_type<T>(), memory::format::x}, cpu_engine}, b);

    /* create memory descriptors for convolution data w/ no specified format */
    auto src_md = new memory::desc({src_tz}, memory_data_type<T>(),
                                   memory::format::any);

    memory::desc* bias_md = NULL;
    if (b != NULL)
        bias_md = new memory::desc({bias_tz}, memory_data_type<T>(),
                                   memory::format::any);
    auto weights_md = new memory::desc({weights_tz},
                                       memory_data_type<T>(), memory::format::any);
    auto dst_md = new memory::desc({dst_tz}, memory_data_type<T>(),
                                   memory::format::any);

    /* create a convolution */
    convolution_forward::desc* fwd_desc = NULL;
    if (b != NULL)
        fwd_desc = new convolution_forward::desc(prop_kind::forward,
                                                 convolution_direct, *src_md, *weights_md, *bias_md,
                                                 *dst_md, strides, padding, padding,
                                                 padding_kind::zero);
    else
        fwd_desc = new convolution_forward::desc(prop_kind::forward,
                                                 convolution_direct, *src_md, *weights_md,
                                                 *dst_md, strides, padding, padding,
                                                 padding_kind::zero);

    auto fwd_prim_desc = new convolution_forward::primitive_desc(*fwd_desc, cpu_engine);


    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    auto src_memory = user_src_memory;
    auto weights_memory = user_weights_memory;
    auto dst_memory = new memory(fwd_prim_desc->dst_primitive_desc());

    /* create convolution primitive and add it to net */
    primitive* fwd;
    if (b != NULL)
        fwd = new convolution_forward(*fwd_prim_desc, *src_memory,
                                      *weights_memory, *user_bias_memory, *dst_memory);
    else
        fwd = new convolution_forward(*fwd_prim_desc, *src_memory,
                                      *weights_memory, *dst_memory);

    primitives_.push_back(*fwd);
    stream_ = new stream(stream::kind::eager);
}

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

template class Convolution2D<float>;
