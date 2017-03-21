#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "relu.h"

using namespace mkldnn;

extern engine cpu_engine;

#if 0
template<typename T>
Relu<T>::Relu(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
              T* y, int y_d1, int y_d2, int y_d3, int y_d4)

{
    //auto cpu_engine = engine(engine::cpu, 0);

    //asm("int $3");
    memory::dims relu_src_tz = {x_d1, x_d2, x_d3, x_d4};
    memory::dims relu_dst_tz = {y_d1, y_d2, y_d3, y_d4};

    /* create memory for user data */
    auto relu_user_src_memory = new memory({{{relu_src_tz}, memory::data_type::f32,
        memory::format::nchw}, cpu_engine}, x);
    auto relu_dst_memory = new memory({{{relu_src_tz}, memory::data_type::f32,
        memory::format::nchw}, cpu_engine}, y);

    /* create memory descriptors for relu data w/ no specified format */
    auto relu_src_md = new memory::desc({relu_src_tz}, memory::data_type::f32,
        memory::format::nchw);
    //    memory::format::any);
    //auto relu_dst_md = new memory::desc({relu_dst_tz}, memory::data_type::f32,
    //    memory::format::any);

    /* create reorders between user and data if it is needed and
     *  add it to net_ before relu*/
    auto relu_src_memory = relu_user_src_memory;
#if 1
    if (memory::primitive_desc(*relu_src_md, cpu_engine) != relu_user_src_memory->get_primitive_desc()) {
        //relu_src_memory = new memory(*relu_src_md);
        //primitives_.push_back(reorder(*relu_user_src_memory, *relu_src_memory));
        abort();
    }
#endif

    const double negative_slope = 1.0;

    /* create relu primitive and add it to net_ */
    auto relu_desc = new relu_forward::desc(prop_kind::forward,
            *relu_src_md, negative_slope);
    auto relu_prim_desc = new relu_forward::primitive_desc(*relu_desc, cpu_engine);

    auto relu_fwd = new relu_forward(*relu_prim_desc, *relu_src_memory,
            *relu_dst_memory);

    primitives_.push_back(*relu_fwd);
    stream_ = new stream(stream::kind::eager);
    //stream_->submit(primitives_);
#if 0
    net_.push_back(relu_forward(relu_prim_desc, relu_src_memory,
            relu_dst_memory));
#endif

}
#endif

template<typename T>
Relu<T>::Relu(): relu_user_src_memory_(NULL), relu_dst_memory_(NULL)
               , relu_src_md_(NULL), relu_desc_(NULL), relu_prim_desc_(NULL)
               , relu_fwd_(NULL), fw_stream_(NULL)
               , relu_diff_dst_memory_(NULL), relu_diff_dst_md_(NULL)
               , relu_bwd_desc_(NULL), relu_bwd_pd_(NULL)
               , relu_bwd_(NULL), bw_stream_(NULL)

{

}

template<typename T>
int Relu<T>::forward_setup(T* x, int x_size,
                           T* y, int y_size)
{
    memory::dims relu_src_tz = {x_size};
    memory::dims relu_dst_tz = {y_size};

    /* create memory for user data */
    relu_user_src_memory_ = new memory({{{relu_src_tz}, memory::data_type::f32,
        memory::format::x}, cpu_engine}, x);
    relu_dst_memory_ = new memory({{{relu_src_tz}, memory::data_type::f32,
        memory::format::x}, cpu_engine}, y);

    /* create memory descriptors for relu data w/ no specified format */
    relu_src_md_ = new memory::desc({relu_src_tz}, memory::data_type::f32,
        memory::format::x);
    //    memory::format::any);
    //auto relu_dst_md = new memory::desc({relu_dst_tz}, memory::data_type::f32,
    //    memory::format::any);

    /* create reorders between user and data if it is needed and
     *  add it to net_ before relu*/
    auto relu_src_memory = relu_user_src_memory_;
#if 1
    if (memory::primitive_desc(*relu_src_md_, cpu_engine) != relu_user_src_memory_->get_primitive_desc()) {
        //src_memory = new memory(*src_md_);
        //primitives_.push_back(reorder(*user_src_memory_, *src_memory));
        abort();
    }
#endif

    const double negative_slope = 0.0;//1.0;

    /* create relu primitive and add it to net_ */
    relu_desc_ = new relu_forward::desc(prop_kind::forward,
            *relu_src_md_, negative_slope);
    relu_prim_desc_ = new relu_forward::primitive_desc(*relu_desc_, cpu_engine);

    relu_fwd_ = new relu_forward(*relu_prim_desc_, *relu_src_memory,
            *relu_dst_memory_);

    fw_primitives_.push_back(*relu_fwd_);
    fw_stream_ = new stream(stream::kind::eager);

    return 0;
}

template<typename T>
int Relu<T>::forward(T* x, int x_size,
                     T* y, int y_size)
{

    LOG(INFO) << "Convolution forward";
    if (!fw_stream_) {
        forward_setup(x, x_size, y, y_size);
    }
    fw_stream_->submit(fw_primitives_).wait();
    return 0;
}

template<typename T>
int Relu<T>::backward_setup(T* x, int x_size,
                      T* gy, int gy_size,
                      T* gx, int gx_size)
{
    const double negative_slope = 0.0;//1.0;

    /* Backward relu */
    memory::dims relu_diff_src_tz = {gx_size};
    memory::dims relu_diff_dst_tz = {gy_size};

    relu_diff_src_memory_ = new memory({{{relu_diff_src_tz}, memory::data_type::f32,
        memory::format::x}, cpu_engine}, gx);
    relu_diff_dst_memory_ = new memory({{{relu_diff_dst_tz}, memory::data_type::f32,
        memory::format::x}, cpu_engine}, gy);

    /* create memory descriptors for relu data w/ no specified format */
    relu_diff_dst_md_ = new memory::desc({relu_diff_dst_tz}, memory::data_type::f32,
        memory::format::x);

    /* create backward relu primitive_descriptor */
    relu_bwd_desc_ = new relu_backward::desc(*relu_diff_dst_md_, *relu_src_md_, negative_slope);
    relu_bwd_pd_ = new relu_backward::primitive_desc(*relu_bwd_desc_, cpu_engine, *relu_prim_desc_);

    /* create memory for relu diff src */
    //auto relu_diff_src_memory = memory(relu_bwd_pd.diff_src_primitive_desc());

    /* finally create a backward relu primitive */
    relu_bwd_ = new relu_backward(*relu_bwd_pd_, *relu_user_src_memory_,
                                  *relu_diff_dst_memory_, *relu_diff_src_memory_);
    bw_primitives_.push_back(*relu_bwd_);
    bw_stream_ = new stream(stream::kind::eager);

    LOG(INFO) << "Convolution backward";
    return 0;
}

template<typename T>
int Relu<T>::backward(T* x, int x_size,
                      T* gy, int gy_size,
                      T* gx, int gx_size)
{
    if (!bw_stream_) {
        backward_setup(x, x_size, gy, gy_size, gx, gx_size);
    }
    bw_stream_->submit(bw_primitives_).wait();
    return 0;
}

template class Relu<float>;
