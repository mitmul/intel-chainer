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
Relu<T>::Relu()
: relu_fwd_user_src_mem_(NULL), relu_fwd_dst_mem_(NULL)
               , relu_fwd_src_md_(NULL), relu_fwd_desc_(NULL), relu_fwd_pd_(NULL)
               , relu_fwd_(NULL), fwd_stream_(NULL)
               , relu_diff_dst_mem_(NULL), relu_diff_dst_md_(NULL)
               , relu_bwd_desc_(NULL), relu_bwd_pd_(NULL)
               , relu_bwd_(NULL), bwd_stream_(NULL)
{

}

template<typename T>
int Relu<T>::forward_setup(T* x, int x_size,
                           T* y, int y_size)
{
    memory::dims relu_src_tz = {x_size};
    memory::dims relu_dst_tz = {y_size};

    /* create memory for user data */
    relu_fwd_user_src_mem_.reset(new memory({{{relu_src_tz}, memory::data_type::f32,
        memory::format::x}, cpu_engine}, x));
    relu_fwd_dst_mem_.reset(new memory({{{relu_src_tz}, memory::data_type::f32,
        memory::format::x}, cpu_engine}, y));

    /* create memory descriptors for relu data w/ no specified format */
    relu_fwd_src_md_.reset(new memory::desc({relu_src_tz}, memory::data_type::f32,
        memory::format::x));

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
void Relu<T>::fwd_reset_mem(T* x,
                            T* y)
{
        relu_fwd_user_src_mem_->set_data_handle(x);
        relu_fwd_dst_mem_->set_data_handle(y);
}

template<typename T>
int Relu<T>::forward(T* x, int x_size,
                     T* y, int y_size)
{
    LOG(INFO) << "forward: " << x << " : " << x_size << " : " << y << " : " << y_size;
    //LOG(INFO) << "Convolution forward";
    if (!fwd_stream_) {
        forward_setup(x, x_size, y, y_size);
        fwd_reset_mem(x, y);
        fwd_stream_->submit(fwd_primitives_).wait();
    } else {
        fwd_reset_mem(x, y);
        fwd_stream_->rerun().wait();
    }
    return 0;
}

template<typename T>
int Relu<T>::test_buf(
                     T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                     T* y, int y_size
                    )
{
    LOG(INFO) << "test buf: " << x << " : " << x_d1 << " : " << x_d2 << " : " << x_d3 << " : " << x_d4 << " : " << y << " : " << y_size;
    return 0;
}

template<typename T>
int Relu<T>::backward_setup(T* x, int x_size,
                      T* gy, int gy_size,
                      T* gx, int gx_size)
{
    const double negative_slope = 0.0;//1.0;

    /* Backward relu */
    memory::dims relu_src_tz = {x_size};
    memory::dims relu_diff_src_tz = {gx_size};
    memory::dims relu_diff_dst_tz = {gy_size};

    relu_diff_src_mem_.reset(new memory({{{relu_diff_src_tz}, memory::data_type::f32,
        memory::format::x}, cpu_engine}, gx));
    relu_diff_dst_mem_.reset(new memory({{{relu_diff_dst_tz}, memory::data_type::f32,
        memory::format::x}, cpu_engine}, gy));

#if 0
    /* create memory descriptors for relu data with user memory format */
    relu_bwd_src_md_.reset(memory::desc({relu_src_tz}, memory::data_type::f32,
        memory::format::x);
    relu_bwd_user_src_mem_.reset(new memory({{{relu_src_tz}, memory::data_type::f32,
        memory::format::x}, cpu_engine}, x));
    auto relu_fwd_desc = relu_forward::desc(prop_kind::forward,
            relu_src_md, negative_slope);
    auto relu_fwd_pd = new relu_forward::primitive_desc(relu_fwd_desc, cpu_engine);
#endif

    /* create memory descriptors for relu data with user memory format */
    relu_diff_dst_md_.reset(new memory::desc({relu_diff_dst_tz}, memory::data_type::f32,
        memory::format::x));

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
void Relu<T>::bwd_reset_mem(T* x,
                            T* gy,
                            T* gx)
{
    relu_fwd_user_src_mem_->set_data_handle(x);
    relu_diff_dst_mem_->set_data_handle(gy);
    relu_diff_src_mem_->set_data_handle(gx);
}

template<typename T>
int Relu<T>::backward(T* x, int x_size,
                      T* gy, int gy_size,
                      T* gx, int gx_size)
{
    LOG(INFO) << "backward: " << x << " : " << x_size << " : " << gy << " : " << gy_size << " : " << gx << " : " << gx_size;
    if (!bwd_stream_) {
        backward_setup(x, x_size, gy, gy_size, gx, gx_size);
        bwd_reset_mem(x, gy, gx);
        bwd_stream_->submit(bwd_primitives_).wait();
    } else {
        bwd_reset_mem(x, gy, gx);
        bwd_stream_->rerun().wait();
    }
    return 0;
}

template class Relu<float>;
