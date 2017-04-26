/*
 *COPYRIGHT
 *All modification made by Intel Corporation: © 2017 Intel Corporation.
 *Copyright (c) 2015 Preferred Infrastructure, Inc.
 *Copyright (c) 2015 Preferred Networks, Inc.
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *THE SOFTWARE.
 *
 *
 *######################################################################
 *# The CuPy is designed based on NumPy's API.
 *# CuPy's source code and documents contain the original NumPy ones.
 *######################################################################
 *Copyright (c) 2005-2016, NumPy Developers.
 *All rights reserved.
 *
 *Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are
 *met:
 *
 *    * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    * Neither the name of the NumPy Developers nor the names of any
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *######################################################################
 */


#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "relu.h"

using namespace mkldnn;

extern engine cpu_engine;

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


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s