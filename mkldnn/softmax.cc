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
#include <vector>
#include "mkldnn.hpp"
#include "softmax.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
void Softmax<T>::update_user_mem(T* x, T* y)
{
    src_user = x;
    dst_user = y;
}

template<typename T>
void Softmax<T>::update_user_data(std::shared_ptr<mkldnn::memory> src_mem,
                                  std::shared_ptr<mkldnn::memory> dst_mem)
{
    src_mem->set_data_handle(src_user);
    dst_mem->set_data_handle(dst_user);
}

// Class Softmax<T>
template<typename T> Softmax<T>*
Softmax<T>::softmax_create_forward(T* x, int dummy_x,
                                   T* y, int dummy_y,
                                   int* dims, int ndim, int axis)
{
    Softmax<T>* inst = NULL;

    // Useless parameter
    // Ravel 2D or 4D nparray to 1D to unify interface
    // In SWIG we have to use fixing transformation map to get C pointer
    (void)dummy_x;
    (void)dummy_y;

#define SOFTMAX_2D 2
#define SOFTMAX_4D 4
    if (SOFTMAX_2D == ndim) {
        // LayerFactory get map
        if (!(inst =
            (Softmax_2D<T>*)LayerFactory<T>::get_instance().get_softmax2d_layer(dims[0], dims[1], axis))) {
            // New Softmax inst
            inst = new Softmax_2D<T>(dims, axis);

            // Setup forward
            inst->setup_forward();

            // LayerFactory set map
            LayerFactory<T>::get_instance().set_softmax2d_layer(dims[0], dims[1], axis, inst);
        } else {
            ;
        }

        // Update current user memory
        // FIXME: Do not drop memory util finishing forward() in Python layer
        inst->update_user_mem(x, y);
    } else if (SOFTMAX_4D == ndim) {
        // LayerFactory get map
        if (!(inst =
            (Softmax_4D<T>*)LayerFactory<T>::get_instance().get_softmax4d_layer(dims[0], dims[1], dims[2], dims[3], axis))) {
            // New Softmax inst
            inst = new Softmax_4D<T>(dims, axis);

            // Setup forward
            inst->setup_forward();

            // LayerFactory set map
            LayerFactory<T>::get_instance().set_softmax4d_layer(dims[0], dims[1], dims[2], dims[3], axis, inst);
        } else {
            ;
        }

        // Update current user memory
        // FIXME: Do not drop memory util finishing forward() in Python layer
        inst->update_user_mem(x, y);
    } else {
        ; //Not supported;
    }

    return inst;
};

// Class Softmax_2D<T>
template<typename T>
int Softmax_2D<T>::get_res_size()
{
    return sizeof(T) * dims[0] * dims[1];
}

template<typename T>
int Softmax_2D<T>::setup_forward()
{
    // (1) One shape specifies a certain primitive
    memory::dims src_tz = {dims[0], dims[1]};
    memory::dims dst_tz = {dims[0], dims[1]};

    // (2) Prepare user memory primitive and memory desc
    src_mem.reset(new memory({{{src_tz}, memory_data_type<T>(),
                                      memory::format::nc}, cpu_engine}));
    dst_mem.reset(new memory({{{dst_tz}, memory_data_type<T>(),
                                      memory::format::nc}, cpu_engine}));

    // Use set_data_handle to switch param ptr of created primitive
    // Need persistent memory no more
    // src = (T*)src_mem->get_data_handle();
    // dst = (T*)dst_mem->get_data_handle();

    src_md.reset(new memory::desc({src_tz}, memory_data_type<T>(),
                                   memory::format::nc));

    // (3) Create softmax primitive desc
    fwd_desc.reset(new softmax_forward::desc(prop_kind::forward_scoring, *src_md, axis));
    fwd_pd.reset(new softmax_forward::primitive_desc(*fwd_desc, cpu_engine));

    // (4) Create src primitive and src_reorder primitive
    //
    // (No Reorder in Softmax)
    //
    // primitive *src_mem = user_src_mem, *src_reorder = NULL;
    // if (memory::primitive_desc(fwd_pd->src_primitive_desc()) !=
    //     user_src_mem->get_primitive_desc()){
    //     src_mem = new memory(fwd_pd->src_primitive_desc());
    //     src_reorder = new reorder(*user_src_mem, *src_mem);
    // }

    // (5) Create dst primitive and dst_reorder primitive
    //
    // (No Reorder in Softmax)
    //
    // primitive *dst_mem = user_dst_mem, *dst_reorder = NULL;
    // if (memory::primitive_desc(fwd_pd->dst_primitive_desc()) !=
    //     user_dst_mem->get_primitive_desc()) {
    //     dst_mem = new memory(fwd_pd->dst_primitive_desc());
    //     dst_reorder = new reorder(*dst_mem, *user_dst_mem);
    // }

    // (6) Create softmax primitive
    fwd.reset(new softmax_forward(*fwd_pd, *src_mem, *dst_mem));

    // (7) Construct net
    fwd_primitives_.push_back(*fwd);
    fwd_stream_.reset(new stream(stream::kind::eager));

    return 0;
}

template<typename T>
int Softmax_2D<T>::setup_backward()
{
    return 0;
}

template<typename T>
int Softmax_2D<T>::forward()
{
    // Update data ptr in src_mem and dst_mem
    this->update_user_data(src_mem, dst_mem);

    // Launch stream
    if (this->is_first_fwd()) {
        fwd_stream_->submit(fwd_primitives_).wait();
    this->mark_first_fwd();
    } else {
        fwd_stream_->rerun().wait();
    }

    return 0;
}

template<typename T>
int Softmax_2D<T>::backward()
{
    return 0;
}

// Class Softmax_4D
template<typename T>
int Softmax_4D<T>::get_res_size()
{
    return sizeof(T) * dims[0] * dims[1] * dims[2] * dims[3];
}

template<typename T>
int Softmax_4D<T>::setup_forward()
{
    return 0;
}

template<typename T>
int Softmax_4D<T>::setup_backward()
{
    return 0;
}


template<typename T>
int Softmax_4D<T>::forward()
{
    return 0;
}

template<typename T>
int Softmax_4D<T>::backward()
{
    return 0;
}

template class Softmax<float>;
template class Softmax_2D<float>;
template class Softmax_4D<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
