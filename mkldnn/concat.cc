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
#include "concat.h"
#include "utils.h"

using namespace mkldnn;
using namespace std;

extern engine cpu_engine;

template<typename T>
Concat<T>::Concat() {
    fwd_stream_.reset(new stream(stream::kind::eager));
    bwd_stream_.reset(new stream(stream::kind::eager));
}

template<typename T>
Concat<T>::~Concat() {
}

template<typename T>
void Concat<T>::forward_setup(int num_concats, Concat<T>::concat_data* concat_input,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4,
        int axis) {
//    LOG(INFO) << "Enter forward_setup"; 
//    LOG(INFO) << "y_d1=" << y_d1 << "; y_d2=" << y_d2 << "; y_d3="<<y_d3 << "; y_d4=" << y_d4;
    output_tz_ = {y_d1, y_d2, y_d3, y_d4}; //dst memory dim
    axis_ = axis; //yli135: currently, seems only support 1

    for (int i = 0; i < num_concats; i++) {
        memory::dims input_tz = concat_input[i].dims;
        memory::format src_mfmt = memory::format::nchw;

        shared_ptr<memory::primitive_desc> input_mpd;
        input_mpd.reset(new memory::primitive_desc({input_tz, memory_data_type<T>(), src_mfmt}, cpu_engine));
        srcs_pd_.push_back(*input_mpd);
       
        std::shared_ptr<memory> input_mem;
        input_mem.reset(new memory({{{input_tz},memory_data_type<T>(), src_mfmt}, cpu_engine}));
       
        fwd_input_primitives_.push_back(input_mem);
        fwd_input_primitives_at_.push_back(*fwd_input_primitives_[i]);
    }

    //set the user des memory primitive/desc
    user_dst_md_.reset(new memory::desc(output_tz_, memory_data_type<T>(), memory::format::any));
    user_dst_mem_.reset(new memory(
                {{{output_tz_}, memory_data_type<T>(), memory::format::nchw}, cpu_engine}));

    // create concat primitive desc and primitive
    fwd_concat_pd_.reset(
            new concat::primitive_desc(*user_dst_md_, axis_, srcs_pd_));

    /*
     * Check whether need to reorder for dst mem
     */
    dst_mem_ = user_dst_mem_;
    bool fwd_reorder_concat_dst = false;
    if (memory::primitive_desc(fwd_concat_pd_.get()->dst_primitive_desc())
            != user_dst_mem_.get()->get_primitive_desc()) {
        LOG(INFO) << "concat fwd reorder dst memory";
        dst_mem_.reset(
                new memory(fwd_concat_pd_.get()->dst_primitive_desc()));
        concat_reorder_dst_ = reorder(*dst_mem_, *user_dst_mem_);
        fwd_reorder_concat_dst = true;
    }

    fwd_concat_prim_.reset(
            new concat(*fwd_concat_pd_, fwd_input_primitives_at_, *dst_mem_));
   
    /* push primitives into primitive vector */
    fwd_primitives_.push_back(*fwd_concat_prim_);
    if (fwd_reorder_concat_dst) {
        fwd_primitives_.push_back(concat_reorder_dst_);
    }
}

template<typename T>
void Concat<T>::forward(int num_concats, char** data, int* n, int* c, int* h, int* w,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4,
        int axis) {
    /*
     * transfer the python tuple data to native concat_data struct
     */
    concat_data concat_input[num_concats];
    for(int i = 0; i < num_concats; i++) {
        concat_input[i].data = (T*)data[i];
        concat_input[i].dims = {n[i], c[i], h[i], w[i]};       
    }

    /*
     * Call forward setup
     */
    if (fwd_concat_prim_ == NULL) {
        forward_setup(num_concats, concat_input,
                y, y_d1, y_d2, y_d3, y_d4,
                axis);
    }
   
    /*
     * set memory data handle for input memory
     */
    for (int i = 0; i < num_concats; i++) {
        fwd_input_primitives_[i]->set_data_handle(concat_input[i].data);
    }
   
    /* set memory handle for dst memory */
    user_dst_mem_->set_data_handle(y);

    if (fwd_first_run_) {
        fwd_stream_->submit(fwd_primitives_).wait();
    } else {
        fwd_stream_->rerun().wait();
    }
}

template<typename T>
void Concat<T>::backward_setup(int num_concats, Concat<T>::concat_data* concat_output,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        int axis) {
    /* init the offset */
    memory::dims offsets = {0, 0, 0, 0};
   
    /*
     * prepare user diff dst memory
     * reuse memory desc as fwd: user_dst_mpd_
     * only create a new memory used for diff dst to set handle
     * */
    user_diff_dst_mpd_.reset(new memory::primitive_desc(
                {output_tz_, memory_data_type<T>(), memory::format::nchw}, cpu_engine));
    user_diff_dst_prim_.reset(new memory(
                {{{output_tz_}, memory_data_type<T>(), memory::format::nchw}, cpu_engine}));

    /*
     * prepare the user diff src memory
     */
    for (int i = 0; i < num_concats; i++) {
        memory::dims diff_input_tz = concat_output[i].dims;
        memory::format diff_src_mfmt = memory::format::nchw;

        shared_ptr<memory::primitive_desc> diff_input_mpd;
        diff_input_mpd.reset(new memory::primitive_desc({diff_input_tz, memory_data_type<T>(), diff_src_mfmt}, cpu_engine));
        diff_srcs_pd_.push_back(*diff_input_mpd);
       
        std::shared_ptr<memory> diff_input_mem;
        diff_input_mem.reset(new memory({{{diff_input_tz},memory_data_type<T>(), diff_src_mfmt}, cpu_engine}));

        bwd_reorder_diff_src_mem_.push_back(*diff_input_mem);
       
        std::shared_ptr<view::primitive_desc> view_pd;
        view_pd.reset(
                new view::primitive_desc(*user_diff_dst_mpd_, diff_input_tz, offsets));

        std::shared_ptr<reorder::primitive_desc> reorder_pd;
        reorder_pd.reset(
                new reorder::primitive_desc(view_pd.get()->dst_primitive_desc(), *diff_input_mpd));

        shared_ptr<mkldnn::reorder> reorder_prim;
        reorder_prim.reset(
                new reorder(*reorder_pd, *user_diff_dst_prim_, bwd_reorder_diff_src_mem_[i]));
        bwd_primitives_.push_back(*reorder_prim);

        offsets[axis_] += diff_input_tz[axis_];
    }
}

template<typename T>
void Concat<T>::backward(int num_concats, char** data, int* n, int* c, int* h, int* w,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        int axis) {
    /*
     * transfer the python tuple data to native concat_data struct
     */
    concat_data concat_output[num_concats];
    for(int i = 0; i < num_concats; i++) {
        concat_output[i].data = (T*)data[i];
        concat_output[i].dims = {n[i], c[i], h[i], w[i]};       
    }
    if (reorders_.size() == 0) {
        backward_setup(num_concats, concat_output,
                gy, gy_d1, gy_d2, gy_d3, gy_d4,
                axis);
    }

    /*
     * set memory data handle for diff src memory
     */
    for (int i = 0; i < num_concats; i++) {
        bwd_reorder_diff_src_mem_[i].set_data_handle(concat_output[i].data);
    }
   
    /* set memory handle for dst memory */
    user_diff_dst_prim_->set_data_handle(gy);
   
    if (bwd_first_run_) {
        bwd_stream_->submit(bwd_primitives_).wait();
    } else {
        bwd_stream_->rerun().wait();
    }
}

template class Concat<float>;
template class Concat<double>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
