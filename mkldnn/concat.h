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


#ifndef _MKLDNN_CONCAT_H
#define _MKLDNN_CONCAT_H

#include <glog/logging.h>
#include <iostream>
#include <mkldnn.hpp>
#include <vector>
#include "layer.h"
#include "layer_factory.h"

template <typename T>
class Concat : public Layer<T> {
public:
    struct concat_data {
        T* data;
        mkldnn::memory::dims dims;    
    };

    Concat<T>();
    ~Concat<T>();
       
    void forward_setup(int num_concats, concat_data* concat_input,
            T* y, int y_d1, int y_d2, int y_d3, int y_d4,
            int axis); 

    void forward(int num_concats, char** data, int* n, int* c, int* h, int* w,
            T* y, int y_d1, int y_d2, int y_d3, int y_d4,
            int axis);

    void backward_setup(int num_concats, concat_data* concat_output,
            T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
            int axis); 
    
    void backward(int num_concats, char** data, int* n, int* c, int* h, int* w,
            T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
            int axis);
private:
        
       int axis_;
       bool fwd_first_run_ = true;
       bool bwd_first_run_ = true;

       //forward
       mkldnn::memory::dims output_tz_; //output dims
       std::shared_ptr<mkldnn::memory::primitive_desc> user_dst_mpd_; //user dst mpd
       std::shared_ptr<mkldnn::memory::desc> user_dst_md_; //user dst md
       std::shared_ptr<mkldnn::memory> user_dst_mem_; // usr dst memory
       std::shared_ptr<mkldnn::memory> dst_mem_; // mkldnn dst memory
       std::shared_ptr<mkldnn::stream> fwd_stream_; // fwd stream
       std::vector<mkldnn::primitive> fwd_primitives_; //fwd primitive vector
       std::shared_ptr<mkldnn::concat::primitive_desc> fwd_concat_pd_; //fwd prim desc
       std::shared_ptr<mkldnn::concat> fwd_concat_prim_; // fwd primitive
       mkldnn::primitive concat_reorder_dst_; //reorder y
       std::vector<mkldnn::primitive::at> fwd_input_primitives_at_; //fwd input primitives     
       std::vector<std::shared_ptr<mkldnn::memory>> fwd_input_primitives_; // fwd input memory
       std::vector<mkldnn::memory::primitive_desc> srcs_pd_; //src primitve desc vector



       //backward
       std::shared_ptr<mkldnn::memory::primitive_desc> user_diff_dst_mpd_; //user diff dst mpd
       std::shared_ptr<mkldnn::memory> user_diff_dst_prim_; // usr diff dst memory
       std::vector<mkldnn::memory> bwd_reorder_diff_src_mem_; // diff src memory vectory
       std::vector<mkldnn::reorder> reorders_;
       std::vector<mkldnn::primitive> bwd_primitives_; //bwd primitive vector
       std::shared_ptr<mkldnn::stream> bwd_stream_;
       std::vector<mkldnn::memory::primitive_desc> diff_srcs_pd_; //diff src primitve desc vector
      
        
};

#endif  // _MKLDNN_CONCAT_H


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s