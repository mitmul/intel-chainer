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
#include "common.h"
#include "mkldnn.hpp"
#include "sum.h"
#include "utils.h"

using namespace mkldnn;
using namespace std;

extern engine cpu_engine;

template<typename T>
Sum<T>::Sum() {
    sum_stream_.reset(new stream(stream::kind::eager));
}

template<typename T>
Sum<T>::~Sum() {
}

template<typename T>
void Sum<T>::sum_setup(int num_sum, Sum<T>::sum_data* sum_input,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4) {
    LOG(INFO) << "Enter sum forward_setup"; 
    LOG(INFO) << "y_d1=" << y_d1 << "; y_d2=" << y_d2 << "; y_d3="<<y_d3 << "; y_d4=" << y_d4;
    memory::dims output_tz = {y_d1, y_d2, y_d3, y_d4};
    memory::format src_mfmt = memory::format::nchw;

      for (int i = 0; i < num_sum; i++) {
          memory::dims input_tz = sum_input[i].dims;

          // input should have same dim as output
          assert(output_tz == input_tz);

          // sum primitive doesn't expose API to get inputs' internal format
          // so here always set input internal format as nchw
          shared_ptr<memory::primitive_desc> input_mem_pd;
          input_mem_pd.reset(new memory::primitive_desc(
                      {input_tz, memory_data_type<T>(), memory::format::nchw}, cpu_engine));
          srcs_mem_pd_.push_back(*input_mem_pd);
          scale_.push_back((double)1.0);

          shared_ptr<memory> input_mem;
          input_mem.reset(new memory({{{input_tz}, memory_data_type<T>(), src_mfmt}, cpu_engine}, dummy));
          srcs_mem_.push_back(input_mem);
          srcs_mem_at_.push_back(*srcs_mem_[i]);
      }
       
      //create user dst memory prim/desc
      user_dst_md_.reset(new memory::desc(output_tz, memory_data_type<T>(),memory::format::any));
      user_dst_mem_.reset(new memory({{{output_tz}, memory_data_type<T>(), memory::format::nchw}, cpu_engine}, dummy));
   
      sum_pd_.reset(new mkldnn::sum::primitive_desc(*user_dst_md_, scale_,  srcs_mem_pd_));



      /*
       * Check whether need to reorder for dst mem
       */  
      dst_mem_ = user_dst_mem_;
      bool reorder_sum_dst = false;
      if (sum_pd_.get()->dst_primitive_desc()
              != user_dst_mem_.get()->get_primitive_desc()) {
          LOG(INFO) << "sum reorder dst memory";
          dst_mem_.reset(
                  new memory(sum_pd_.get()->dst_primitive_desc()));
          sum_reorder_dst_ = reorder(*dst_mem_, *user_dst_mem_);
          reorder_sum_dst = true;
      }

      sum_prim_.reset(new mkldnn::sum(*sum_pd_, srcs_mem_at_, *dst_mem_));

      sum_prims_.push_back(*sum_prim_);
      if (reorder_sum_dst) {
          sum_prims_.push_back(sum_reorder_dst_);
      }

}

template<typename T>
void Sum<T>::sum(int num_sum, char** data, int* n, int* c, int* h, int* w,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4) {

    sum_data sum_input[num_sum];
    for (int i = 0; i < num_sum; i++) {
        sum_input[i].data = (T*)data[i];
        sum_input[i].dims = {n[i], c[i], h[i], w[i]};
    }

    if (sum_prim_ == NULL) {
        sum_setup(num_sum, sum_input,
                y, y_d1, y_d2, y_d3, y_d4);
    }

    /*
     * set mem data handle for input memory
     */
    for (int i = 0; i < num_sum; i++) {
        srcs_mem_[i]->set_data_handle(sum_input[i].data);
    }

    /* set mem handle for dst mem */
    user_dst_mem_->set_data_handle(y);

    if (first_run_) {
        sum_stream_->submit(sum_prims_).wait();
        first_run_ = false;
    } else {
        sum_stream_->rerun().wait();
    }
}

template class Sum<float>;
template class Sum<double>;
