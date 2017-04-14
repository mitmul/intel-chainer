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


#ifndef _MKLDNN_SUM_H
#define _MKLDNN_SUM_H

#include <glog/logging.h>
#include <iostream>
#include <mkldnn.hpp>
#include <vector>
#include "layer.h"
#include "layer_factory.h"

template <typename T>
class Sum : public Layer<T> {
public:
    struct sum_data {
        T* data;
        mkldnn::memory::dims dims;   
    };

    Sum<T>();
    ~Sum<T>();
      
    void sum_setup(int num_sum, sum_data* sum_input,
            T* y, int y_d1, int y_d2, int y_d3, int y_d4);

    void sum(int num_sum, char** data, int* n, int* c, int* h, int* w,
            T* y, int y_d1, int y_d2, int y_d3, int y_d4);

private:
       bool first_run_ = true;

       std::shared_ptr<mkldnn::stream> sum_stream_;
       std::vector<mkldnn::memory::primitive_desc> srcs_mem_pd_;
       std::vector<std::shared_ptr<mkldnn::memory>> srcs_mem_;
       std::vector<mkldnn::primitive::at> srcs_mem_at_;
       std::vector<double> scale_;

       std::shared_ptr<mkldnn::memory::desc> user_dst_md_;
       std::shared_ptr<mkldnn::memory> user_dst_mem_;
       std::shared_ptr<mkldnn::memory> dst_mem_;
       mkldnn::primitive sum_reorder_dst_;

       std::shared_ptr<mkldnn::sum::primitive_desc> sum_pd_;
       std::shared_ptr<mkldnn::sum> sum_prim_;
       std::vector<mkldnn::primitive> sum_prims_;
};

#endif  // _MKLDNN_SUM_H
