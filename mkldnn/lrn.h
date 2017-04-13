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


#pragma once
#ifndef _LRN_H_
#define _LRN_H_


#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "layer.h"
#include "layer_factory.h"

template <typename T>
class LocalResponseNormalization : public Layer<T> 
{
public:
    struct lrn_params 
    {
        double alpha, beta,k;
        int local_size;
        mkldnn::prop_kind aprop_kind;
        mkldnn::algorithm aalgorithm;
        mkldnn::memory::format data_format;
        mkldnn::memory::format diff_data_format;
    };
public:
    LocalResponseNormalization(int n, double k, double alpha, double beta, mkldnn::algorithm alg_kind);
    ~LocalResponseNormalization();
public:
    int forward();

    static void do_forward(
        T*   x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
        T*   y,  int y_d1,  int y_d2,  int y_d3,  int y_d4,
        // T*   ws, int ws_d1, int ws_d2, int ws_d3, int ws_d4,
        int n, double k, double alpha, double beta,
        mkldnn::algorithm alg_kind = mkldnn::algorithm::lrn_across_channels) 
    {
        LOG(INFO) << "do forward";
        auto forward_object = get_forward_object(
            x_d1, x_d2, x_d3, x_d4, n, k, alpha, beta, alg_kind);
        LOG(INFO) << "forward";
        forward_object->forward(x,  x_d1,  x_d2,  x_d3,  x_d4,
                                y,  y_d1,  y_d2,  y_d3,  y_d4);
    }
    static void do_backward(
        T*   x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
        T*   gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        T*   gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
        // T*   ws, int ws_d1, int ws_d2, int ws_d3, int ws_d4,
        int n, double k, double alpha, double beta,
        mkldnn::algorithm alg_kind = mkldnn::algorithm::lrn_across_channels) 
    {
        auto backward_object = get_backward_object(
            x_d1, x_d2, x_d3, x_d4, n, k, alpha, beta, alg_kind);

        backward_object->backward(x,  x_d1,  x_d2,  x_d3,  x_d4,
                                  gy, gy_d1, gy_d2, gy_d3, gy_d4,
                                  gx, gx_d1, gx_d2, gx_d3, gx_d4);
    }
private:
    int backward(
        T* x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4);
    int backward_setup(
        T* x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4);
    void bwd_reset_mem(T* x,T* gy,T* gx);

    int forward(
        T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4);
    int forward_setup(
        T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4);

    void fwd_reset_mem(T* x,T* y);
protected:
    static LocalResponseNormalization<T>* get_forward_object(
        int x_d1, int x_d2, int x_d3, int x_d4,
        int n, double k, double alpha, double beta, mkldnn::algorithm alg_kind);

    static LocalResponseNormalization<T>* get_backward_object(
        int x_d1, int x_d2, int x_d3, int x_d4,
        int n, double k, double alpha, double beta, mkldnn::algorithm alg_kind);
private:
    lrn_params p_;

    //forward
    std::shared_ptr<mkldnn::memory>                           user_x_mem_;
    std::shared_ptr<mkldnn::memory>                           user_y_mem_;
    std::shared_ptr<mkldnn::memory>                           x_mem_;
    std::shared_ptr<mkldnn::memory>                           y_mem_;
    std::shared_ptr<mkldnn::memory::desc>                     x_md_;
    std::shared_ptr<mkldnn::memory::desc>                     y_md_;
    std::shared_ptr<mkldnn::memory>                           bw_x_mem_;
    std::shared_ptr<mkldnn::memory>                           gx_mem_;
    std::shared_ptr<mkldnn::memory>                           gy_mem_;
    std::shared_ptr<mkldnn::memory>                           workspace_mem_;

    std::shared_ptr<mkldnn::lrn_forward::desc> lrn_fwd_desc_;
    std::shared_ptr<mkldnn::lrn_forward::primitive_desc> lrn_fwd_pd_;
    std::shared_ptr<mkldnn::lrn_forward> lrn_fwd_;
    std::shared_ptr<mkldnn::stream> fwd_stream_;
    std::vector<mkldnn::primitive> fwd_primitives_;

    //backward
    std::shared_ptr<mkldnn::memory> lrn_bwd_user_src_mem_, lrn_diff_src_mem_, lrn_diff_dst_mem_;
    std::shared_ptr<mkldnn::memory::desc> lrn_bwd_src_desc_, lrn_diff_src_desc_, lrn_diff_dst_desc_;

    std::shared_ptr<mkldnn::lrn_backward::desc> lrn_bwd_desc_;
    std::shared_ptr<mkldnn::lrn_backward::primitive_desc> lrn_bwd_pd_;
    std::shared_ptr<mkldnn::lrn_backward> lrn_bwd_;
    std::shared_ptr<mkldnn::stream> bwd_stream_;
    std::vector<mkldnn::primitive> bwd_primitives_;


    mkldnn::primitive                         reorder_x_;
    mkldnn::primitive                         reorder_y_;
    // primitive                                 reorder_bwd_x_;
    mkldnn::primitive                         reorder_gx_;
    mkldnn::primitive                         reorder_gy_;

    std::shared_ptr<mkldnn::engine> eng_;

};

#endif // _LRN_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
