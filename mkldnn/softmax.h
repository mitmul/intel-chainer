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


#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

#include <glog/logging.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <mkldnn.hpp>
#include "layer.h"
#include "layer_factory.h"

template <typename T>
class Softmax : public Layer<T> {
public:
    Softmax() : first_fwd(true) {}
    Softmax(int* dims, int axis) : first_fwd(true) {}

    void update_user_mem(T* x, T* y);
    void update_user_data(std::shared_ptr<mkldnn::memory> src_mem,
                          std::shared_ptr<mkldnn::memory> dst_mem);
    bool is_first_fwd(void) { return first_fwd; };
    void mark_first_fwd(void) { first_fwd = false; };

    static Softmax<T>* softmax_create_forward(T* x, int dummy_x,
                                              T* y, int dummy_y,
                                              int* dims, int ndim, int axis);

    virtual int get_res_size() { LOG(INFO) << "Softmax donot implement get_res_size"; return -1; /* Implement in instance */ }
    virtual int forward() { LOG(INFO) << "Softmax donot implement forward"; return -1; /* Implement in instance */ }
    virtual int backward() { LOG(INFO) << "Softmax donot implement backward"; return -1; /* Implement in instance */ }
    virtual int setup_forward() { LOG(INFO) << "Softmax donot implement setup_forward"; return -1; /* Implement in instance */ }
    virtual int setup_backward() { LOG(INFO) << "Softmax donot implement setup_backward"; return -1; /* Implement in instance */ }

protected:
    T*      src_user;    // user input memory of current function updated by every softmax_create_forward
    T*      dst_user;    // user output memory of current function updated by every softmax_create_forward
    bool    first_fwd;

private:
    // Map stream <-> inst
    static std::unordered_map<std::string, void*> map;
};

template <typename T>
class Softmax_2D : public Softmax<T> {
public:
    Softmax_2D(int* dims, int axis) :
               dims(dims),
               axis(axis)
               /* src(NULL), */
               /* dst(NULL)  */ {}

    int get_res_size();
    int forward();
    int backward();
    int setup_forward();
    int setup_backward();

private:
    // Instance shape/identity
    int* dims;        // input/output dimension of all functions initialized by constructor
    int  axis;        // softmax base axis of all functions initialized by constructor

    // Resources
    // T* src;        // Persistent source memory linked to primitive
    // T* dst;        // Persistent destination memory linked to primitive
    std::shared_ptr<mkldnn::memory>                         src_mem;
    std::shared_ptr<mkldnn::memory>                         dst_mem;
    std::shared_ptr<mkldnn::memory::desc>                   src_md;
    std::shared_ptr<mkldnn::primitive>                      fwd;
    std::shared_ptr<mkldnn::softmax_forward::desc>          fwd_desc;
    std::shared_ptr<mkldnn::softmax_forward::primitive_desc> fwd_pd;
    std::shared_ptr<mkldnn::stream>                         fwd_stream_;
    std::vector<mkldnn::primitive>                          fwd_primitives_;
};

template <typename T>
class Softmax_4D : public Softmax<T> {
public:
    Softmax_4D(int* dims, int axis) :
               dims(dims),
               axis(axis)
               /* src(NULL), */
               /* dst(NULL)  */ {}

    int get_res_size();
    int forward();
    int backward();
    int setup_forward();
    int setup_backward();

private:
    // Instance shape/identity
    int* dims;        // input/output dimension of all functions initialized by constructor
    int  axis;        // softmax base axis of all functions initialized by constructor

    // Resources
    // T* src;        // Persistent source memory linked to primitive
    // T* dst;        // Persistent destination memory linked to primitive
    std::shared_ptr<mkldnn::memory>                         src_mem;
    std::shared_ptr<mkldnn::memory>                         dst_mem;
    std::shared_ptr<mkldnn::memory::desc>                   src_md;
    std::shared_ptr<mkldnn::primitive>                      softmax;
    std::shared_ptr<mkldnn::softmax_forward::desc>          softmax_desc;
    std::shared_ptr<mkldnn::softmax_forward::primitive_desc> softmax_pd;
    std::shared_ptr<mkldnn::stream>                         stream_;
    std::vector<mkldnn::primitive>                          primitives_;
};

#endif // _SOFTMAX_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
