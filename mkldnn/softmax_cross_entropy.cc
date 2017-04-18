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
#include <vector>
#include "softmax_cross_entropy.h"
#include "softmax.h"

template<typename T> SoftmaxCrossEntropy<T>*
SoftmaxCrossEntropy<T>::softmax_cross_entropy_create_forward(int* dims, int ndim)
{
    static SoftmaxCrossEntropy_2D<T> inst_2d;
    static SoftmaxCrossEntropy_4D<T> inst_4d;
    SoftmaxCrossEntropy<T>* inst = NULL;

#define SOFTMAX_CROSS_ENTROPY_2D 2
#define SOFTMAX_CROSS_ENTROPY_4D 4
    if (SOFTMAX_CROSS_ENTROPY_2D == ndim) {
        inst = &inst_2d;
    } else if (SOFTMAX_CROSS_ENTROPY_4D == ndim) {
        inst = &inst_4d;
    } else {
        ; //Not supported;
    }

    return inst;
}

template<typename T> SoftmaxCrossEntropy<T>*
SoftmaxCrossEntropy<T>::softmax_cross_entropy_create_backward(int* dims, int ndim)
{
    return softmax_cross_entropy_create_forward(dims, ndim);
}

template<typename T>
int SoftmaxCrossEntropy_2D<T>::forward(T* x, int dummy_x,
                                       T* y, int dummy_y,
                                       int* dims, int ndim)
{
    // Softmax mkldnn optimization
    Softmax<T>* softmax = Softmax<T>::softmax_create_forward(x, dummy_x,
                                                             y, dummy_y,
                                                             dims, ndim, 1);
    softmax->forward();

    // log(F_Softmax)
    int n, c;
    float* cur;

    for (n = 0; n < dims[0]; n++) {
        for (c = 0; c < dims[1]; c++) {
            cur = y + n * dims[1] + c;
            *cur = logf(*cur);
        }
    }

    return 0;
}

template<typename T>
int SoftmaxCrossEntropy_2D<T>::backward(T* gx, int dummy_gx,
                                        int* label, int nlabel,
                                        int* dims, int ndim)
{
    int n, c;

    for (n = 0; n < dims[0]; n++) {
        if (label[n] >= 0 && label[n] < dims[1]) {
            c = label[n];
            *(gx + n * dims[1] + c) -= 1;
        }
    }

    return 0;
}

template<typename T>
int SoftmaxCrossEntropy_4D<T>::forward(T* x, int dummy_x,
                                       T* y, int dummy_y,
                       int* dims, int ndim)
{
    return 0;
}

template<typename T>
int SoftmaxCrossEntropy_4D<T>::backward(T* gx, int dummy_gx,
                                        int* label, int nlabel,
                    int* dims, int ndim)
{
    return 0;
}

template class SoftmaxCrossEntropy<float>;
template class SoftmaxCrossEntropy_2D<float>;
template class SoftmaxCrossEntropy_4D<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
