#ifndef _SOFTMAX_CROSS_ENTROPY_H_
#define _SOFTMAX_CROSS_ENTROPY_H_

#include <glog/logging.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "softmax.h"
#include "layer.h"

template <typename T>
class SoftmaxCrossEntropy : public Layer<T> {
public:
    SoftmaxCrossEntropy() {}

    static SoftmaxCrossEntropy<T>* softmax_cross_entropy_create_forward(int* dims, int ndim);
    static SoftmaxCrossEntropy<T>* softmax_cross_entropy_create_backward(int* dims, int ndim);

    virtual int forward(T* x, int dummy_x,
                        T* y, int dummy_y,
                        int* dims, int ndim)
    { LOG(INFO) << "Softmax donot implement forward"; return -1; /* Implement in instance */ }
    virtual int backward(T* gx, int dummy_gx,
                         int* label, int nlabel,
                         int* dims, int ndim)
    { LOG(INFO) << "Softmax donot implement backward"; return -1; /* Implement in instance */ }
};

template <typename T>
class SoftmaxCrossEntropy_2D : public SoftmaxCrossEntropy<T> {
public:
    SoftmaxCrossEntropy_2D() {}

    int forward(T* x, int dummy_x,
                T* y, int dummy_y,
                int* dims, int ndim);
    int backward(T* gx, int dummy_gx,
                 int* label, int nlabel,
                 int* dims, int ndim);
};

template <typename T>
class SoftmaxCrossEntropy_4D : public SoftmaxCrossEntropy<T> {
public:
    SoftmaxCrossEntropy_4D() {}

    int forward(T* x, int dummy_x,
                T* y, int dummy_y,
                int* dims, int ndim);
    int backward(T* gx, int dummy_gx,
                 int* label, int nlabel,
                 int* dims, int ndim);
};

#endif // _SOFTMAX_CROSS_ENTROPY_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
