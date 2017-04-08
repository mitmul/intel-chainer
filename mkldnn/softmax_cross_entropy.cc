#include <glog/logging.h>
#include <vector>
#include "softmax_cross_entropy.h"
#include "mkldnn_softmax.h"

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