#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "layer_factory.h"

// helper functions to convert layer unique data to a string
#if 0
static std::string pointer_to_string(void* ptr)
{
    std::ostringstream os;
    os << std::hex << static_cast<void*>(ptr) << "_";
    return os.str();
}
#endif

static std::string int_to_string(int value)
{
    std::ostringstream os;
    os << std::hex << "I" << value << "_";
    return os.str();
}

static std::string float_to_string(float value)
{
    std::ostringstream os;
    os << "F" << value << "_";
    return os.str();
}

static std::string double_to_string(double value)
{
    std::ostringstream os;
    os << "D" << value << "_";
    return os.str();
}
// end of helper functions

using namespace mkldnn;
template<typename T>
LayerFactory<T>::LayerFactory()
{
}

template<typename T>
Layer<T>* LayerFactory<T>::get_layer(std::string key)
{
    auto stream_iter = map_.find(key);
    if (stream_iter == map_.end()) {
        return NULL;
    } else {
        return stream_iter->second;
    }
}

template<typename T>
void LayerFactory<T>::set_layer(std::string key, Layer<T>* layer)
{
    auto stream_iter = map_.find(key);
    if (stream_iter == map_.end()) {
        map_[key]=layer;
    } else {
        throw new std::invalid_argument("cannot set same key to a new stream");
    }
}

#define RELU_PREFIX "relu_"

template<typename T>
Layer<T>* LayerFactory<T>::get_relu_layer(int size)
{
    std::string key = RELU_PREFIX;

    key += int_to_string(size);
    return get_layer(key);
}

template<typename T>
void LayerFactory<T>::set_relu_layer(int size, Layer<T>*   layer)
{
    std::string key = RELU_PREFIX;

    key += int_to_string(size);
    set_layer(key, layer);
}

#define RELU4D_PREFIX "relu4d_"

template<typename T>
Layer<T>* LayerFactory<T>::get_relu4d_layer(
          int x_d1, int x_d2, int x_d3, int x_d4)
{
    std::string key = RELU4D_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);

    return get_layer(key);
}

template<typename T>
void LayerFactory<T>::set_relu4d_layer(
        int x_d1, int x_d2, int x_d3, int x_d4,
        Layer<T>* layer)
{
    std::string key = RELU4D_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);

    set_layer(key, layer);
}

#define MAX_POOLING_PREFIX "maxpool_"
template<typename T>
Layer<T>* LayerFactory<T>::get_max_pool_layer(
          int x_d1, int x_d2, int x_d3, int x_d4,
          int stride_y, int stride_x,
          int ksize_h,  int ksize_w,
          int pad_u,    int pad_d,
          int pad_l,    int pad_r)
{
    std::string key = MAX_POOLING_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);
    key += int_to_string(stride_y);
    key += int_to_string(stride_x);
    key += int_to_string(ksize_h);
    key += int_to_string(ksize_w);
    key += int_to_string(pad_u);
    key += int_to_string(pad_d);
    key += int_to_string(pad_l);
    key += int_to_string(pad_r);

    return get_layer(key);
}

template<typename T>
void LayerFactory<T>::set_max_pool_layer(
        int x_d1, int x_d2, int x_d3, int x_d4,
        int stride_y, int stride_x,
        int ksize_h,  int ksize_w,
        int pad_u,    int pad_d,
        int pad_l,    int pad_r,
        Layer<T>* layer)
{
    std::string key = MAX_POOLING_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);
    key += int_to_string(stride_y);
    key += int_to_string(stride_x);
    key += int_to_string(ksize_h);
    key += int_to_string(ksize_w);
    key += int_to_string(pad_u);
    key += int_to_string(pad_d);
    key += int_to_string(pad_l);
    key += int_to_string(pad_r);

    set_layer(key, layer);
}

#define AVG_POOLING_PREFIX "avgpool_"
template<typename T>
Layer<T>* LayerFactory<T>::get_avg_pool_layer(
          int x_d1, int x_d2, int x_d3, int x_d4,
          int stride_y, int stride_x,
          int ksize_h,  int ksize_w,
          int pad_u,    int pad_d,
          int pad_l,    int pad_r)
{
    std::string key = AVG_POOLING_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);
    key += int_to_string(stride_y);
    key += int_to_string(stride_x);
    key += int_to_string(ksize_h);
    key += int_to_string(ksize_w);
    key += int_to_string(pad_u);
    key += int_to_string(pad_d);
    key += int_to_string(pad_l);
    key += int_to_string(pad_r);

    return get_layer(key);
}

template<typename T>
void LayerFactory<T>::set_avg_pool_layer(
        int x_d1, int x_d2, int x_d3, int x_d4,
        int stride_y, int stride_x,
        int ksize_h,  int ksize_w,
        int pad_u,    int pad_d,
        int pad_l,    int pad_r,
        Layer<T>* layer)
{
    std::string key = AVG_POOLING_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);
    key += int_to_string(stride_y);
    key += int_to_string(stride_x);
    key += int_to_string(ksize_h);
    key += int_to_string(ksize_w);
    key += int_to_string(pad_u);
    key += int_to_string(pad_d);
    key += int_to_string(pad_l);
    key += int_to_string(pad_r);

    set_layer(key, layer);
}

#define LRN_PREFIX "lrn_"
template<typename T>
Layer<T>* LayerFactory<T>::get_lrn_layer(int             x_d1,
                                         int             x_d2,
                                         int             x_d3,
                                         int             x_d4,
                                         int             local_size,
                                         double           k,
                                         double           alpha,
                                         double           beta)
{
    std::string key = LRN_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);
    key += int_to_string(local_size);
    key += double_to_string(k);
    key += double_to_string(alpha);
    key += double_to_string(beta);

    return get_layer(key);
}

template<typename T>
void LayerFactory<T>::set_lrn_layer(int              x_d1,
                                    int              x_d2,
                                    int              x_d3,
                                    int              x_d4,
                                    int              local_size,
                                    double            k,
                                    double            alpha,
                                    double            beta,
                                    Layer<T>*    layer)
{
    std::string key = LRN_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);
    key += int_to_string(local_size);
    key += double_to_string(k);
    key += double_to_string(alpha);
    key += double_to_string(beta);

    set_layer(key, layer);
}

#define SOFTMAX2D_PREFIX "softmax2d_"
template<typename T>
Layer<T>* LayerFactory<T>::get_softmax2d_layer(int                d1,
                                               int                d2,
                                               int                axis)
{
    std::string key = SOFTMAX2D_PREFIX;

    key += int_to_string(d1);
    key += int_to_string(d2);
    key += int_to_string(axis);

    return get_layer(key);
}

template<typename T>
void LayerFactory<T>::set_softmax2d_layer(int                d1,
                                          int                d2,
                                          int                axis,
                                          Layer<T>*      layer)
{
    std::string key = SOFTMAX2D_PREFIX;

    key += int_to_string(d1);
    key += int_to_string(d2);
    key += int_to_string(axis);

    set_layer(key, layer);
}

#define SOFTMAX4D_PREFIX "softmax4d_"
template<typename T>
Layer<T>* LayerFactory<T>::get_softmax4d_layer(int                d1,
                                               int                d2,
                                               int                d3,
                                               int                d4,
                                               int                axis)
{
    std::string key = SOFTMAX4D_PREFIX;

    key += int_to_string(d1);
    key += int_to_string(d2);
    key += int_to_string(d3);
    key += int_to_string(d4);
    key += int_to_string(axis);

    return get_layer(key);
}

template<typename T>
void LayerFactory<T>::set_softmax4d_layer(int                d1,
                                          int                d2,
                                          int                d3,
                                          int                d4,
                                          int                axis,
                                          Layer<T>*      layer)
{
    std::string key = SOFTMAX4D_PREFIX;

    key += int_to_string(d1);
    key += int_to_string(d2);
    key += int_to_string(d3);
    key += int_to_string(d4);
    key += int_to_string(axis);

    set_layer(key, layer);
}

#define CONVOLUTION2D_PREFIX "conv2d_"
template<typename T>
Layer<T>* LayerFactory<T>::get_conv2d_layer(
          int x_d1, int x_d2, int x_d3, int x_d4,
          int W_d1, int W_d2, int W_d3, int W_d4,
          int b_d1,
          int ksize_h, int ksize_w,
          int stride_y, int stride_x,
          int pad_l_h, int pad_l_w,
          int pad_r_h, int pad_r_w)
{
    std::string key = CONVOLUTION2D_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);
    key += int_to_string(W_d1);
    key += int_to_string(W_d2);
    key += int_to_string(W_d3);
    key += int_to_string(W_d4);
    key += int_to_string(b_d1);
    key += int_to_string(ksize_h);
    key += int_to_string(ksize_w);
    key += int_to_string(stride_y);
    key += int_to_string(stride_x);
    key += int_to_string(pad_l_h);
    key += int_to_string(pad_l_w);
    key += int_to_string(pad_r_h);
    key += int_to_string(pad_r_w);

    return get_layer(key);
}

template<typename T>
void LayerFactory<T>::set_conv2d_layer(
        int x_d1, int x_d2, int x_d3, int x_d4,
        int W_d1, int W_d2, int W_d3, int W_d4,
        int b_d1,
        int ksize_h, int ksize_w,
        int stride_y, int stride_x,
        int pad_l_h, int pad_l_w,
        int pad_r_h, int pad_r_w,
        Layer<T>* layer)
{
    std::string key = CONVOLUTION2D_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);
    key += int_to_string(W_d1);
    key += int_to_string(W_d2);
    key += int_to_string(W_d3);
    key += int_to_string(W_d4);
    key += int_to_string(b_d1);
    key += int_to_string(ksize_h);
    key += int_to_string(ksize_w);
    key += int_to_string(stride_y);
    key += int_to_string(stride_x);
    key += int_to_string(pad_l_h);
    key += int_to_string(pad_l_w);
    key += int_to_string(pad_r_h);
    key += int_to_string(pad_r_w);

    return set_layer(key, layer);
}

#define LINEAR_PREFIX "linear_"
template<typename T>
Layer<T>* LayerFactory<T>::get_linear_layer(
            int x_d1, int x_d2,
            int W_d1, int W_d2,
            int b_d1)
{
    std::string key = LINEAR_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(W_d1);
    key += int_to_string(W_d2);
    key += int_to_string(b_d1);
    return get_layer(key);
}

template<typename T>
void LayerFactory<T>::set_linear_layer(
        int x_d1, int x_d2,
        int W_d1, int W_d2,
        int b_d1,
        Layer<T>* layer)
{
    std::string key = LINEAR_PREFIX;
    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(W_d1);
    key += int_to_string(W_d2);
    key += int_to_string(b_d1);
    return set_layer(key, layer);
}

template class LayerFactory<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
