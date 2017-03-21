#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "layer_factory.h"

// helper functions to convert layer unique data to a string
#if 1
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
    os << std::hex << value << "_";
    return os.str();
}

static std::string float_to_string(float value)
{
    std::ostringstream os;
    os << value << "_";
    return os.str();
}
// end of helper functions

using namespace mkldnn;
LayerFactory::LayerFactory()
{
}

Layer<float>* LayerFactory::getLayer(std::string key)
{
    auto stream_iter = map.find(key);
    if (stream_iter == map.end()) {
        return NULL;
    } else {
        return stream_iter->second;
    }
}

void LayerFactory::setLayer(std::string key, Layer<float>* layer)
{
    auto stream_iter = map.find(key);
    if (stream_iter == map.end()) {
        map[key]=layer;
    } else {
        throw new std::invalid_argument("cannot set same key to a new stream");
    }
}

#define RELU_PREFIX "relu_"

Layer<float>* LayerFactory::getRELULayer(int size)
{
    std::string key = RELU_PREFIX;

    key += int_to_string(size);
    return getLayer(key);
}

void LayerFactory::setRELULayer(int size, Layer<float>*   layer)
{
    std::string key = RELU_PREFIX;

    key += int_to_string(size);
    setLayer(key, layer);
}

#define RELU4D_PREFIX "relu4d_"

Layer<float>* LayerFactory::getRELU4dLayer(
        int x_d1, int x_d2, int x_d3, int x_d4)
{
    std::string key = RELU4D_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);

    return getLayer(key);
}

void LayerFactory::setRELU4dLayer(
        int x_d1, int x_d2, int x_d3, int x_d4,
        Layer<float>* layer)
{
    std::string key = RELU4D_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);

    setLayer(key, layer);
}

#define MAX_POOLING_PREFIX "maxpool_"
Layer<float>* LayerFactory::getMaxPoolLayer(
        int x_d1, int x_d2, int x_d3, int x_d4,
        int stride_y, int stride_x,
        int ksize_h, int ksize_w,
        int pad_l_h, int pad_l_w,
        int pad_r_h, int pad_r_w)
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
    key += int_to_string(pad_l_h);
    key += int_to_string(pad_l_w);
    key += int_to_string(pad_r_h);
    key += int_to_string(pad_r_w);

    return getLayer(key);
}

void LayerFactory::setMaxPoolLayer(
        int x_d1, int x_d2, int x_d3, int x_d4,
        int stride_y, int stride_x,
        int ksize_h, int ksize_w,
        int pad_l_h, int pad_l_w,
        int pad_r_h, int pad_r_w,
        Layer<float>* layer)
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
    key += int_to_string(pad_l_h);
    key += int_to_string(pad_l_w);
    key += int_to_string(pad_r_h);
    key += int_to_string(pad_r_w);

    setLayer(key, layer);
}

#define AVG_POOLING_PREFIX "avgpool_"
Layer<float>* LayerFactory::getAvgPoolLayer(
        int x_d1, int x_d2, int x_d3, int x_d4,
        int stride_y, int stride_x,
        int ksize_h, int ksize_w,
        int pad_l_h, int pad_l_w,
        int pad_r_h, int pad_r_w)
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
    key += int_to_string(pad_l_h);
    key += int_to_string(pad_l_w);
    key += int_to_string(pad_r_h);
    key += int_to_string(pad_r_w);

    return getLayer(key);
}

void LayerFactory::setAvgPoolLayer(
        int x_d1, int x_d2, int x_d3, int x_d4,
        int stride_y, int stride_x,
        int ksize_h, int ksize_w,
        int pad_l_h, int pad_l_w,
        int pad_r_h, int pad_r_w,
        Layer<float>* layer)
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
    key += int_to_string(pad_l_h);
    key += int_to_string(pad_l_w);
    key += int_to_string(pad_r_h);
    key += int_to_string(pad_r_w);

    setLayer(key, layer);
}

#define LRN_PREFIX "lrn_"
Layer<float>* LayerFactory::getLRNLayer(int             x_d1,
                                        int             x_d2,
                                        int             x_d3,
                                        int             x_d4,
                                        int             local_size,
                                        float           alpha,
                                        float           beta)
{
    std::string key = LRN_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);
    key += int_to_string(local_size);
    key += float_to_string(alpha);
    key += float_to_string(beta);

    return getLayer(key);
}

void LayerFactory::setLRNLayer(int              x_d1,
                               int              x_d2,
                               int              x_d3,
                               int              x_d4,
                               int              local_size,
                               float            alpha,
                               float            beta,
                               Layer<float>*    layer)
{
    std::string key = LRN_PREFIX;

    key += int_to_string(x_d1);
    key += int_to_string(x_d2);
    key += int_to_string(x_d3);
    key += int_to_string(x_d4);
    key += int_to_string(local_size);
    key += float_to_string(alpha);
    key += float_to_string(beta);

    setLayer(key, layer);
}

#define SOFTMAX2D_PREFIX "softmax2d_"
Layer<float>* LayerFactory::getSoftmax2DLayer(int                d1,
                                              int                d2,
                                              int                axis)
{
    std::string key = SOFTMAX2D_PREFIX;

    key += int_to_string(d1);
    key += int_to_string(d2);
    key += int_to_string(axis);

    return getLayer(key);
}

void LayerFactory::setSoftmax2DLayer(int                d1,
                                     int                d2,
                                     int                axis,
                                     Layer<float>*      layer)
{
    std::string key = SOFTMAX2D_PREFIX;

    key += int_to_string(d1);
    key += int_to_string(d2);
    key += int_to_string(axis);

    setLayer(key, layer);
}

#define SOFTMAX4D_PREFIX "softmax4d_"
Layer<float>* LayerFactory::getSoftmax4DLayer(int                d1,
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

    return getLayer(key);
}

void LayerFactory::setSoftmax4DLayer(int                d1,
                                     int                d2,
                                     int                d3,
                                     int                d4,
                                     int                axis,
                                     Layer<float>*      layer)
{
    std::string key = SOFTMAX4D_PREFIX;

    key += int_to_string(d1);
    key += int_to_string(d2);
    key += int_to_string(d3);
    key += int_to_string(d4);
    key += int_to_string(axis);

    setLayer(key, layer);
}
