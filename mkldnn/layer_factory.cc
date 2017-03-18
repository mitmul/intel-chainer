#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "layer_factory.h"

// helper functions to convert layer unique data to a string
static std::string pointer_to_string(void* ptr)
{
    std::ostringstream os;
    os << std::hex << static_cast<void*>(ptr) << "_";
    return os.str();
}

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

#define RELU_FWD_PREFIX "relu_fwd_"
#define RELU_BWD_PREFIX "relu_bwd_"

Layer<float>* LayerFactory::getRELUFwdLayer(void* input, void* output)
{
    std::string key = RELU_FWD_PREFIX;

    key += pointer_to_string(input);
    key += pointer_to_string(output);
    return getLayer(key);
}

void LayerFactory::setRELUFwdLayer(void* input, void* output, Layer<float>*   layer)
{
    std::string key = RELU_FWD_PREFIX;

    key += pointer_to_string(input);
    key += pointer_to_string(output);
    setLayer(key, layer);
}

Layer<float>* LayerFactory::getRELUBwdLayer(
        void* input, void* output_diff, void* input_diff)
{
    std::string key = RELU_BWD_PREFIX;

    key += pointer_to_string(input);
    key += pointer_to_string(output_diff);
    key += pointer_to_string(input_diff);
    return getLayer(key);
}

void LayerFactory::setRELUBwdLayer(
        void* input, void* output_diff, void* input_diff,
        Layer<float>* layer)
{
    std::string key = RELU_BWD_PREFIX;

    key += pointer_to_string(input);
    key += pointer_to_string(output_diff);
    key += pointer_to_string(input_diff);
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

#define AVG_POOLING_FWD_PREFIX "avgpool_fwd_"
#define AVG_POOLING_BWD_PREFIX "avgpool_bwd_"
Layer<float>* LayerFactory::getAvgPoolFwdLayer(
        void* input, void* output,
        int stride_y, int stride_x,
        int ksize_h, int ksize_w,
        int pad_l_h, int pad_l_w,
        int pad_r_h, int pad_r_w)
{
    std::string key = AVG_POOLING_FWD_PREFIX;

    key += pointer_to_string(input);
    key += pointer_to_string(output);
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

void LayerFactory::setAvgPoolFwdLayer(
        void* input, void* output,
        int stride_y, int stride_x,
        int ksize_h, int ksize_w,
        int pad_l_h, int pad_l_w,
        int pad_r_h, int pad_r_w,
        Layer<float>* layer)
{
    std::string key = AVG_POOLING_FWD_PREFIX;

    key += pointer_to_string(input);
    key += pointer_to_string(output);
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

Layer<float>* LayerFactory::getAvgPoolBwdLayer(
        void* input_diff, void* output_diff, void* workspace,
        int stride_y, int stride_x,
        int ksize_h, int ksize_w,
        int pad_l_h, int pad_l_w,
        int pad_r_h, int pad_r_w)
{
    std::string key = AVG_POOLING_BWD_PREFIX;

    key += pointer_to_string(input_diff);
    key += pointer_to_string(output_diff);
    key += pointer_to_string(workspace);
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

void LayerFactory::setAvgPoolBwdLayer(
        void* input_diff, void* output_diff, void* workspace,
        int stride_y, int stride_x,
        int ksize_h, int ksize_w,
        int pad_l_h, int pad_l_w,
        int pad_r_h, int pad_r_w,
        Layer<float>*   layer)
{
    std::string key = AVG_POOLING_BWD_PREFIX;

    key += pointer_to_string(input_diff);
    key += pointer_to_string(output_diff);
    key += pointer_to_string(workspace);
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

#define LRN_FWD_PREFIX "lrn_fwd_"
#define LRN_BWD_PREFIX "lrn_bwd_"
Layer<float>* LayerFactory::getLRNFwdLayer(void*              input,
                                             void*              output,
                                             int                local_size,
                                             float              alpha,
                                             float              beta)
{
    std::string key = LRN_FWD_PREFIX;

    key += pointer_to_string(input);
    key += pointer_to_string(output);
    key += int_to_string(local_size);
    key += float_to_string(alpha);
    key += float_to_string(beta);

    return getLayer(key);
}

void LayerFactory::setLRNFwdLayer(void*              input,
                                    void*              output,
                                    int                local_size,
                                    float              alpha,
                                    float              beta,
                                    Layer<float>*      layer)
{
    std::string key = LRN_FWD_PREFIX;

    key += pointer_to_string(input);
    key += pointer_to_string(output);
    key += int_to_string(local_size);
    key += float_to_string(alpha);
    key += float_to_string(beta);

    setLayer(key, layer);
}

Layer<float>* LayerFactory::getLRNBwdLayer(void*              input_diff,
                                               void*              output_diff,
                                               int                local_size,
                                               float              alpha,
                                               float              beta)
{
    std::string key = LRN_BWD_PREFIX;

    key += pointer_to_string(input_diff);
    key += pointer_to_string(output_diff);
    key += int_to_string(local_size);
    key += float_to_string(alpha);
    key += float_to_string(beta);

    return getLayer(key);
}

void LayerFactory::setLRNBwdLayer(void*              input_diff,
                                    void*              output_diff,
                                    int                local_size,
                                    float              alpha,
                                    float              beta,
                                    Layer<float>*      layer)
{
    std::string key = LRN_BWD_PREFIX;

    key += pointer_to_string(input_diff);
    key += pointer_to_string(output_diff);
    key += int_to_string(local_size);
    key += float_to_string(alpha);
    key += float_to_string(beta);

    setLayer(key, layer);
}

#define SOFTMAX2D_FWD_PREFIX "softmax2d_fwd_"
Layer<float>* LayerFactory::getSoftmax2DFwdLayer(int                d1,
                                                   int                d2,
                                                   int                axis)
{
    std::string key = SOFTMAX2D_FWD_PREFIX;

    key += int_to_string(d1);
    key += int_to_string(d2);
    key += int_to_string(axis);

    return getLayer(key);
}

void LayerFactory::setSoftmax2DFwdLayer(int                d1,
                                          int                d2,
                                          int                axis,
                                          Layer<float>*      layer)
{
    std::string key = SOFTMAX2D_FWD_PREFIX;

    key += int_to_string(d1);
    key += int_to_string(d2);
    key += int_to_string(axis);

    setLayer(key, layer);
}

#define SOFTMAX4D_FWD_PREFIX "softmax4d_fwd_"
Layer<float>* LayerFactory::getSoftmax4DFwdLayer(int                d1,
                                                   int                d2,
                                                   int                d3,
                                                   int                d4,
                                                   int                axis)
{
    std::string key = SOFTMAX4D_FWD_PREFIX;

    key += int_to_string(d1);
    key += int_to_string(d2);
    key += int_to_string(d3);
    key += int_to_string(d4);
    key += int_to_string(axis);

    return getLayer(key);
}

void LayerFactory::setSoftmax4DFwdLayer(int                d1,
                                          int                d2,
                                          int                d3,
                                          int                d4,
                                          int                axis,
                                          Layer<float>*      layer)
{
    std::string key = SOFTMAX4D_FWD_PREFIX;

    key += int_to_string(d1);
    key += int_to_string(d2);
    key += int_to_string(d3);
    key += int_to_string(d4);
    key += int_to_string(axis);

    setLayer(key, layer);
}
