#ifndef _STREAM_FACTORY_
#define _STREAM_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "layer.h"
#include <unordered_map>

// Usage:
// When stream is created, call:
// LayerFactory::getInstance().setRELUFwdLayer(<input pointer>, <layer>)
// then when forward is needed, call
// layer = LayerFactory::getInstance().getRELUFwdLayer(<input pointer>)

template <typename T>
class LayerFactory {
private:
    LayerFactory();

public:
    static LayerFactory& getInstance() {
        static LayerFactory instance;
        return instance;
    }

private:
    Layer<T>* getLayer(std::string      key);
    void      setLayer(std::string      key,
                       Layer<T>*        layer);

public:
    // relu stream
    Layer<T>* getRELULayer(int          size);
    void      setRELULayer(int          size,
                           Layer<T>*    layer);
    // relu4d stream
    Layer<T>* getRELU4dLayer(int        x_d1,
                             int        x_d2,
                             int        x_d3,
                             int        x_d4);
    void      setRELU4dLayer(int        x_d1,
                             int        x_d2,
                             int        x_d3,
                             int        x_d4,
                             Layer<T>*  layer);

    // maxpool stream
    Layer<T>* getMaxPoolLayer(int       x_d1,
                              int       x_d2,
                              int       x_d3,
                              int       x_d4,
                              int       stride_y,
                              int       stride_x,
                              int       ksize_h,
                              int       ksize_w,
                              int       pad_l_h,
                              int       pad_l_w,
                              int       pad_r_h,
                              int       pad_r_w);

    void      setMaxPoolLayer(int       x_d1,
                              int       x_d2,
                              int       x_d3,
                              int       x_d4,
                              int       stride_y,
                              int       stride_x,
                              int       ksize_h,
                              int       ksize_w,
                              int       pad_l_h,
                              int       pad_l_w,
                              int       pad_r_h,
                              int       pad_r_w,
                              Layer<T>* layer);

    // avgpool stream
    Layer<T>* getAvgPoolLayer(int       x_d1,
                              int       x_d2,
                              int       x_d3,
                              int       x_d4,
                              int       stride_y,
                              int       stride_x,
                              int       ksize_h,
                              int       ksize_w,
                              int       pad_l_h,
                              int       pad_l_w,
                              int       pad_r_h,
                              int       pad_r_w);

    void      setAvgPoolLayer(int       x_d1,
                              int       x_d2,
                              int       x_d3,
                              int       x_d4,
                              int       stride_y,
                              int       stride_x,
                              int       ksize_h,
                              int       ksize_w,
                              int       pad_l_h,
                              int       pad_l_w,
                              int       pad_r_h,
                              int       pad_r_w,
                              Layer<T>* layer);

    // Local Response Normalization stream
    // TODO cross channel support
    Layer<T>* getLRNLayer(int               x_d1,
                              int               x_d2,
                              int               x_d3,
                              int               x_d4,
                              int               local_size,
                              double             k,
                              double             alpha,
                              double             beta);
    void          setLRNLayer(int               x_d1,
                              int               x_d2,
                              int               x_d3,
                              int               x_d4,
                              int               local_size,
                              double             k,
                              double             alpha,
                              double             beta,
                              Layer<T>*     layer);

    // Softmax Cross Entropy stream
    Layer<T>* getSoftmax2DLayer(int               d1,
                                int               d2,
                                int               axis);
    void      setSoftmax2DLayer(int               d1,
                                int               d2,
                                int               axis,
                                Layer<T>*     layer);
    Layer<T>* getSoftmax4DLayer(int               d1,
                                int               d2,
                                int               d3,
                                int               d4,
                                int               axis);
    void      setSoftmax4DLayer(int               d1,
                                int               d2,
                                int               d3,
                                int               d4,
                                int               axis,
                                Layer<T>*     layer);

    // Convolution2d stream
    Layer<T>* getConv2dLayer( int           x_d1,
                              int           x_d2,
                              int           x_d3,
                              int           x_d4,
                              int           W_d1,
                              int           W_d2,
                              int           W_d3,
                              int           W_d4,
                              int           b_d1,
                              int           ksize_h,
                              int           ksize_w,
                              int           stride_y,
                              int           stride_x,
                              int           pad_l_h,
                              int           pad_l_w,
                              int           pad_r_h,
                              int           pad_r_w);

    void       setConv2dLayer(int           x_d1,
                              int           x_d2,
                              int           x_d3,
                              int           x_d4,
                              int           W_d1,
                              int           W_d2,
                              int           W_d3,
                              int           W_d4,
                              int           b_d1,
                              int           ksize_h,
                              int           ksize_w,
                              int           stride_y,
                              int           stride_x,
                              int           pad_l_h,
                              int           pad_l_w,
                              int           pad_r_h,
                              int           pad_r_w,
                              Layer<T>*     layer);

    //Linear stream
    Layer<T>* getLinearLayer(int            x_d1,
                             int            x_d2,
                             int            W_d1,
                             int            W_d2,
                             int            b_d1);
    void      setLinearLayer(int            x_d1,
                             int            x_d2,
                             int            W_d1,
                             int            W_d2,
                             int            b_d1,
                             Layer<T>*      layer);

    LayerFactory(LayerFactory const&)  = delete;
    void operator=(LayerFactory const&) = delete;

private:
    //LayerFactory(LayerFactory const&);
    //void operator=(LayerFactory const&);
    std::unordered_map<std::string, Layer<T>*> map;
};

#endif // _STREAM_FACTORY_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s