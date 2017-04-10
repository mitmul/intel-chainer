#ifndef _STREAM_FACTORY_
#define _STREAM_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "layer.h"
#include <unordered_map>

// Usage:
// When stream is created, call:
// LayerFactory::get_instance().setRELUFwdLayer(<input pointer>, <layer>)
// then when forward is needed, call
// layer = LayerFactory::get_instance().getRELUFwdLayer(<input pointer>)

template <typename T>
class LayerFactory {
private:
    LayerFactory();

public:
    static LayerFactory& get_instance() {
        static LayerFactory instance_;
        return instance_;
    }

private:
    Layer<T>* get_layer(std::string      key);
    void      set_layer(std::string      key,
                        Layer<T>*        layer);

public:
    // relu stream
    Layer<T>* get_relu_layer(int          size);
    void      set_relu_layer(int          size,
                             Layer<T>*    layer);
    // relu4d stream
    Layer<T>* get_relu4d_layer(int        x_d1,
                               int        x_d2,
                               int        x_d3,
                               int        x_d4);
    void      set_relu4d_layer(int        x_d1,
                               int        x_d2,
                               int        x_d3,
                               int        x_d4,
                               Layer<T>*  layer);

    // maxpool stream
    Layer<T>* get_max_pool_layer(int       x_d1,
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

    void      set_max_pool_layer(int       x_d1,
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
    Layer<T>* get_avg_pool_layer(int       x_d1,
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

    void      set_avg_pool_layer(int       x_d1,
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
    Layer<T>* get_lrn_layer(int               x_d1,
                            int               x_d2,
                            int               x_d3,
                            int               x_d4,
                            int               local_size,
                            double             k,
                            double             alpha,
                            double             beta);
    void          set_lrn_layer(int               x_d1,
                                int               x_d2,
                                int               x_d3,
                                int               x_d4,
                                int               local_size,
                                double             k,
                                double             alpha,
                                double             beta,
                                Layer<T>*     layer);

    // Softmax Cross Entropy stream
    Layer<T>* get_softmax2d_layer(int               d1,
                                  int               d2,
                                  int               axis);
    void      set_softmax2d_layer(int               d1,
                                  int               d2,
                                  int               axis,
                                  Layer<T>*     layer);
    Layer<T>* get_softmax4d_layer(int               d1,
                                  int               d2,
                                  int               d3,
                                  int               d4,
                                  int               axis);
    void      set_softmax4d_layer(int               d1,
                                  int               d2,
                                  int               d3,
                                  int               d4,
                                  int               axis,
                                  Layer<T>*     layer);

    // Convolution2d stream
    Layer<T>* get_conv2d_layer( int           x_d1,
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

    void       set_conv2d_layer(int           x_d1,
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
    Layer<T>* get_linear_layer(int            x_d1,
                               int            x_d2,
                               int            W_d1,
                               int            W_d2,
                               int            b_d1);
    void      set_linear_layer(int            x_d1,
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
    std::unordered_map<std::string, Layer<T>*> map_;
};

#endif // _STREAM_FACTORY_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
