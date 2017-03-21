#pragma once

#include <mkldnn.hpp>
#include <vector>
#include "layer.h"
#include "layer_factory.h"

template <typename T>
class Relu4D : public Layer<T>{
public:
    Relu4D();
    int forward_setup(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                      T* y, int y_d1, int y_d2, int y_d3, int y_d4);
    void fwd_reset_mem(T* x,
                       T* y);
    int forward(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                T* y, int y_d1, int y_d2, int y_d3, int y_d4);

    int backward_setup(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                       T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
                       T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4);
    void bwd_reset_mem(T* x,
                       T* gy,
                       T* gx);
    int backward(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                 T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
                 T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4);

    static Relu4D<T>* get_forward_object(
            int x_d1, int x_d2, int x_d3, int x_d4) {
        Relu4D<T>* relu4d_forward = NULL;
        relu4d_forward = dynamic_cast<Relu4D<T>*>(
                LayerFactory::getInstance().getRELU4dLayer
                (x_d1, x_d2, x_d3, x_d4));
        if (relu4d_forward == NULL) {
            relu4d_forward = new Relu4D<T>();
            LOG(INFO) << "new relu4d obj " << relu4d_forward << " dmin " << x_d1 << " : "
                << x_d2 << " : " << x_d3 << " : " << x_d4;
#if 0
            relu4d_forward->forward_setup(x, x_d1, x_d2, x_d3, x_d4,
                    y, x_d1, x_d2, x_d3, x_d4);
#endif
            LayerFactory::getInstance().setRELU4dLayer(
                    x_d1, x_d2, x_d3, x_d4, relu4d_forward);
        }
        return relu4d_forward;
    }

    static Relu4D<T>* get_backward_object(
                      int x_d1, int x_d2, int x_d3, int x_d4) {
        Relu4D<T>* relu4d_backward = NULL;
            relu4d_backward = dynamic_cast<Relu4D<T>*>(
                                LayerFactory::getInstance().getRELU4dLayer
                                (x_d1, x_d2, x_d3, x_d4));
        assert (relu4d_backward != NULL);  // we must have already done forward
                                            // before
        if (relu4d_backward->backward_first_setup_ == true) {
#if 0
            relu4d_backward->backward_setup(x, x_d1, x_d2, x_d3, x_d4,
                    gy, x_d1, x_d2, x_d3, x_d4,
                    gx, x_d1, x_d2, x_d3, x_d4);
            relu4d_backward->backward_first_setup_ = false;
#endif
        }
        return relu4d_backward;
    }

    static void do_forward(
                T* x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
                T* y,  int y_d1,  int y_d2,  int y_d3,  int y_d4) {
        Relu4D<T> *forward_object = get_forward_object(x_d1, x_d2, x_d3, x_d4);
        forward_object->forward(x,  x_d1,  x_d2,  x_d3,  x_d4,
                                y,  y_d1,  y_d2,  y_d3,  y_d4);
    }

    static void do_backward(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                 T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
                 T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4) {
        Relu4D<T> *backward_object = get_backward_object(
                                        x_d1, x_d2, x_d3, x_d4);
        backward_object->backward(x, x_d1, x_d2, x_d3, x_d4,
                       gy, gy_d1, gy_d2, gy_d3, gy_d4,
                       gx, gx_d1, gx_d2, gx_d3, gx_d4);
    }
private:
    //forward
    std::shared_ptr<mkldnn::memory> relu_fwd_user_src_mem_, relu_fwd_dst_mem_;
    std::shared_ptr<mkldnn::memory::desc> relu_fwd_src_md_;
    std::shared_ptr<mkldnn::relu_forward::desc> relu_fwd_desc_;
    std::shared_ptr<mkldnn::relu_forward::primitive_desc> relu_fwd_pd_;
    std::shared_ptr<mkldnn::relu_forward> relu_fwd_;
    std::shared_ptr<mkldnn::stream> fwd_stream_;
    std::vector<mkldnn::primitive> fwd_primitives_;

    //backward
    std::shared_ptr<mkldnn::memory> relu_diff_src_mem_, relu_diff_dst_mem_;
    std::shared_ptr<mkldnn::memory::desc> relu_diff_dst_md_;
    std::shared_ptr<mkldnn::relu_backward::desc> relu_bwd_desc_;
    std::shared_ptr<mkldnn::relu_backward::primitive_desc> relu_bwd_pd_;
    std::shared_ptr<mkldnn::relu_backward> relu_bwd_;

    std::shared_ptr<mkldnn::stream> bwd_stream_;
    std::vector<mkldnn::primitive> bwd_primitives_;
    //bool backward_first_setup_;
};
