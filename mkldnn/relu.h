#pragma once

#include <mkldnn.hpp>
#include <vector>
#include "layer.h"
#include "layer_factory.h"

template <typename T>
class Relu : public Layer<T>{
public:
    Relu();
    int test_buf(
                     T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                     T* y, int y_size
                    );
    int forward_setup(T* x, int x_size,
                      T* y, int y_size);
    void fwd_reset_mem(T* x,
                       T* y);
    int forward(T* x, int x_size,
                T* y, int y_size);

    int backward_setup(T* x, int x_size,
                 T* gy, int gy_size,
                 T* gx, int gx_size);
    void bwd_reset_mem(T* x,
                       T* gy,
                       T* gx);
    int backward(T* x, int x_size,
                 T* gy, int gy_size,
                 T* gx, int gx_size);

    static Relu<T>* get_forward_object(int x_d1) {
        Relu<T>* relu_forward = NULL;
        relu_forward = dynamic_cast<Relu<T>*>(
                LayerFactory::getInstance().getRELULayer(x_d1));
        if (relu_forward == NULL) {
            relu_forward = new Relu<T>();
            LOG(INFO) << "new relu obj " << relu_forward << " dim " << x_d1;
            LayerFactory::getInstance().setRELULayer(
                    x_d1, relu_forward);
        }
        return relu_forward;
    }

    static Relu<T>* get_backward_object(int x_d1) {
        Relu<T>* relu_backward = NULL;
            relu_backward = dynamic_cast<Relu<T>*>(
                                LayerFactory::getInstance().getRELULayer(x_d1));
        assert (relu_backward != NULL);  // we must have already done forward
                                            // before
        return relu_backward;
    }

    static void do_forward(
                T* x,  int x_d1,
                T* y,  int y_d1) {
        Relu<T> *forward_object = get_forward_object(x_d1);
        forward_object->forward(x,  x_d1,
                                y,  y_d1);
    }

    static void do_backward(T* x, int x_d1,
                 T* gy, int gy_d1,
                 T* gx, int gx_d1) {
        Relu<T> *backward_object = get_backward_object(x_d1);
        backward_object->backward(x, x_d1,
                       gy, gy_d1,
                       gx, gx_d1);
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
};
