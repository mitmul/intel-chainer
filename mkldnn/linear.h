#ifndef _LINEAR_H_
#define _LINEAR_H_

#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "layer.h"
#include "layer_factory.h"
#include <glog/logging.h>
template <typename T>
class MKLDNNLinear:public Layer<T> {
private:
    static MKLDNNLinear<T>* get_forward_object(
                                            T* x, int x_d1, int x_d2,
                                            T* W, int W_d1, int W_d2,
                                            T* b, int b_d1)
    {
        MKLDNNLinear<T>* linear_forward = NULL;
        linear_forward = dynamic_cast<MKLDNNLinear<T>*>(
                            LayerFactory<T>::getInstance().getLinearLayer(
                                x_d1, x_d2,
                                W_d1, W_d2,
                                b_d1));
        if (linear_forward == NULL) {
            linear_forward = new MKLDNNLinear();
            LayerFactory<T>::getInstance().setLinearLayer(
                                x_d1, x_d2,
                                W_d1, W_d2,
                                b_d1,
                                linear_forward);
        }
        return linear_forward;
    }   

    static MKLDNNLinear<T>* get_backward_object(
                                            T* x, int x_d1, int x_d2,
                                            T* W, int W_d1, int W_d2,
                                            T* b, int b_d1)
    {
        MKLDNNLinear<T>* linear_backward;
        linear_backward = dynamic_cast<MKLDNNLinear<T>*>(
                            LayerFactory<T>::getInstance().getLinearLayer(
                                x_d1, x_d2,
                                W_d1, W_d2,
                                b_d1));
        assert(linear_backward != NULL);//We must have already done backward before
        return linear_backward;
    }

public:
    static void do_forward( T* x, int x_d1, int x_d2,
                            T* W, int W_d1, int W_d2,
                            T* b, int b_d1,
                            T* y, int y_d1, int y_d2)
    {
        MKLDNNLinear<T> *fwd_object = get_forward_object(
                                            x, x_d1, x_d2,
                                            W, W_d1, W_d2,
                                            b, b_d1);
        fwd_object->forward(x, x_d1, x_d2,
                            W, W_d1, W_d2,
                            b, b_d1,
                            y, y_d1, y_d2);
    }

    static void do_forward(T* x, int x_d1, int x_d2,
                           T* W, int W_d1, int W_d2,
                           T* y, int y_d1, int y_d2)
    {
        MKLDNNLinear<T>* fwd_object = get_forward_object(x, x_d1, x_d2,
                                                         W, W_d1, W_d2,
                                                         NULL, -1);
        fwd_object->forward(x, x_d1, x_d2,
                            W, W_d1, W_d2,
                            y, y_d1, y_d2);
    }
    
    static void do_backward(T* x, int x_d1, int x_d2,
                            T* W, int W_d1, int W_d2,
                            T* b, int b_d1,
                            T* gy, int gy_d1, int gy_d2,
                            T* gW, int gW_d1, int gW_d2,
                            T* gx, int gx_d1, int gx_d2,
                            T* gb, int gb_d1)
    {
        MKLDNNLinear<T> *bwd_object = get_backward_object(x, x_d1, x_d2,
                                                          W, W_d1, W_d2,
                                                          b, b_d1);
        bwd_object->backward(x, x_d1, x_d2,
                             W, W_d1, W_d2,
                             b, b_d1,
                             gy, gy_d1, gy_d2,
                             gW, gW_d1, gW_d2,
                             gx, gx_d1, gx_d2,
                             gb, gb_d1);
    } 
 
    static void do_backward(T* x, int x_d1, int x_d2,
                            T* W, int W_d1, int W_d2,
                            T* gy, int gy_d1, int gy_d2,
                            T* gW, int gW_d1, int gW_d2,
                            T* gx, int gx_d1, int gx_d2)
    {
        MKLDNNLinear<T> *bwd_object = get_backward_object(x, x_d1, x_d2,
                                                          W, W_d1, W_d2,
                                                          NULL, -1);
        bwd_object->backward(x, x_d1, x_d2,
                             W, W_d1, W_d2,
                             gy, gy_d1, gy_d2,
                             gW, gW_d1, gW_d2,
                             gx, gx_d1, gx_d2);
    } 
 

    MKLDNNLinear();

    ~MKLDNNLinear();

    int setup_forward(T* x, int x_d1, int x_d2,
                       T* W, int W_d1, int W_d2,
                       T* b, int b_d1,
                       T* y, int y_d1, int y_d2);

    int forward(T* x, int x_d1, int x_d2,
                T* W, int W_d1, int W_d2,
                T* b, int b_d1,
                T* y, int y_d1, int y_d2);

    int forward(T* x, int x_d1, int x_d2,
                T* W, int W_d1, int W_d2,
                T* y, int y_d1, int y_d2);
   
    int setup_backward(T* x,  int x_d1, int x_d2,
                        T* W,  int W_d1, int W_d2,
                        T* b,  int b_d1,
                        T* gy, int gy_d1, int gy_d2,
                        T* gW, int gW_d1, int gW_d2,
                        T* gx, int gx_d1, int gx_d2,
                        T* gb, int gb_d1);

    int backward(T* x,  int x_d1, int x_d2,
                 T* W,  int W_d1, int W_d2,
                 T* b,  int b_d1,
                 T* gy, int gy_d1, int gy_d2,
                 T* gW, int gW_d1, int gW_d2,
                 T* gx, int gx_d1, int gx_d2,
                 T* gb, int gb_d1);

    int backward(T* x,  int x_d1, int x_d2,
                 T* W,  int W_d1, int W_d2,
                 T* gy, int gy_d1, int gy_d2,
                 T* gW, int gW_d1, int gW_d2,
                 T* gx, int gx_d1, int gx_d2);


private:
    //user primmemory
    std::shared_ptr<mkldnn::memory> user_src_memory_;
    std::shared_ptr<mkldnn::memory> user_weights_memory_;
    std::shared_ptr<mkldnn::memory> user_bias_memory_;
    std::shared_ptr<mkldnn::memory> user_dst_memory_;
    std::shared_ptr<mkldnn::memory> user_dst_diff_memory_;
    std::shared_ptr<mkldnn::memory> user_src_diff_memory_;
    std::shared_ptr<mkldnn::memory> user_weights_diff_memory_;
    std::shared_ptr<mkldnn::memory> user_bias_diff_memory_;
    /*******mkldnn internal prim memory*****/
    //forward
    std::shared_ptr<mkldnn::memory> fwd_internal_src_memory_;
    std::shared_ptr<mkldnn::memory> fwd_internal_weights_memory_;
    std::shared_ptr<mkldnn::memory> fwd_internal_bias_memory_;
    std::shared_ptr<mkldnn::memory> fwd_internal_dst_memory_;
    //backward
    std::shared_ptr<mkldnn::memory> bwd_internal_src_memory_;
    std::shared_ptr<mkldnn::memory> bwd_internal_weights_memory_;
    std::shared_ptr<mkldnn::memory> bwd_internal_dst_diff_memory_;
    std::shared_ptr<mkldnn::memory> bwd_internal_src_diff_memory_;
    std::shared_ptr<mkldnn::memory> bwd_internal_weights_diff_memory_;
    std::shared_ptr<mkldnn::memory> bwd_internal_bias_diff_memory_;
    //reorder primitve
    //forward
    mkldnn::primitive fwd_reorder_src_;
    mkldnn::primitive fwd_reorder_weights_;
    mkldnn::primitive fwd_reorder_dst_;
    mkldnn::primitive bwd_reorder_src_;
    mkldnn::primitive bwd_reorder_weights_;
    mkldnn::primitive bwd_reorder_src_diff_;
    mkldnn::primitive bwd_reorder_weights_diff_;
    mkldnn::primitive bwd_reorder_dst_diff_;
    //linear forward backward primitive
    std::shared_ptr<mkldnn::inner_product_forward::desc> linear_fwd_desc_;
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> linear_fwd_pd_;
    std::shared_ptr<mkldnn::primitive> linear_fwd_;
    std::shared_ptr<mkldnn::inner_product_backward_data::desc> linear_bwd_data_desc_;
    std::shared_ptr<mkldnn::inner_product_backward_weights::desc> linear_bwd_weights_desc_;
    std::shared_ptr<mkldnn::inner_product_backward_data::primitive_desc> linear_bwd_data_pd_;
    std::shared_ptr<mkldnn::inner_product_backward_weights::primitive_desc> linear_bwd_weights_pd_;
    std::shared_ptr<mkldnn::primitive> linear_bwd_data_;
    std::shared_ptr<mkldnn::primitive> linear_bwd_weights_;
    std::vector<mkldnn::primitive> bwd_data_primitives_;
    std::vector<mkldnn::primitive> bwd_weights_primitives_;


protected:
    mkldnn::stream* bwd_data_stream_;
    mkldnn::stream* bwd_weights_stream_;
};

#endif // _CONVOLUTION_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s