#ifndef _LINEAR_H_
#define _LINEAR_H_

#include <mkldnn.hpp>
#include <vector>
#include <memory>
#include "layer.h"
template <typename T>
class MKLDNNLinear:public Layer<T> {
public:
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
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> linearFwd_pd;
    std::shared_ptr<mkldnn::inner_product_backward_data::primitive_desc> linearBwdData_pd;
    std::shared_ptr<mkldnn::inner_product_backward_weights::primitive_desc> linearBwdWeights_pd;
    std::vector<mkldnn::primitive> bwd_data_primitives_;
    std::vector<mkldnn::primitive> bwd_weights_primitives_;

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
protected:
    mkldnn::stream* bwd_data_stream_;
    mkldnn::stream* bwd_weights_stream_;
};

#endif // _CONVOLUTION_H_
