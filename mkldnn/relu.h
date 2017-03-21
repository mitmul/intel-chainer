#pragma once

#include <mkldnn.hpp>
#include <vector>

template <typename T>
class Relu {
public:
    Relu();
    int test_buf(
                     T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                     T* y, int y_size
                    );
#if 0
    Relu(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
         T* y, int y_d1, int y_d2, int y_d3, int y_d4);
#endif
    int forward_setup(T* x, int x_size,
                      T* y, int y_size);
    int forward(T* x, int x_size,
                T* y, int y_size);

    int backward_setup(T* x, int x_size,
                 T* gy, int gy_size,
                 T* gx, int gx_size);
    int backward(T* x, int x_size,
                 T* gy, int gy_size,
                 T* gx, int gx_size);
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
