#pragma once

#include <mkldnn.hpp>
#include <vector>

template <typename T>
class Relu {
public:
    Relu();
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
    mkldnn::memory *relu_user_src_memory_;
    mkldnn::memory *relu_dst_memory_;
    mkldnn::memory::desc* relu_src_md_;
    mkldnn::relu_forward::desc* relu_desc_;
    mkldnn::relu_forward::primitive_desc* relu_prim_desc_;
    mkldnn::relu_forward* relu_fwd_;
    mkldnn::stream* fw_stream_;
    std::vector<mkldnn::primitive> fw_primitives_;

    //backward
    mkldnn::memory *relu_diff_src_memory_, *relu_diff_dst_memory_;
    mkldnn::memory::desc* relu_diff_dst_md_;
    mkldnn::relu_backward::desc* relu_bwd_desc_;
    mkldnn::relu_backward::primitive_desc* relu_bwd_pd_;
    mkldnn::relu_backward *relu_bwd_;

    mkldnn::stream* bw_stream_;
    std::vector<mkldnn::primitive> bw_primitives_;
};
