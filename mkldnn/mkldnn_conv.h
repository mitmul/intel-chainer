#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include <mkldnn.hpp>
#include <vector>
#include <memory>

template <typename T>
class Convolution2D {
public:
    Convolution2D();
    ~Convolution2D();
    
    /*
     * Convolution forward primitive setup 
     * Params:
     * X: input, (n,c,h,w)
     * W: weight, (n, out_c, h, w)
     * b: bias
     * y: output, (n, out_c, out_h, out_w)
     */
    void forward_setup(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
            T* W, int W_d1, int W_d2, int W_d3, int W_d4,
            T* b, int b_d1,
            T* y, int y_d1, int y_d2, int y_d3, int y_d4,
            int s1, int s2,
            int p1, int p2);
    
    /*
     * Convolution forward with bias
     * Params:
     * X: input, (n,c,h,w)
     * W: weight, (n, out_c, h, w)
     * b: bias
     * y: output, (n, out_c, out_h, out_w)
     */
    int forward(T* x, int x_d1, int x_d2, int x_d3, int x_d4, 
            T* W, int W_d1, int W_d2, int W_d3, int W_d4,
            T* b, int b_d1,
            T* y, int y_d1, int y_d2, int y_d3, int y_d4,
            int s1, int s2,
            int p1, int p2);

    /*
     * Convolution forward without bias
     * Params:
     * X: input, (n,c,h,w)
     * W: weight, (n, out_c, h, w)
     * y: output, (n, out_c, out_h, out_w)
     */
    int forward(T* x, int x_d1, int x_d2, int x_d3, int x_d4, 
            T* W, int W_d1, int W_d2, int W_d3, int W_d4,
            T* y, int y_d1, int y_d2, int y_d3, int y_d4,
            int s1, int s2,
            int p1, int p2);
    
    /*
     * Covolution backward primitive setup
     * Params:
     * 
     */
    void backward_setup(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
            T* W, int W_d1, int W_d2, int W_d3, int W_d4,
            T* b, int b_d1,
            T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
            T* gW, int gW_d1, int gW_d2, int gW_d3, int gW_d4,
            T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
            T* gb, int gb_d1);

    int backward( T* x, int x_d1, int x_d2, int x_d3, int x_d4,
            T* W, int W_d1, int W_d2, int W_d3, int W_d4,
            T* b, int b_d1,
            T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
            T* gW, int gW_d1, int gW_d2, int gW_d3, int gW_d4,
            T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
            T* gb, int gb_d1);

    int backward( T* x, int x_d1, int x_d2, int x_d3, int x_d4,
            T* W, int W_d1, int W_d2, int W_d3, int W_d4,
            T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
            T* gW, int gW_d1, int gW_d2, int gW_d3, int gW_d4,
            T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4);
private:
    // convolution primitive
    std::shared_ptr<mkldnn::primitive> conv_fwd_;
    std::shared_ptr<mkldnn::primitive> conv_bwd_data_;
    std::shared_ptr<mkldnn::primitive> conv_bwd_weights_;
    //memory reorder primitive
    //forward
    mkldnn::primitive conv_reorder_src_; // reorder x
    mkldnn::primitive conv_reorder_weights_; //reorder W
    mkldnn::primitive conv_reorder_dst_; //reorder y
    //backward
    mkldnn::primitive conv_bwd_reorder_src_; // reorder x
    mkldnn::primitive conv_bwd_reorder_diff_weights_; //reorder gW
    mkldnn::primitive conv_bwd_reorder_dst_; //reorder gY
    mkldnn::primitive conv_bwd_reorder_weights_; //reorder W
    mkldnn::primitive conv_bwd_reorder_diff_src_; //reorder gX

    bool fwd_reorder_conv_src_ = false;
    bool fwd_reorder_conv_weights_ = false;
    bool fwd_reorder_conv_dst_ = false;

    bool bwd_reorder_src_ = false;
    bool bwd_reorder_diff_dst_ = false;
    bool bwd_reorder_diff_weights_ = false;
    bool bwd_reorder_weights_ = false;
    bool bwd_reorder_diff_src_ = false;

    bool fwd_first_run_ = true;
    bool bwd_first_run_ = true;

    //desc & prmitive desc
    //forward
    std::shared_ptr<mkldnn::convolution_forward::desc> fwd_desc_;
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> fwd_prim_desc_;
    //backward
    std::shared_ptr<mkldnn::convolution_backward_weights::desc> bwd_weights_desc_;
    std::shared_ptr<mkldnn::convolution_backward_weights::primitive_desc> bwd_weights_prim_desc_;
    std::shared_ptr<mkldnn::convolution_backward_data::desc> bwd_data_desc_;
    std::shared_ptr<mkldnn::convolution_backward_data::primitive_desc> bwd_data_prim_desc_;
    
    //stream
    std::shared_ptr<mkldnn::stream> fwd_stream_;
    std::vector<mkldnn::primitive> fwd_primitives_;
    std::shared_ptr<mkldnn::stream> bwd_weights_stream_;
    std::vector<mkldnn::primitive> bwd_weights_primitives_;
    std::shared_ptr<mkldnn::stream> bwd_data_stream_;
    std::vector<mkldnn::primitive> bwd_data_primitives_;

    //memory dims
    mkldnn::memory::dims src_tz_;
    mkldnn::memory::dims weights_tz_;
    mkldnn::memory::dims dst_tz_;
    mkldnn::memory::dims bias_tz_;
    mkldnn::memory::dims strides_;
    mkldnn::memory::dims padding_;

    //user memory
    //forward
    std::shared_ptr<mkldnn::memory> user_src_memory_; //x
    std::shared_ptr<mkldnn::memory> user_weights_memory_; //W
    std::shared_ptr<mkldnn::memory> user_bias_memory_; //b
    std::shared_ptr<mkldnn::memory> user_dst_memory_; //y
    //backward
    std::shared_ptr<mkldnn::memory> user_bwd_diff_src_memory_; //gX
    std::shared_ptr<mkldnn::memory> user_bwd_diff_weights_memory_; //gW
    std::shared_ptr<mkldnn::memory> user_bwd_diff_bias_memory_; //gb
    std::shared_ptr<mkldnn::memory> user_bwd_diff_dst_memory_; //gy
    std::shared_ptr<mkldnn::memory> user_bwd_src_memory_; //x
    std::shared_ptr<mkldnn::memory> user_bwd_weights_memory_; //W
//    std::shared_ptr<mkldnn::memory> user_bwd_dst_memory_; //y

    //MKLDNN memory
    //forward
    std::shared_ptr<mkldnn::memory> src_memory_; // x
    std::shared_ptr<mkldnn::memory> weights_memory_;// W
    std::shared_ptr<mkldnn::memory> bias_memory_;// b
    std::shared_ptr<mkldnn::memory> dst_memory_; //y
    //backward
    std::shared_ptr<mkldnn::memory> bwd_src_memory_; // x
    std::shared_ptr<mkldnn::memory> bwd_weights_memory_; //W
    std::shared_ptr<mkldnn::memory> bwd_diff_weights_memory_; //gW
    std::shared_ptr<mkldnn::memory> bwd_diff_src_memory_; //gX
    std::shared_ptr<mkldnn::memory> bwd_diff_bias_memory_; //gb
    std::shared_ptr<mkldnn::memory> bwd_diff_dst_memory_; // gy
    
    //memory desc
    //forward & backward can share same mem desc
    std::shared_ptr<mkldnn::memory::desc> src_md_; //x & gx
    std::shared_ptr<mkldnn::memory::desc> weights_md_;// W & gW
    std::shared_ptr<mkldnn::memory::desc> bias_md_; // b & gb
    std::shared_ptr<mkldnn::memory::desc> dst_md_; // y & gy
};

#endif // _CONVOLUTION_H_
