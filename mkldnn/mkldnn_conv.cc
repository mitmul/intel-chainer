#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "mkldnn_conv.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
Convolution2D<T>::Convolution2D()
{
    fwd_stream_.reset(new stream(stream::kind::eager));
    bwd_weights_stream_.reset(new stream(stream::kind::eager));
    bwd_data_stream_.reset(new stream(stream::kind::eager));
}

template<typename T>
Convolution2D<T>::~Convolution2D()
{
}

template<typename T>
void Convolution2D<T>::forward_setup(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2, int W_d3, int W_d4,
        T* b, int b_d1,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4,
        int s1, int s2,
        int p1, int p2)
{
  //  LOG(INFO) << "Convolution forward_setup";

    //LOG(INFO) << "x =(" << x_d1 << "," << x_d2 << "," << x_d3 << "," << x_d4 << ")";
    //LOG(INFO) << "W =(" << W_d1 << "," << W_d2 << "," << W_d3 << "," << W_d4 << ")";
    //LOG(INFO) << "b =(" << b_d1 << ")";
    //LOG(INFO) << "y =(" << y_d1 << "," << y_d2 << "," << y_d3 << "," << y_d4 << ")";

    src_tz_ = {x_d1, x_d2, x_d3, x_d4};
    weights_tz_ = {W_d1, W_d2, W_d3, W_d4};
    dst_tz_ = {y_d1, y_d2, y_d3, y_d4};
    strides_ = {s1, s2};
    bias_tz_ = {b_d1};
    padding_ = {p1, p2};

    /* create memory for user data */
    user_src_memory_.reset(new memory({{{src_tz_}, memory::data_type::f32,
                                      memory::format::nchw}, cpu_engine}, x));
    user_weights_memory_.reset(new memory({{{weights_tz_},
                                          memory::data_type::f32, memory::format::oihw}, cpu_engine}, W));
    /* in current design, output is also allocated in python part */
    user_dst_memory_.reset(new memory({{{dst_tz_}, memory::data_type::f32,
                                      memory::format::nchw}, cpu_engine}, y));
    if (b != NULL)
        user_bias_memory_.reset(new memory({{{bias_tz_},
                                      memory::data_type::f32, memory::format::x}, cpu_engine}, b));
    
    /* create memory descriptors for convolution data w/ no specified format */
    src_md_.reset(new memory::desc({src_tz_}, memory::data_type::f32,
                                   memory::format::any));
    weights_md_.reset(new memory::desc({weights_tz_},
                                       memory::data_type::f32, memory::format::any));
    dst_md_.reset(new memory::desc({dst_tz_}, memory::data_type::f32,
                                   memory::format::any));
    if (b != NULL)
        bias_md_.reset(new memory::desc({bias_tz_}, memory::data_type::f32,
                                   memory::format::any));
    
    /* create a convolution */
    if (b != NULL)
        fwd_desc_.reset(new convolution_forward::desc(prop_kind::forward,
                                                 convolution_direct, *src_md_, *weights_md_, *bias_md_,
                                                 *dst_md_, strides_, padding_, padding_,
                                                 padding_kind::zero));
    else
        fwd_desc_.reset(new convolution_forward::desc(prop_kind::forward,
                                                 convolution_direct, *src_md_, *weights_md_,
                                                 *dst_md_, strides_, padding_, padding_,
                                                 padding_kind::zero));

    fwd_prim_desc_.reset(new convolution_forward::primitive_desc(*fwd_desc_, cpu_engine));

    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    src_memory_ = user_src_memory_;
    bool reorder_conv_src = false;
    if (memory::primitive_desc(fwd_prim_desc_.get()->src_primitive_desc()) 
            != user_src_memory_.get()->get_primitive_desc()) {
    //    LOG(INFO) << "fwd reorder src dim";
        src_memory_.reset(new memory(fwd_prim_desc_.get()->src_primitive_desc()));
        conv_reorder_src_ = reorder(*user_src_memory_,*src_memory_);
        reorder_conv_src = true;
    }

    weights_memory_ = user_weights_memory_;
    bool reorder_conv_weights = false;
    if (memory::primitive_desc((*fwd_prim_desc_).weights_primitive_desc())
            != (*user_weights_memory_).get_primitive_desc()) {
     //   LOG(INFO) << "fwd reorder weight dim";
        weights_memory_.reset(new memory(fwd_prim_desc_.get()->weights_primitive_desc()));
        conv_reorder_weights_ = reorder(*user_weights_memory_, *weights_memory_);
        reorder_conv_weights = true;
    }

    dst_memory_ = user_dst_memory_;
    bool reorder_conv_dst = false;

    if (memory::primitive_desc(fwd_prim_desc_.get()->dst_primitive_desc())
            != user_dst_memory_.get()->get_primitive_desc()) {
     //   LOG(INFO) << "fwd reorder output dim";
        dst_memory_.reset(new memory(fwd_prim_desc_.get()->dst_primitive_desc()));
        conv_reorder_dst_ = reorder(*dst_memory_, *user_dst_memory_);
        reorder_conv_dst = true;
    }
    
    /* create convolution primitive and add it to net */
    if (b != NULL)
        conv_fwd_.reset(new convolution_forward(*fwd_prim_desc_, *src_memory_,
                                      *weights_memory_, *user_bias_memory_, *dst_memory_));
    else
        conv_fwd_.reset(new convolution_forward(*fwd_prim_desc_, *src_memory_,
                                      *weights_memory_, *dst_memory_));
    
    //put all primitives into fwd_stream_
    if (reorder_conv_src){
        fwd_primitives_.push_back(conv_reorder_src_);
    }     
    if (reorder_conv_weights){
        fwd_primitives_.push_back(conv_reorder_weights_);
    }     
    fwd_primitives_.push_back(*conv_fwd_);
    if (reorder_conv_dst){
        fwd_primitives_.push_back(conv_reorder_dst_);
    }     
    return;
}

template<typename T>
int Convolution2D<T>::forward(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2, int W_d3, int W_d4,
        T* b, int b_d1,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4,
        int s1, int s2,
        int p1, int p2)
{
//    LOG(INFO) << "Convolution forward";
    if (conv_fwd_ == NULL) {
        forward_setup(x, x_d1, x_d2, x_d3, x_d4,
                W, W_d1, W_d2, W_d3, W_d4,
                b, b_d1,
                y, y_d1, y_d2, y_d3, y_d4,
                s1, s2,
                p1, p2);
    }
 
    fwd_stream_->submit(fwd_primitives_);
    return 0;
}

template<typename T>
int Convolution2D<T>::forward(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2, int W_d3, int W_d4,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4,
        int s1, int s2,
        int p1, int p2)
{
//    LOG(INFO) << "Convolution forward without bias";
//    LOG(INFO) << conv_fwd_;
        
    forward(x, x_d1, x_d2, x_d3, x_d4,
            W, W_d1, W_d2, W_d3, W_d4,
            NULL, -1,
            y, y_d1, y_d2, y_d3, y_d4,
            s1, s2,
            p1, p2);
    return 0;
}

template<typename T>
void Convolution2D<T>::backward_setup( T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2, int W_d3, int W_d4,
        T* b, int b_d1,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        T* gW, int gW_d1, int gW_d2, int gW_d3, int gW_d4,
        T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
        T* gb, int gb_d1)
{
    //LOG(INFO) << "Covolution backward_setup";
    /* create user format memory*/
    user_bwd_src_memory_.reset(new memory({{{ src_tz_ }, memory::data_type::f32,
                memory::format::nchw }, cpu_engine }, x)); //x
    user_bwd_weights_memory_.reset(new memory({{{ weights_tz_ }, memory::data_type::f32,
                memory::format::oihw }, cpu_engine }, W)); //W
    user_bwd_dst_memory_.reset(new memory({{{ dst_tz_ }, memory::data_type::f32,
                memory::format::nchw }, cpu_engine }, gy)); //gy
    user_bwd_diff_weights_memory_.reset(new memory({{{ weights_tz_ }, memory::data_type::f32,
                memory::format::oihw }, cpu_engine }, gW)); //gW
    user_bwd_diff_src_memory_.reset(new memory({{{ src_tz_ }, memory::data_type::f32,
                memory::format::nchw }, cpu_engine }, gx)); //gx
    if ( b != NULL ) {
        user_bwd_diff_bias_memory_.reset(new memory({{{ bias_tz_}, memory::data_type::f32,
                    memory::format::x,}, cpu_engine}, gb)); //gB
    }

    /* 
     * create backward convolution operator desc
     * memory desc can be shared between forward and backward, since they have same shape
     * */
    if ( b != NULL) {
        /* 
         * weight backward conv desc (gW = gy * X)
         * src_md: x
         * weigths_md: gW
         * bias_md: gb
         * dst_md: gy
         * */
        bwd_weights_desc_.reset(new convolution_backward_weights::desc(
                    convolution_direct, *src_md_, *weights_md_,
                    *bias_md_, *dst_md_, strides_, padding_, padding_, padding_kind::zero));
    } else {
        /* 
         * weight backward conv prim desc (gW = gy * X)
         * src_md: x
         * weigths_md: gW
         * dst_md: gy
         * */
        bwd_weights_desc_.reset(new convolution_backward_weights::desc(
                    convolution_direct, *src_md_, *weights_md_,
                    *dst_md_, strides_, padding_, padding_, padding_kind::zero));
    }
    
    /* 
     * data backward conv prim desc (gX = gy * W)
     * for data backward conv, no need b/gb
     * src_md: gx
     * weigths_md: W
     * dst_md: gy
     * */
    bwd_data_desc_.reset(new convolution_backward_data::desc(
                convolution_direct, *src_md_, *weights_md_,
                *dst_md_, strides_, padding_, padding_, padding_kind::zero));

    /* create backward conv prim desc*/
    bwd_weights_prim_desc_.reset(new convolution_backward_weights::primitive_desc(
                *bwd_weights_desc_, cpu_engine, *fwd_prim_desc_));
    bwd_data_prim_desc_.reset(new convolution_backward_data::primitive_desc(
                *bwd_data_desc_, cpu_engine, *fwd_prim_desc_));

    /*
     * for best performance convolution backward might choose different memory format for src and diffsrc
     * than the memory formats preferred by forward convolution for src and dst respectively
     * create reorder primitive for src from forward convolution to the format chosen by backward convolution */

    /* user_bwd_src_memory_ ==> x */
    bwd_src_memory_ = user_bwd_src_memory_;
    bool reorder_bwd_src = false;
    if (memory::primitive_desc(bwd_weights_prim_desc_.get()->src_primitive_desc())
            != user_bwd_src_memory_.get()->get_primitive_desc()) {
      //  LOG(INFO) << "bwd reorder x";
        bwd_src_memory_.reset(new memory(bwd_weights_prim_desc_.get()->src_primitive_desc()));
        conv_bwd_reorder_src_ = reorder(*user_bwd_src_memory_, *bwd_src_memory_);
        reorder_bwd_src = true;
    }

    /* user_bwd_dst_memory_ ==> gy */
    bwd_diff_dst_memory_ = user_bwd_dst_memory_;
    bool reorder_bwd_diff_dst = false;
    if (memory::primitive_desc(bwd_weights_prim_desc_.get()->diff_dst_primitive_desc())
            != user_bwd_dst_memory_.get()->get_primitive_desc()) {
      //  LOG(INFO) << "bwd reorder gy";
        bwd_diff_dst_memory_.reset(new memory(bwd_weights_prim_desc_.get()->diff_dst_primitive_desc()));
        conv_bwd_reorder_dst_ = reorder(*user_bwd_dst_memory_, *bwd_diff_dst_memory_);
        reorder_bwd_diff_dst = true;
    }

    /* user_bwd_diff_weights_memory_ ==> gW */
    bwd_diff_weights_memory_ = user_bwd_diff_weights_memory_;
    bool reorder_bwd_diff_weights = false;
    if (memory::primitive_desc(bwd_weights_prim_desc_.get()->diff_weights_primitive_desc())
            != user_bwd_diff_weights_memory_.get()->get_primitive_desc()) {
       // LOG(INFO) << "bwd reorder gW";
        bwd_diff_weights_memory_.reset(new memory(bwd_weights_prim_desc_.get()->diff_weights_primitive_desc()));
        conv_bwd_reorder_diff_weights_ = reorder(*bwd_diff_weights_memory_, *user_bwd_diff_weights_memory_);
        reorder_bwd_diff_weights = true;
    }

    /* user_bwd_weights_memory_ ==> W */
    bwd_weights_memory_ = user_bwd_weights_memory_;
    bool reorder_bwd_weights = false;
    if (memory::primitive_desc(bwd_data_prim_desc_.get()->weights_primitive_desc())
            != user_bwd_weights_memory_.get()->get_primitive_desc()) {
        // LOG(INFO) << "bwd reorder W";
        bwd_weights_memory_.reset(new memory(bwd_data_prim_desc_.get()->weights_primitive_desc()));
        conv_bwd_reorder_weights_ = reorder(*user_bwd_weights_memory_, *bwd_weights_memory_);
        reorder_bwd_weights = true;
    }

    /* user_bwd_diff_src_memory_ ==> gX */
    bwd_diff_src_memory_ = user_bwd_diff_src_memory_;
    bool reorder_bwd_diff_src = false;
    if (memory::primitive_desc(bwd_data_prim_desc_.get()->diff_src_primitive_desc())
            != user_bwd_diff_src_memory_.get()->get_primitive_desc()) {
        // LOG(INFO) << "bwd reorder gX";
        bwd_diff_src_memory_.reset(new memory(bwd_data_prim_desc_.get()->diff_src_primitive_desc()));
        conv_bwd_reorder_diff_src_ = reorder(*bwd_diff_src_memory_, *user_bwd_diff_src_memory_);
        reorder_bwd_diff_src = true; 
    } 

    /* create weight conv bwd prim */
    if (b != NULL) {
        /* 
         * create convolution backward primitive (gW = gy * X) 
         * src_memory: x
         * diff_dst_memory: gy
         * diff_weights_memory: gW
         * diff_bias_memory: gb
         * */
        conv_bwd_weights_.reset( new convolution_backward_weights(
                    *bwd_weights_prim_desc_, *bwd_src_memory_,
                    *bwd_diff_dst_memory_, *bwd_diff_weights_memory_, *user_bwd_diff_bias_memory_));
    } else {
        /* 
         * create convolution backward primitive (gW = gy * x)
         * src_memory: x
         * diff_dst_memory: gy
         * diff_weights_memory: gW
         * */
        conv_bwd_weights_.reset( new convolution_backward_weights(
                    *bwd_weights_prim_desc_, *bwd_src_memory_,
                    *bwd_diff_dst_memory_, *bwd_diff_weights_memory_));
    }

    /* 
     * create data conv bwd prim (gX = gy * W)
     * */
    conv_bwd_data_.reset(new convolution_backward_data(
                *bwd_data_prim_desc_, *bwd_diff_dst_memory_, *bwd_weights_memory_, *bwd_diff_src_memory_));

    /*
     * create weight conv bwd stream (gW = gy * X)
     *
     * reorder_x -> reorder_gy -> weight_conv_bwd -> reoder_gW
     *
     */
    if (reorder_bwd_src) {
        bwd_weights_primitives_.push_back(conv_bwd_reorder_src_);
    }
    if (reorder_bwd_diff_dst) {
        bwd_weights_primitives_.push_back(conv_bwd_reorder_dst_);
    }
    bwd_weights_primitives_.push_back(*conv_bwd_weights_);
    if (reorder_bwd_diff_weights) {
        bwd_weights_primitives_.push_back(conv_bwd_reorder_diff_weights_);
    }

    /*
     * create data conv bwd stream (gX = gy * W)
     *
     * reorder_W -> reorder_gy -> data_conv_bwd -> reorder_gX
     *
     */
    if (reorder_bwd_weights) {
        bwd_data_primitives_.push_back(conv_bwd_reorder_weights_);
    }
    if (reorder_bwd_diff_dst) {
        bwd_data_primitives_.push_back(conv_bwd_reorder_dst_);
    }
    bwd_data_primitives_.push_back(*conv_bwd_data_);
    if (reorder_bwd_diff_src) {
        bwd_data_primitives_.push_back(conv_bwd_reorder_diff_src_);
    }


    return;
}

template<typename T>
int Convolution2D<T>::backward( T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2, int W_d3, int W_d4,
        T* b, int b_d1,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        T* gW, int gW_d1, int gW_d2, int gW_d3, int gW_d4,
        T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4,
        T* gb, int gb_d1)
{
//    LOG(INFO) << "Convolution backward with bias";
    if (conv_bwd_weights_ == NULL) {
        backward_setup(x, x_d1, x_d2, x_d3, x_d4,
                W, W_d1, W_d2, W_d3, W_d4,
                b, b_d1,
                gy, gy_d1, gy_d2, gy_d3, gy_d4,
                gW, gW_d1, gW_d2, gW_d3, gW_d4,
                gx, gx_d1, gx_d2, gx_d3, gx_d4,
                gb, gb_d1);
    }
    bwd_weights_stream_->submit(bwd_weights_primitives_);
    bwd_data_stream_->submit(bwd_data_primitives_);
    return 0;
}

template<typename T>
int Convolution2D<T>::backward( T* x, int x_d1, int x_d2, int x_d3, int x_d4,
        T* W, int W_d1, int W_d2, int W_d3, int W_d4,
        T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
        T* gW, int gW_d1, int gW_d2, int gW_d3, int gW_d4,
        T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4)
{
//    LOG(INFO) << "Convolution backward without bias";
    backward(x, x_d1, x_d2, x_d3, x_d4,
            W, W_d1, W_d2, W_d3, W_d4,
            NULL, -1,
            gy, gy_d1, gy_d2, gy_d3, gy_d4,
            gW, gW_d1, gW_d2, gW_d3, gW_d4,
            gx, gx_d1, gx_d2, gx_d3, gx_d4,
            NULL, -1);
    return 0;
}

template class Convolution2D<float>;
template class Convolution2D<double>;
