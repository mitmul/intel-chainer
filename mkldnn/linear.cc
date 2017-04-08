#include <glog/logging.h>
#include <iostream>
#include "common.h"
#include "mkldnn.hpp"
#include <memory>
#include "linear.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
MKLDNNLinear<T>::MKLDNNLinear()
{
    this->forward_stream_ = new stream(stream::kind::eager);
    this->bwd_data_stream_ = new stream(stream::kind::eager);
    this->bwd_weights_stream_ = new stream(stream::kind::eager);
}

template<typename T>
MKLDNNLinear<T>::~MKLDNNLinear()
{
    
}

template <typename T>
int MKLDNNLinear<T>::setup_forward(T* x, int x_d1, int x_d2, //x_d1 = n, x_d2 = ic  ----- input
                                         T* W, int W_d1, int W_d2, //W_d1 = oc, W_d2 = ic
                                         T* b, int b_d1, 
                                         T* y, int y_d1, int y_d2) // y_d1 = n, y_d2 = ic ----- output
{
    /*
    LOG(INFO) << "Linear Forward Init, with b";
    LOG(INFO) << "x = (" << x_d1 << "," << x_d2 << ")";
    LOG(INFO) << "W = (" << W_d1 << "," << W_d2 << ")";
    LOG(INFO) << "b = (" << b_d1 << ")";
    LOG(INFO) << "y = (" << y_d1 << "," << y_d2 << ")";
    */
    
    // Initialize memory descriptors (format = any) to create linear descriptor
    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt = memory::format::any;
    
    memory::dims src_tz = memory::dims{x_d1, x_d2};
    memory::dims weights_tz = memory::dims{W_d1, W_d2};
    memory::dims bias_tz = {b_d1};
    memory::dims dst_tz = memory::dims{y_d1, y_d2};
    
    memory::desc init_src_md({src_tz}, mpcsn, mfmt);
    memory::desc init_weights_md({weights_tz}, mpcsn, mfmt);

    memory::desc init_bias_md({bias_tz}, mpcsn, mfmt);
    memory::desc init_dst_md({dst_tz}, mpcsn, mfmt);
    
    //Initialize linear layer primitive descriptor
    if (b != NULL) {
        linear_fwd_desc_.reset(new inner_product_forward::desc(prop_kind::forward, init_src_md, init_weights_md,
                                                        init_bias_md, init_dst_md));
    } else {
        linear_fwd_desc_.reset(new inner_product_forward::desc(prop_kind::forward, init_src_md, init_weights_md,
                                                        init_dst_md));
    }
    //------Determing engine to use ----------
    //Current, treat the engine is MKLDNN:CPU
    linear_fwd_pd_.reset(new inner_product_forward::primitive_desc(*linear_fwd_desc_, cpu_engine));

    //Create user memory primitive 
    user_src_memory_.reset(new memory({{{src_tz}, mpcsn, memory::format::nc}, cpu_engine}, dummy));
    user_weights_memory_.reset(new memory({{{weights_tz}, mpcsn, memory::format::oi}, cpu_engine}, dummy));
    if (b != NULL)
        user_bias_memory_.reset(new memory({{{bias_tz}, mpcsn, memory::format::x}, cpu_engine}, dummy));
    /* in current design, output is also allocated in python part */
    user_dst_memory_.reset(new memory({{{dst_tz}, mpcsn, memory::format::nc}, cpu_engine}, dummy));
    

    //create mkldnn memory primitive descripor
    fwd_internal_src_memory_ = user_src_memory_;
    fwd_internal_weights_memory_ = user_weights_memory_;
    fwd_internal_bias_memory_ = user_bias_memory_;
    fwd_internal_dst_memory_ = user_dst_memory_;
    
   //create reoder primitve if needed
    bool is_src_reordered = false;
    bool is_weights_reordered = false;
    bool is_dst_reordered = false;
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc
    /* create reorder primitives between user src and internal src if required */
    if ((*user_src_memory_).get_primitive_desc() != MemPD(linear_fwd_pd_.get()->src_primitive_desc())) {
       LOG(INFO) << "fwd reorder x";
       fwd_internal_src_memory_.reset(new memory(linear_fwd_pd_.get()->src_primitive_desc()));
       fwd_reorder_src_ = reorder(*user_src_memory_, *fwd_internal_src_memory_);
       is_src_reordered = true;
    }
    
    /* create reorder primitives between user weights and internal weights if required */
    if ((*user_weights_memory_).get_primitive_desc() != MemPD(linear_fwd_pd_.get()->weights_primitive_desc())) {
       LOG(INFO) << "fwd reorder W";
       fwd_internal_weights_memory_.reset(new memory(linear_fwd_pd_.get()->weights_primitive_desc()));
       fwd_reorder_weights_ = reorder(*user_weights_memory_, *fwd_internal_weights_memory_);
       is_weights_reordered = true;
    }
 
    /* create reorder primitives between user dst and internal dst if required */
    if ((*user_dst_memory_).get_primitive_desc() != MemPD(linear_fwd_pd_.get()->dst_primitive_desc())) {
       LOG(INFO) << "fwd reorder y";
       fwd_internal_dst_memory_.reset(new memory(linear_fwd_pd_.get()->dst_primitive_desc()));
       fwd_reorder_dst_ = reorder(*fwd_internal_dst_memory_, *user_dst_memory_);
       is_dst_reordered = true;
    }
    if (b != NULL)
        linear_fwd_.reset(new inner_product_forward(*linear_fwd_pd_, *fwd_internal_src_memory_, *fwd_internal_weights_memory_, *fwd_internal_bias_memory_, *fwd_internal_dst_memory_));
    else
        linear_fwd_.reset(new inner_product_forward(*linear_fwd_pd_, *fwd_internal_src_memory_, *fwd_internal_weights_memory_, *fwd_internal_dst_memory_));
    if (is_src_reordered)
        this->forward_primitives_.push_back(fwd_reorder_src_);
    if (is_weights_reordered)
        this->forward_primitives_.push_back(fwd_reorder_weights_);
    this->forward_primitives_.push_back(*linear_fwd_);
    if (is_dst_reordered)
        this->forward_primitives_.push_back(fwd_reorder_dst_);
    this->forward_first_use_ = true;
    return 0;
}
template <typename T>
int MKLDNNLinear<T>::setup_backward(T* x,  int x_d1, int x_d2,
                                     T* W,  int W_d1, int W_d2,
                                     T* b,  int b_d1,
                                     T* gy, int gy_d1, int gy_d2,
                                     T* gW, int gW_d1, int gW_d2,
                                     T* gx, int gx_d1, int gx_d2,
                                     T* gb, int gb_d1)
{
    // LOG(INFO) << "Linear Backward Init";
    //Initialze memory descriptors (format = any) to create linear descriptor
    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt = memory::format::any;
    
    memory::dims src_tz = memory::dims{x_d1, x_d2};
    memory::dims weights_tz = memory::dims{W_d1, W_d2};
    memory::dims bias_tz = memory::dims{b_d1};
    memory::dims dst_tz = memory::dims{gy_d1, gy_d2};

    memory::desc init_src_md({src_tz}, mpcsn, mfmt);
    memory::desc init_weights_md({weights_tz}, mpcsn, mfmt);
    std::shared_ptr<mkldnn::memory::desc> init_bias_md_p;
    if (b != NULL) {
        init_bias_md_p.reset(new memory::desc({bias_tz}, mpcsn, mfmt));
    } else {
        init_bias_md_p.reset(new memory::desc({memory::dims{0}}, mpcsn, memory::format::format_undef));
    }
    memory::desc init_dst_md({dst_tz}, mpcsn, mfmt);
    
    // Initialze linear primitive descriptor
    
    linear_bwd_data_desc_.reset(new inner_product_backward_data::desc(init_src_md, init_weights_md, init_dst_md));
    if (b != NULL){
        linear_bwd_weights_desc_.reset(new inner_product_backward_weights::desc(init_src_md, init_weights_md, *init_bias_md_p, init_dst_md));
    } else {
        linear_bwd_weights_desc_.reset(new inner_product_backward_weights::desc(init_src_md, init_weights_md, *init_bias_md_p, init_dst_md));
    }
    //-----Determining engine to use-----------------------
    //Current engine is MKLDNN:CPU
    linear_bwd_data_pd_.reset(new inner_product_backward_data::primitive_desc(*linear_bwd_data_desc_,
                cpu_engine, *linear_fwd_pd_));
    linear_bwd_weights_pd_.reset(new inner_product_backward_weights::primitive_desc(*linear_bwd_weights_desc_,
                cpu_engine, *linear_fwd_pd_));
    //Create user memory primitive
    user_src_diff_memory_.reset(new memory({{{src_tz}, mpcsn, memory::format::nc}, cpu_engine}, dummy));
    user_weights_diff_memory_.reset(new memory({{{weights_tz}, mpcsn, memory::format::oi}, cpu_engine}, dummy));
    user_dst_diff_memory_.reset(new memory({{{dst_tz}, mpcsn, memory::format::nc}, cpu_engine}, dummy));
    if (b != NULL)
        user_bias_diff_memory_.reset(new memory({{{bias_tz}, mpcsn, memory::format::x}, cpu_engine}, dummy));

    //create internal memory primivive
    bwd_internal_src_memory_ = user_src_memory_;
    bwd_internal_weights_memory_ = user_weights_memory_;
    bwd_internal_src_diff_memory_ = user_src_diff_memory_;
    bwd_internal_weights_diff_memory_ = user_weights_diff_memory_;
    bwd_internal_dst_diff_memory_ = user_dst_diff_memory_;
    if (b != NULL) 
        bwd_internal_bias_diff_memory_ = user_bias_diff_memory_;
    //--------------check reorder-------------------------
    bool is_src_reordered = false;
    bool is_weights_reordered = false;
    bool is_src_diff_reordered = false;
    bool is_weights_diff_reordered = false;
    bool is_dst_diff_reordered = false;
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc
    if ((*user_src_memory_).get_primitive_desc() 
            != MemPD(linear_bwd_weights_pd_.get()->src_primitive_desc())) {
        LOG(INFO) << "bwd reorder x";
        bwd_internal_src_memory_.reset(new memory(linear_bwd_weights_pd_.get()->src_primitive_desc()));
        bwd_reorder_src_ = reorder(*user_src_memory_, *bwd_internal_src_memory_);
        is_src_reordered = true;
    }
    
    if ((*user_weights_memory_).get_primitive_desc() 
            != MemPD(linear_bwd_data_pd_.get()->weights_primitive_desc())) {
        LOG(INFO) << "bwd reorder w";
        bwd_internal_weights_memory_.reset(new memory(linear_bwd_data_pd_.get()->weights_primitive_desc()));
        bwd_reorder_weights_ = reorder(*user_weights_memory_, *bwd_internal_weights_memory_);
        is_weights_reordered = true;     
    }

    if ((*user_src_diff_memory_).get_primitive_desc() 
            != MemPD(linear_bwd_data_pd_.get()->diff_src_primitive_desc())) {
        LOG(INFO) << "bwd reorder gx";
        bwd_internal_src_diff_memory_.reset(new memory(linear_bwd_data_pd_.get()->diff_src_primitive_desc()));
        bwd_reorder_src_diff_ = reorder(*bwd_internal_src_diff_memory_, *user_src_diff_memory_);
        is_src_diff_reordered = true;
    }

    if ((*user_weights_diff_memory_).get_primitive_desc() 
            != MemPD(linear_bwd_weights_pd_.get()->diff_weights_primitive_desc())) {
        LOG(INFO) << "bwd reorder gw";
        bwd_internal_weights_diff_memory_.reset(new memory(linear_bwd_weights_pd_.get()->diff_weights_primitive_desc()));
        bwd_reorder_weights_diff_ = reorder(*bwd_internal_weights_diff_memory_, *user_weights_diff_memory_);
        is_weights_diff_reordered = true;
    }

    if ((*bwd_internal_dst_diff_memory_).get_primitive_desc()
            != MemPD(linear_bwd_weights_pd_.get()->diff_dst_primitive_desc())) {
        LOG(INFO) << "bwd reorder gy";
        bwd_internal_dst_diff_memory_.reset(new memory(linear_bwd_weights_pd_.get()->diff_dst_primitive_desc()));
        bwd_reorder_dst_diff_ = reorder(*user_dst_diff_memory_, *bwd_internal_dst_diff_memory_);
        is_dst_diff_reordered = true;
    }

    //create linear bwd data primitive  
    linear_bwd_data_.reset(new inner_product_backward_data(*linear_bwd_data_pd_, *bwd_internal_dst_diff_memory_,
                            *bwd_internal_weights_memory_, *bwd_internal_src_diff_memory_));
    if (b != NULL) {
        linear_bwd_weights_.reset(new inner_product_backward_weights(*linear_bwd_weights_pd_, *bwd_internal_src_memory_,
                            *bwd_internal_dst_diff_memory_, *bwd_internal_weights_diff_memory_,
                            *bwd_internal_bias_diff_memory_));
    } else {
        linear_bwd_weights_.reset(new inner_product_backward_weights(*linear_bwd_weights_pd_, *bwd_internal_src_memory_,
                            *bwd_internal_dst_diff_memory_, *bwd_internal_weights_diff_memory_));
    }
    // create data liunear bwd stream (gx = gy dot W)
    if (is_weights_reordered) {
        this->bwd_data_primitives_.push_back(bwd_reorder_weights_); }
    if (is_dst_diff_reordered) {
        this->bwd_data_primitives_.push_back(bwd_reorder_dst_diff_);
    }
    this->bwd_data_primitives_.push_back(*linear_bwd_data_);
    if (is_src_diff_reordered) {
        this->bwd_data_primitives_.push_back(bwd_reorder_src_diff_);
    }
    // create weight linear bwd stream (gW = gy dot x)
    if (is_src_reordered) {
        this->bwd_weights_primitives_.push_back(bwd_reorder_src_);
    }
    if (is_dst_diff_reordered) {
        this->bwd_weights_primitives_.push_back(bwd_reorder_dst_diff_);
    }
    this->bwd_weights_primitives_.push_back(*linear_bwd_weights_);
    if (is_weights_diff_reordered) {
        this->bwd_weights_primitives_.push_back(bwd_reorder_weights_diff_);
    }
    this->backward_first_use_ = true;
    return 0;
}

template <typename T>
int MKLDNNLinear<T>::backward(T* x,  int x_d1, int x_d2,
                              T* W,  int W_d1, int W_d2,
                              T* b,  int b_d1,
                              T* gy, int gy_d1, int gy_d2,
                              T* gW, int gW_d1, int gW_d2,

                              T* gx, int gx_d1, int gx_d2,
                              T* gb, int gb_d1)
{
    //LOG(INFO) <<"Linear backward with bias";
    if (linear_bwd_data_pd_ == NULL) {
        setup_backward(x,  x_d1,  x_d2,
                       W,  W_d1,  W_d2,
                       b,  b_d1,
                       gy, gy_d1, gy_d2,
                       gW, gW_d1, gW_d2,
                       gx, gx_d1, gx_d2,
                       gb, gb_d1);
    }
    user_src_memory_->set_data_handle(x);
    user_weights_memory_->set_data_handle(W);
    user_dst_diff_memory_->set_data_handle(gy);
    user_weights_diff_memory_->set_data_handle(gW);
    user_src_diff_memory_->set_data_handle(gx);
    user_bias_diff_memory_->set_data_handle(gb);

    if (this->backward_first_use_) {
        LOG(INFO) << "linear backward first use";
        this->bwd_weights_stream_->submit(this->bwd_weights_primitives_).wait();
        this->bwd_data_stream_->submit(this->bwd_data_primitives_).wait();
        this->backward_first_use_ = false;
    } else {
        this->bwd_weights_stream_->rerun().wait();
        this->bwd_data_stream_->rerun().wait();
    }
    return 0;
}

template <typename T>
int MKLDNNLinear<T>::backward(T* x,  int x_d1, int x_d2,
                              T* W,  int W_d1, int W_d2,
                              T* gy, int gy_d1, int gy_d2,
                              T* gW, int gW_d1, int gW_d2,
                              T* gx, int gx_d1, int gx_d2)
{
    //LOG(INFO) <<"Linear backward with bias";

    if (linear_bwd_data_pd_ == NULL) {
        setup_backward(x,  x_d1,  x_d2,
                      W,  W_d1,  W_d2,
                      NULL,  -1,
                      gy, gy_d1, gy_d2,
                      gW, gW_d1, gW_d2,
                      gx, gx_d1, gx_d2,
                      NULL, -1);
    }
 
    user_src_memory_->set_data_handle(x);
    user_weights_memory_->set_data_handle(W);
    user_dst_diff_memory_->set_data_handle(gy);
    user_weights_diff_memory_->set_data_handle(gW);
    user_src_diff_memory_->set_data_handle(gx);

    if (this->backward_first_use_) {
        LOG(INFO) << "linear backward first use";
        this->bwd_weights_stream_->submit(this->bwd_weights_primitives_).wait();
        this->bwd_data_stream_->submit(this->bwd_data_primitives_).wait();
        this->backward_first_use_ = false;
    } else {
        this->bwd_weights_stream_->rerun().wait();
        this->bwd_data_stream_->rerun().wait();
    }

    return 0;
}

template <typename T>
int MKLDNNLinear<T>::forward(T* x, int x_d1, int x_d2,
                             T* W, int W_d1, int W_d2,
                             T* b, int b_d1,
                             T* y, int y_d1, int y_d2)
{
    //LOG(INFO) << "Linear forward";
    //LOG(INFO) << "x = (" << x_d1 << "," << x_d2 << ")";
    //LOG(INFO) << "W = (" << W_d1 << "," << W_d2 << ")";
    //LOG(INFO) << "b = (" << b_d1 << ")";
    //LOG(INFO) << "y = (" << y_d1 << "," << y_d2 << ")";
    if (linear_fwd_pd_ == NULL) {
        setup_forward(x, x_d1, x_d2,
                      W, W_d1, W_d2,
                      b, b_d1,
                      y, y_d1, y_d2);
    }
    user_src_memory_->set_data_handle(x);
    user_weights_memory_->set_data_handle(W);
    user_bias_memory_->set_data_handle(b);
    user_dst_memory_->set_data_handle(y);

    if (this->forward_first_use_) {
        LOG(INFO) << "linear forward first use";
        this->forward_stream_->submit(this->forward_primitives_).wait();
        this->forward_first_use_ = false;
    } else {
        this->forward_stream_->rerun().wait();
    }
    return 0;
}

template <typename T>
int MKLDNNLinear<T>::forward(T* x, int x_d1, int x_d2,
                             T* W, int W_d1, int W_d2,
                             T* y, int y_d1, int y_d2)
{
    //LOG(INFO) << "Linear forward";
    //LOG(INFO) << "x = (" << x_d1 << "," << x_d2 << ")";
    //LOG(INFO) << "W = (" << W_d1 << "," << W_d2 << ")";
    //LOG(INFO) << "y = (" << y_d1 << "," << y_d2 << ")";

    if (linear_fwd_pd_ == NULL) {
        setup_forward(x, x_d1, x_d2,
                      W, W_d1, W_d2,
                      NULL, -1,
                      y, y_d1, y_d2);
    }
    user_src_memory_->set_data_handle(x);
    user_weights_memory_->set_data_handle(W);
    user_dst_memory_->set_data_handle(y);
 
    if (this->forward_first_use_) {
        //LOG(INFO) << "linear forward first use";
        this->forward_stream_->submit(this->forward_primitives_).wait();
        this->forward_first_use_ = false;
    } else {
        this->forward_stream_->rerun().wait();
    }
    return 0;
}


template class MKLDNNLinear<float>;



// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s