#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include <memory>
#include "linear.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
MKLDNNLinear<T>::MKLDNNLinear()
{
    fwd_stream_.reset(new stream(stream::kind::eager));
    bwd_data_stream_.reset(new stream(stream::kind::eager));
    bwd_weights_stream_.reset(new stream(stream::kind::eager));
}

template<typename T>
MKLDNNLinear<T>::~MKLDNNLinear()
{
    
}

template <typename T>
void MKLDNNLinear<T>::forward_setup(T* x, int x_d1, int x_d2, //x_d1 = n, x_d2 = ic  ----- input
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
    std::shared_ptr<inner_product_forward::desc> linearFwd_desc;
    if (b != NULL) {
        linearFwd_desc.reset(new inner_product_forward::desc(prop_kind::forward, init_src_md, init_weights_md,
                                                        init_bias_md, init_dst_md));
    } else {
        linearFwd_desc.reset(new inner_product_forward::desc(prop_kind::forward, init_dst_md, init_weights_md,
                                                        init_dst_md));
    }
    //------Determing engine to use ----------
    //Current, treat the engine is MKLDNN:CPU
    linearFwd_pd.reset(new inner_product_forward::primitive_desc(*linearFwd_desc, cpu_engine));

    //Create user memory primitive 
    fwd_user_src_memory_.reset(new memory({{{src_tz}, mpcsn, memory::format::nc}, cpu_engine}, x));
    fwd_user_weights_memory_.reset(new memory({{{weights_tz}, mpcsn, memory::format::oi}, cpu_engine}, W));
    if (b != NULL)
        fwd_user_bias_memory_.reset(new memory({{{bias_tz}, mpcsn, memory::format::x}, cpu_engine}, b));
    /* in current design, output is also allocated in python part */
    fwd_user_dst_memory_.reset(new memory({{{dst_tz}, mpcsn, memory::format::nc}, cpu_engine}, y));
    

    //create mkldnn memory primitive descripor
    fwd_internal_src_memory_ = fwd_user_src_memory_;
    fwd_internal_weights_memory_ = fwd_user_weights_memory_;
    fwd_internal_bias_memory_ = fwd_user_bias_memory_;
    fwd_internal_dst_memory_ = fwd_user_dst_memory_;
    
   //create reoder primitve if needed
    mkldnn::primitive linear_reorder_src;
    mkldnn::primitive linear_reorder_weights;
    mkldnn::primitive linear_reorder_dst;
    bool is_src_reordered = false;
    bool is_weights_reordered = false;
    bool is_dst_reordered = false;
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc
    /* create reorder primitives between user src and internal src if required */
    if ((*fwd_user_src_memory_).get_primitive_desc() != MemPD(linearFwd_pd.get()->src_primitive_desc())) {
       fwd_internal_src_memory_.reset(new memory(linearFwd_pd.get()->src_primitive_desc()));
       linear_reorder_src = reorder(*fwd_user_src_memory_, *fwd_internal_src_memory_);
       is_src_reordered = true;
    }
    
    /* create reorder primitives between user weights and internal weights if required */
    if ((*fwd_user_weights_memory_).get_primitive_desc() != MemPD(linearFwd_pd.get()->weights_primitive_desc())) {
       fwd_internal_weights_memory_.reset(new memory(linearFwd_pd.get()->weights_primitive_desc()));
       linear_reorder_weights = reorder(*fwd_user_weights_memory_, *fwd_internal_weights_memory_);
       is_weights_reordered = true;
    }
 
    /* create reorder primitives between user dst and internal dst if required */
    if ((*fwd_user_dst_memory_).get_primitive_desc() != MemPD(linearFwd_pd.get()->dst_primitive_desc())) {
       fwd_internal_dst_memory_.reset(new memory(linearFwd_pd.get()->dst_primitive_desc()));
       linear_reorder_dst = reorder(*fwd_internal_dst_memory_, *fwd_user_dst_memory_);
       is_dst_reordered = true;
    }
    std::shared_ptr<mkldnn::primitive> linear_fwd_;
    if (b != NULL)
        linear_fwd_.reset(new inner_product_forward(*linearFwd_pd, *fwd_internal_src_memory_, *fwd_internal_weights_memory_, *fwd_internal_bias_memory_, *fwd_internal_dst_memory_));
    else
        linear_fwd_.reset(new inner_product_forward(*linearFwd_pd, *fwd_internal_src_memory_, *fwd_internal_weights_memory_, *fwd_internal_dst_memory_));
    if (is_src_reordered)
        fwd_primitives_.push_back(linear_reorder_src);
    if (is_weights_reordered)
        fwd_primitives_.push_back(linear_reorder_weights);
    fwd_primitives_.push_back(*linear_fwd_);
    if (is_dst_reordered)
        fwd_primitives_.push_back(linear_reorder_dst);
}
template <typename T>
void MKLDNNLinear<T>::backward_setup(T* x,  int x_d1, int x_d2,
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
    memory::desc init_bias_md({bias_tz}, mpcsn, mfmt);
    memory::desc init_dst_md({dst_tz}, mpcsn, mfmt);
    
    // Initialze linear primitive descriptor
    std::shared_ptr<inner_product_backward_data::desc> linearBwdData_desc;
    std::shared_ptr<inner_product_backward_weights::desc> linearBwdWeights_desc;
    
    linearBwdData_desc.reset(new inner_product_backward_data::desc(init_src_md, init_weights_md, init_dst_md));
    linearBwdWeights_desc.reset(new inner_product_backward_weights::desc(init_src_md, init_weights_md, init_bias_md, init_dst_md));
    //-----Determining engine to use-----------------------
    //Current engine is MKLDNN:CPU
    linearBwdData_pd.reset(new inner_product_backward_data::primitive_desc(*linearBwdData_desc,
                cpu_engine, *linearFwd_pd));
    linearBwdWeights_pd.reset(new inner_product_backward_weights::primitive_desc(*linearBwdWeights_desc,
                cpu_engine, *linearFwd_pd));
    //Create user memory primitive
    bwd_user_src_memory_.reset(new memory({{{src_tz}, mpcsn, memory::format::nc}, cpu_engine}, x));
    bwd_user_weights_memory_.reset(new memory({{{weights_tz}, mpcsn, memory::format::oi}, cpu_engine}, W));
    bwd_user_src_diff_memory_.reset(new memory({{{src_tz}, mpcsn, memory::format::nc}, cpu_engine}, gx));
    bwd_user_weights_diff_memory_.reset(new memory({{{weights_tz}, mpcsn, memory::format::oi}, cpu_engine}, gW));
    bwd_user_dst_diff_memory_.reset(new memory({{{dst_tz}, mpcsn, memory::format::nc}, cpu_engine}, gy));
    if (b != NULL)
        bwd_user_bias_diff_memory_.reset(new memory({{{bias_tz}, mpcsn, memory::format::x}, cpu_engine}, b));

    //create internal memory primivive
    bwd_internal_src_memory_ = bwd_user_src_memory_;
    bwd_internal_weights_memory_ = bwd_user_weights_memory_;
    bwd_internal_src_diff_memory_ = bwd_user_src_diff_memory_;
    bwd_internal_weights_diff_memory_ = bwd_user_weights_diff_memory_;
    bwd_internal_dst_diff_memory_ = bwd_user_dst_diff_memory_;
    if (b != NULL) 
        bwd_internal_bias_diff_memory_ = bwd_user_bias_diff_memory_;
    //--------------check reorder-------------------------
    mkldnn::primitive linear_reorder_src;
    mkldnn::primitive linear_reorder_weights;
    mkldnn::primitive linear_reorder_src_diff;
    mkldnn::primitive linear_reorder_weights_diff;
    mkldnn::primitive linear_reorder_dst_diff;
    bool is_src_reordered = false;
    bool is_weights_reordered = false;
    bool is_src_diff_reordered = false;
    bool is_weights_diff_reordered = false;
    bool is_dst_diff_reordered = false;
    typedef typename memory::primitive_desc MemPD; // short name for memory::primitive_desc
    if ((*bwd_user_src_memory_).get_primitive_desc() 
            != MemPD(linearBwdWeights_pd.get()->src_primitive_desc())) {
        // LOG(INFO) << "bwd reorder x";
        bwd_internal_src_memory_.reset(new memory(linearBwdWeights_pd.get()->src_primitive_desc()));
        linear_reorder_src = reorder(*bwd_user_src_memory_, *bwd_internal_src_memory_);
        is_src_reordered = true;
    }
    
    if ((*bwd_user_weights_memory_).get_primitive_desc() 
            != MemPD(linearBwdData_pd.get()->weights_primitive_desc())) {
        //LOG(INFO) << "bwd reorder w";
        bwd_internal_weights_memory_.reset(new memory(linearBwdData_pd.get()->weights_primitive_desc()));
        linear_reorder_weights = reorder(*bwd_user_weights_memory_, *bwd_internal_weights_memory_);
        is_weights_reordered = true;     
    }

    if ((*bwd_user_src_diff_memory_).get_primitive_desc() 
            != MemPD(linearBwdData_pd.get()->diff_src_primitive_desc())) {
        //LOG(INFO) << "bwd reorder gx";
        bwd_internal_src_diff_memory_.reset(new memory(linearBwdData_pd.get()->diff_src_primitive_desc()));
        linear_reorder_src_diff = reorder(*bwd_internal_src_diff_memory_, *bwd_user_src_diff_memory_);
        is_src_diff_reordered = true;
    }

    if ((*bwd_user_weights_diff_memory_).get_primitive_desc() 
            != MemPD(linearBwdWeights_pd.get()->diff_weights_primitive_desc())) {
        //LOG(INFO) << "bwd reorder gw";
        bwd_internal_weights_diff_memory_.reset(new memory(linearBwdWeights_pd.get()->diff_weights_primitive_desc()));
        linear_reorder_weights_diff = reorder(*bwd_internal_weights_diff_memory_, *bwd_user_weights_diff_memory_);
        is_weights_diff_reordered = true;
    }

    if ((*bwd_internal_dst_diff_memory_).get_primitive_desc()
            != MemPD(linearBwdWeights_pd.get()->diff_dst_primitive_desc())) {
        //LOG(INFO) << "bwd reorder gy";
        bwd_internal_dst_diff_memory_.reset(new memory(linearBwdWeights_pd.get()->diff_dst_primitive_desc()));
        linear_reorder_dst_diff = reorder(*bwd_user_dst_diff_memory_, *bwd_internal_dst_diff_memory_);
        is_dst_diff_reordered = true;
    }

    //create linear bwd data primitive  
    std::shared_ptr<mkldnn::primitive> linearBwdData;
    std::shared_ptr<mkldnn::primitive> linearBwdWeights;
    linearBwdData.reset(new inner_product_backward_data(*linearBwdData_pd, *bwd_internal_dst_diff_memory_,
                            *bwd_internal_weights_memory_, *bwd_internal_src_diff_memory_));
    if (b != NULL) {
        linearBwdWeights.reset(new inner_product_backward_weights(*linearBwdWeights_pd, *bwd_internal_src_memory_,
                            *bwd_internal_dst_diff_memory_, *bwd_internal_weights_diff_memory_,
                            *bwd_internal_bias_diff_memory_));
    } else {
        linearBwdWeights.reset(new inner_product_backward_weights(*linearBwdWeights_pd, *bwd_internal_src_memory_,
                            *bwd_internal_dst_diff_memory_, *bwd_internal_weights_diff_memory_));
    }
    // create data liunear bwd stream (gx = gy dot W)
    if (is_weights_reordered) {
        bwd_data_primitives_.push_back(linear_reorder_weights); }
    if (is_dst_diff_reordered) {
        bwd_data_primitives_.push_back(linear_reorder_dst_diff);
    }
    bwd_data_primitives_.push_back(*linearBwdData);
    if (is_src_diff_reordered) {
        bwd_data_primitives_.push_back(linear_reorder_src_diff);
    }
    // create weight linear bwd stream (gW = gy dot x)
    if (is_src_reordered) {
        bwd_weights_primitives_.push_back(linear_reorder_src);
    }
    if (is_dst_diff_reordered) {
        bwd_weights_primitives_.push_back(linear_reorder_dst_diff);
    }
    bwd_weights_primitives_.push_back(*linearBwdWeights);
    if (is_weights_diff_reordered) {
        bwd_weights_primitives_.push_back(linear_reorder_weights_diff);
    }
    return ;
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
    if (linearBwdData_pd == NULL) 
        backward_setup(x,  x_d1,  x_d2,
                       W,  W_d1,  W_d2,
                       b,  b_d1,
                       gy, gy_d1, gy_d2,
                       gW, gW_d1, gW_d2,
                       gx, gx_d1, gx_d2,
                       gb, gb_d1);
    bwd_weights_stream_->submit(bwd_weights_primitives_);
    bwd_data_stream_->submit(bwd_data_primitives_);
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
    if (linearBwdData_pd == NULL) 
        backward_setup(x,  x_d1,  x_d2,
                      W,  W_d1,  W_d2,
                      NULL,  -1,
                      gy, gy_d1, gy_d2,
                      gW, gW_d1, gW_d2,
                      gx, gx_d1, gx_d2,
                      NULL, -1);
    bwd_weights_stream_->submit(bwd_weights_primitives_);
    bwd_data_stream_->submit(bwd_data_primitives_);
    return 0;
}

template <typename T>
int MKLDNNLinear<T>::forward(T* x, int x_d1, int x_d2,
                             T* W, int W_d1, int W_d2,
                             T* b, int b_d1,
                             T* y, int y_d1, int y_d2)
{
    //LOG(INFO) << "Linear forward";
    if (linearFwd_pd == NULL)
        forward_setup(x, x_d1, x_d2,
                      W, W_d1, W_d2,
                      b, b_d1,
                      y, y_d1, y_d2);
    fwd_stream_->submit(fwd_primitives_);
    return 0;
}

template <typename T>
int MKLDNNLinear<T>::forward(T* x, int x_d1, int x_d2,
                             T* W, int W_d1, int W_d2,
                             T* y, int y_d1, int y_d2)
{
    LOG(INFO) << "Linear forward";
    if (linearFwd_pd == NULL)
        forward_setup(x, x_d1, x_d2,
                      W, W_d1, W_d2,
                      NULL, -1,
                      y, y_d1, y_d2);
    fwd_stream_->submit(fwd_primitives_);
    return 0;
}


template class MKLDNNLinear<float>;

