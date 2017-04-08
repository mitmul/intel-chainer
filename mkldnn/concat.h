#ifndef _MKLDNN_CONCAT_H
#define _MKLDNN_CONCAT_H

#include <glog/logging.h>
#include <iostream>
#include <mkldnn.hpp>
#include <vector>
#include "layer.h"
#include "layer_factory.h"

template <typename T>
class Concat : public Layer<T> {
public:
    struct concat_data {
        T* data;
        mkldnn::memory::dims dims;    
    };

    Concat<T>();
    ~Concat<T>();
       
    void forward_setup(int num_concats, concat_data* concat_input,
            T* y, int y_d1, int y_d2, int y_d3, int y_d4,
            int axis); 

    void forward(int num_concats, char** data, int* n, int* c, int* h, int* w,
            T* y, int y_d1, int y_d2, int y_d3, int y_d4,
            int axis);

    void backward_setup(int num_concats, concat_data* concat_output,
            T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
            int axis); 
    
    void backward(int num_concats, char** data, int* n, int* c, int* h, int* w,
            T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
            int axis);
private:
        
       int axis_;
       bool fwd_first_run_ = true;
       bool bwd_first_run_ = true;

       //forward
       mkldnn::memory::dims output_tz_; //output dims
       std::shared_ptr<mkldnn::memory::primitive_desc> user_dst_mpd_; //user dst mpd
       std::shared_ptr<mkldnn::memory::desc> user_dst_md_; //user dst md
       std::shared_ptr<mkldnn::memory> user_dst_memory_; // usr dst memory
       std::shared_ptr<mkldnn::memory> dst_memory_; // mkldnn dst memory
       std::shared_ptr<mkldnn::stream> fwd_stream_; // fwd stream
       std::vector<mkldnn::primitive> fwd_primitives_; //fwd primitive vector
       std::shared_ptr<mkldnn::concat::primitive_desc> fwd_concat_pd_; //fwd prim desc
       std::shared_ptr<mkldnn::concat> fwd_concat_prim_; // fwd primitive
       mkldnn::primitive concat_reorder_dst_; //reorder y
       std::vector<mkldnn::primitive::at> fwd_input_primitives_at_; //fwd input primitives     
       std::vector<std::shared_ptr<mkldnn::memory>> fwd_input_primitives_; // fwd input memory
       std::vector<mkldnn::memory::primitive_desc> srcs_prim_desc_; //src primitve desc vector



       //backward
       std::shared_ptr<mkldnn::memory::primitive_desc> user_diff_dst_mpd_; //user diff dst mpd
       std::shared_ptr<mkldnn::memory> user_diff_dst_prim_; // usr diff dst memory
       std::vector<mkldnn::memory> bwd_reorder_diff_src_mem_; // diff src memory vectory
       std::vector<mkldnn::reorder> reorders_;
       std::vector<mkldnn::primitive> bwd_primitives_; //bwd primitive vector
       std::shared_ptr<mkldnn::stream> bwd_stream_;
       std::vector<mkldnn::memory::primitive_desc> diff_srcs_prim_desc_; //diff src primitve desc vector
      
        
};

#endif  // _MKLDNN_CONCAT_H


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s