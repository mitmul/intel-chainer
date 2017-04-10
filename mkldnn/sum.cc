#include <glog/logging.h>
#include <iostream>
#include "common.h"
#include "mkldnn.hpp"
#include "sum.h"
#include "utils.h"

using namespace mkldnn;
using namespace std;

extern engine cpu_engine;

template<typename T>
Sum<T>::Sum() {
    sum_stream_.reset(new stream(stream::kind::eager));
}

template<typename T>
Sum<T>::~Sum() {
}

template<typename T>
void Sum<T>::sum_setup(int num_sum, Sum<T>::sum_data* sum_input,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4) {
    LOG(INFO) << "Enter sum forward_setup";  
    LOG(INFO) << "y_d1=" << y_d1 << "; y_d2=" << y_d2 << "; y_d3="<<y_d3 << "; y_d4=" << y_d4;
    memory::dims output_tz = {y_d1, y_d2, y_d3, y_d4};
    memory::format src_mfmt = memory::format::nchw;

      for (int i = 0; i < num_sum; i++) {
          memory::dims input_tz = sum_input[i].dims;

          // input should have same dim as output
          assert(output_tz == input_tz);

          // sum primitive doesn't expose API to get inputs' internal format
          // so here always set input internal format as nchw
          shared_ptr<memory::primitive_desc> input_mem_pd;
          input_mem_pd.reset(new memory::primitive_desc(
                      {input_tz, memory_data_type<T>(), memory::format::nchw}, cpu_engine));
          srcs_mem_pd_.push_back(*input_mem_pd);
          scale_.push_back((double)1.0);

          shared_ptr<memory> input_mem;
          input_mem.reset(new memory({{{input_tz}, memory_data_type<T>(), src_mfmt}, cpu_engine}, dummy));
          srcs_mem_.push_back(input_mem);
          srcs_mem_at_.push_back(*srcs_mem_[i]);
      }
        
      //create user dst memory prim/desc
      user_dst_md_.reset(new memory::desc(output_tz, memory_data_type<T>(),memory::format::any));
      user_dst_mem_.reset(new memory({{{output_tz}, memory_data_type<T>(), memory::format::nchw}, cpu_engine}, dummy));
    
      sum_pd_.reset(new mkldnn::sum::primitive_desc(*user_dst_md_, scale_,  srcs_mem_pd_));



      /*
       * Check whether need to reorder for dst mem
       */   
      dst_mem_ = user_dst_mem_;
      bool reorder_sum_dst = false;
      if (sum_pd_.get()->dst_primitive_desc()
              != user_dst_mem_.get()->get_primitive_desc()) {
          LOG(INFO) << "sum reorder dst memory";
          dst_mem_.reset(
                  new memory(sum_pd_.get()->dst_primitive_desc()));
          sum_reorder_dst_ = reorder(*dst_mem_, *user_dst_mem_);
          reorder_sum_dst = true;
      }

      sum_prim_.reset(new mkldnn::sum(*sum_pd_, srcs_mem_at_, *dst_mem_));

      sum_prims_.push_back(*sum_prim_);
      if (reorder_sum_dst) {
          sum_prims_.push_back(sum_reorder_dst_);
      }

}

template<typename T>
void Sum<T>::sum(int num_sum, char** data, int* n, int* c, int* h, int* w,
        T* y, int y_d1, int y_d2, int y_d3, int y_d4) {

    sum_data sum_input[num_sum];
    for (int i = 0; i < num_sum; i++) {
        sum_input[i].data = (T*)data[i];
        sum_input[i].dims = {n[i], c[i], h[i], w[i]};
    }

    if (sum_prim_ == NULL) {
        sum_setup(num_sum, sum_input, 
                y, y_d1, y_d2, y_d3, y_d4);
    }

    /*
     * set mem data handle for input memory
     */
    for (int i = 0; i < num_sum; i++) {
        srcs_mem_[i]->set_data_handle(sum_input[i].data);
    }

    /* set mem handle for dst mem */
    user_dst_mem_->set_data_handle(y);

    if (first_run_) {
        sum_stream_->submit(sum_prims_).wait();
        first_run_ = false;
    } else {
        sum_stream_->rerun().wait();
    }
}

template class Sum<float>;
template class Sum<double>;
