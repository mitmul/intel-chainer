#ifndef _MKLDNN_SUM_H
#define _MKLDNN_SUM_H

#include <glog/logging.h>
#include <iostream>
#include <mkldnn.hpp>
#include <vector>
#include "layer.h"
#include "layer_factory.h"

template <typename T>
class Sum : public Layer<T> {
public:
    struct sum_data {
        T* data;
        mkldnn::memory::dims dims;    
    };

    Sum<T>();
    ~Sum<T>();
       
    void sum_setup(int num_sum, sum_data* sum_input,
            T* y, int y_d1, int y_d2, int y_d3, int y_d4); 

    void sum(int num_sum, char** data, int* n, int* c, int* h, int* w,
            T* y, int y_d1, int y_d2, int y_d3, int y_d4);

private:
       bool first_run_ = true;

       std::shared_ptr<mkldnn::stream> sum_stream_;
       std::vector<mkldnn::memory::primitive_desc> srcs_mem_pd_;
       std::vector<std::shared_ptr<mkldnn::memory>> srcs_mem_;
       std::vector<mkldnn::primitive::at> srcs_mem_at_;
       std::vector<double> scale_;

       std::shared_ptr<mkldnn::memory::desc> user_dst_md_;
       std::shared_ptr<mkldnn::memory> user_dst_mem_;
       std::shared_ptr<mkldnn::memory> dst_mem_;
       mkldnn::primitive sum_reorder_dst_;

       std::shared_ptr<mkldnn::sum::primitive_desc> sum_pd_;
       std::shared_ptr<mkldnn::sum> sum_prim_;
       std::vector<mkldnn::primitive> sum_prims_;
};

#endif  // _MKLDNN_SUM_H
