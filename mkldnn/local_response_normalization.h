#pragma once
#ifndef _LRN_H_
#define _LRN_H_

#include <mkldnn.hpp>
#include <vector>

template <typename T>
class LocalResponseNormalization {
public:
    LocalResponseNormalization(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                              T* y, int y_d1, int y_d2, int y_d3, int y_d4,
                              int n, int k, 
                              double alpha, 
                              double beta);

    int forward();
    int backward();
#endif

private:

    struct lrn_params {
      double alpha, beta;
      int local_size;
      memory::format data_format;
      prop_kind aprop_kind;
      algorithm aalgorithm;
      memory::format data_format;
      memory::format diff_data_format;
      // memory::format diff_data_format;
      // int kind; // 0 ac, 1 wc
    };
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;
    std::shared_ptr<memory> diff_src;
    std::shared_ptr<memory> diff_dst;
    std::shared_ptr<memory> workspace;
    std::shared_ptr<memory::desc> src_desc;
    std::shared_ptr<memory::desc> dst_desc;
    std::shared_ptr<memory::desc> diff_src_desc;
    std::shared_ptr<memory::desc> diff_dst_desc;
    std::shared_ptr<lrn_forward::primitive_desc> lrn_fwd_prim_desc;
    std::shared_ptr<lrn_forward::primitive_desc> lrn_bwd_prim_desc;
    lrn_params p;
    memory::dims padR;
    std::shared_ptr<engine> eng;
    memory::data_type data_type;
    bool is_training;

    // mkldnn::stream* stream_;
    // std::vector<mkldnn::primitive> primitives_;

};

#endif // _LRN_H_