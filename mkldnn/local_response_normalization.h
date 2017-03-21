#pragma once
#ifndef _LRN_H_
#define _LRN_H_


#include <mkldnn.hpp>
#include <vector>
#include <memory>

struct lrn_params {
  double alpha, beta;
  int local_size;
  mkldnn::prop_kind aprop_kind;
  mkldnn::algorithm aalgorithm;
  mkldnn::memory::format data_format;
  mkldnn::memory::format diff_data_format;
};

template <typename T>
class LocalResponseNormalization {
public:

    LocalResponseNormalization();
    ~LocalResponseNormalization();

    LocalResponseNormalization(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                              T* y, int y_d1, int y_d2, int y_d3, int y_d4,
                              int n, int k, double alpha, double beta);
    int forward();
    int backward();
private:
    lrn_params p;
    std::shared_ptr<mkldnn::memory> src;
    std::shared_ptr<mkldnn::memory> dst;
    std::shared_ptr<mkldnn::memory> diff_src;
    std::shared_ptr<mkldnn::memory> diff_dst;
    std::shared_ptr<mkldnn::memory> workspace;
    std::shared_ptr<mkldnn::memory::desc> src_desc;
    std::shared_ptr<mkldnn::memory::desc> dst_desc;
    std::shared_ptr<mkldnn::memory::desc> diff_src_desc;
    std::shared_ptr<mkldnn::memory::desc> diff_dst_desc;
    std::shared_ptr<mkldnn::lrn_forward::primitive_desc> lrn_fwd_prim_desc;
    std::shared_ptr<mkldnn::lrn_forward::primitive_desc> lrn_bwd_prim_desc;
    std::shared_ptr<mkldnn::engine> eng;
    bool is_training;
    mkldnn::memory::dims lrn_src_tz;
    mkldnn::memory::dims lrn_dst_tz;

    // mkldnn::stream* stream_;
    // std::vector<mkldnn::primitive> primitives_;

};

#endif // _LRN_H_