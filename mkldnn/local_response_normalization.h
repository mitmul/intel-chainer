#pragma once
#ifndef _LRN_H_
#define _LRN_H_


#include <mkldnn.hpp>
#include <vector>
#include <memory>

struct lrn_params {
  double alpha, beta,k;
  int local_size;
  mkldnn::prop_kind aprop_kind;
  mkldnn::algorithm aalgorithm;
  mkldnn::memory::format data_format;
  mkldnn::memory::format diff_data_format;
};

template <typename T>
class LocalResponseNormalization {
public:

    LocalResponseNormalization(int n, double k, double alpha, double beta);
    ~LocalResponseNormalization();

    LocalResponseNormalization(
      T* x, int x_d1, int x_d2, int x_d3, int x_d4,
      T* y, int y_d1, int y_d2, int y_d3, int y_d4,
      int n, double k, double alpha, double beta);

    int forward_setup(
      T* x, int x_d1, int x_d2, int x_d3, int x_d4,
      T* y, int y_d1, int y_d2, int y_d3, int y_d4);
    void fwd_reset_mem(T* x,T* y);
    int forward(
      T* x, int x_d1, int x_d2, int x_d3, int x_d4,
      T* y, int y_d1, int y_d2, int y_d3, int y_d4);

    int forward();
    int backward_setup(
      T* x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
      T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
      T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4);
    void bwd_reset_mem(T* x,T* gy,T* gx);
    int backward(
      T* x,  int x_d1,  int x_d2,  int x_d3,  int x_d4,
      T* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4,
      T* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4);
private:
    lrn_params p;

    //forward
    std::shared_ptr<mkldnn::memory> lrn_fwd_user_src_mem_, lrn_fwd_dst_mem_, lrn_y_mem_;
    std::shared_ptr<mkldnn::memory::desc> lrn_fwd_src_md_;
    std::shared_ptr<mkldnn::lrn_forward::desc> lrn_fwd_desc_;
    std::shared_ptr<mkldnn::lrn_forward::primitive_desc> lrn_fwd_pd_;
    std::shared_ptr<mkldnn::lrn_forward> lrn_fwd_;
    std::shared_ptr<mkldnn::stream> fwd_stream_;
    std::vector<mkldnn::primitive> fwd_primitives_;

    //backward
    std::shared_ptr<mkldnn::memory> lrn_bwd_user_src_mem_,lrn_diff_src_mem_, lrn_diff_dst_mem_;
    std::shared_ptr<mkldnn::memory::desc> lrn_bwd_src_desc,lrn_diff_src_desc,lrn_diff_dst_desc;
    std::shared_ptr<mkldnn::lrn_backward::desc> lrn_bwd_desc_;
    std::shared_ptr<mkldnn::lrn_backward::primitive_desc> lrn_bwd_pd_;
    std::shared_ptr<mkldnn::lrn_backward> lrn_bwd_;
    std::shared_ptr<mkldnn::stream> bwd_stream_;
    std::vector<mkldnn::primitive> bwd_primitives_;

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
    // mkldnn::memory::dims lrn_src_tz;
    // mkldnn::memory::dims lrn_dst_tz;

    // mkldnn::stream* stream_;
    // std::vector<mkldnn::primitive> primitives_;
    mkldnn::primitive                         reorder_y_;

};

#endif // _LRN_H_