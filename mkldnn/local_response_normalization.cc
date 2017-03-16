#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "local_response_normalization.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename data_type>

#if 0
LocalResponseNormalization<T>::LocalResponseNormalization(
	T* x, int x_d1, int x_d2, int x_d3, int x_d4,
    T* y, int y_d1, int y_d2, int y_d3, int y_d4,
    int n, int k, double alpha, double beta){

	auto cpu_engine = engine(engine::cpu, 0);

    LOG(INFO) << "x =(" << x_d1 << "," << x_d2 << "," << x_d3 << "," << x_d4 << ")";
    LOG(INFO) << "y =(" << y_d1 << "," << y_d2 << "," << y_d3 << "," << y_d4 << ")";
	memory::dims lrn_src_tz = {x_d1, x_d2, x_d3, x_d4};
	memory::dims lrn_dst_tz = {y_d1, y_d2, y_d3, y_d4};
	/* create memory for user data */
    auto lrn_user_src_memory = new memory({{{lrn_src_tz}, memory_data_type<T>(),
        memory::format::nchw}, cpu_engine}, x);
  	auto lrn_dst_memory = new memory({{{lrn_src_tz}, memory_data_type<T>(),
        memory::format::nchw}, cpu_engine}, y);
  	auto lrn_scratch_memory = new memory(lrn_dst_memory.get_primitive_desc());

    /* create memory descriptors*/
    auto lrn_src_md = new memory::desc({lrn_src_tz}, memory_data_type<T>(),
        memory::format::nchw);
    auto lrn_dst_md = new memory::desc({lrn_dst_tz}, mmemory_data_type<T>(),
        memory::format::nchw);

	const uint32_t local_size = 5;

#if 0
 	/* create a lrn primitive descriptor */
    mkldnn_lrn_desc_t lrn_desc;
    mkldnn_lrn_forward_desc_init(&lrn_desc, mkldnn_forward,
                                       mkldnn_lrn_across_channels, lrn_src_md,
                                       local_size,alpha, beta);
    mkldnn_primitive_desc_t lrn_pd;
    mkldnn_primitive_desc_create(&lrn_pd, &lrn_desc, engine, NULL);

    /* create primitives for lrn dst and workspace memory */
    // mkldnn_primitive_t lrn_dst_memory, lrn_workspace_memory;


    mkldnn_primitive_at_t lrn_srcs = { lrn_dst_memory };
    const_mkldnn_primitive_t lrn_dsts[] = { lrn_dst_memory};

    mkldnn_primitive_t lrn;
    mkldnn_primitive_create(&lrn, lrn_pd, &lrn_srcs, lrn_dsts)
#endif

    /* create lrn primitive and add it to net */
    auto lrn_desc = lrn_forward::desc(prop_kind::forward, lrn_across_channels,
                					 lrn_dst_md, local_size,alpha, beta);
    auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, cpu_engine);

    auto lrn_prim = lrn_forward(lrn_prim_desc,lrn_src_md,lrn_dst_memory);
    primitives_.push_back(lrn_prim);
 	// net_.push_back(lrn_forward(lrn_prim_desc, relu_dst_memory,
  //       lrn_scratch_memory, lrn_dst_memory));
    
}
#endif
LocalResponseNormalization<data_type>::LocalResponseNormalization(
	T* x, int x_d1, int x_d2, int x_d3, int x_d4,
    T* y, int y_d1, int y_d2, int y_d3, int y_d4,
    int n, int k, double alpha, double beta){

	p.alpha = alpha;
	p.beta = beta;
	p.aprop_kind = prop_kind::forward_training;
	p.local_size = 5;
	p.data_format = memory::format::nchw;
	p.diff_data_format = memory::format::nchw;
	p.aalgorithm = mkldnn_lrn_across_channels;

	eng.reset(new engine(engine::kind::cpu, 0));
	src_desc.reset(new memory::desc({ x_d1, x_d2, x_d3, x_d4 },
	    data_type, p.data_format));
	dst_desc.reset(new memory::desc({ y_d1, y_d2, y_d3, y_d4 },
	    data_type, p.data_format));
	diff_src_desc.reset(new memory::desc({ x_d1, x_d2, x_d3, x_d4 },
	    data_type, p.diff_data_format));
	diff_dst_desc.reset(new memory::desc({ y_d1, y_d2, y_d3, y_d4 },
	    data_type, p.diff_data_format));

	is_training = p.aprop_kind == prop_kind::forward_training;

}
int LocalResponseNormalization<data_type>::forward(){
	auto lrn_desc = lrn_forward::desc(p.aprop_kind, p.aalgorithm, *src_desc,
	        p.local_size, p.alpha, p.beta);
	lrn_fwd_prim_desc.reset(new lrn_forward::primitive_desc(lrn_desc, *eng));

	src.reset(new memory({*src_desc, *eng}));
	dst.reset(new memory({*dst_desc, *eng}));

	std::vector<primitive> pipeline;
	auto s = stream(stream::kind::lazy);
	if (is_training) {
	    auto workspace_primitive_desc =
	    lrn_fwd_prim_desc->workspace_primitive_desc();
	    workspace.reset(new memory(workspace_primitive_desc));
	    auto l = lrn_forward(*lrn_fwd_prim_desc, *src, *workspace, *dst);
	    pipeline.push_back(l);
	    s.submit(pipeline).wait();
	} else {
	    auto l = lrn_forward(*lrn_fwd_prim_desc, *src, *dst);
	    pipeline.push_back(l);
	    s.submit(pipeline).wait();
	}
	// stream_->submit(primitives_);
	return 0;
}

int LocalResponseNormalization<T>::backward(){
	auto lrn_desc = lrn_backward::desc(p.aalgorithm,
									*src_desc, *diff_dst_desc,
									p.local_size, p.alpha, p.beta);
	diff_src.reset(new memory({*diff_src_desc, *eng}));
	diff_dst.reset(new memory({*diff_dst_desc, *eng}));
	auto lrn_prim_desc = lrn_backward::primitive_desc(lrn_desc, *eng,
									*lrn_fwd_prim_desc);

	// fill_data<data_t>(diff_dst->get_primitive_desc().get_size()
	// / sizeof(data_t), (data_t *)diff_dst->get_data_handle());

	// Execute
	std::vector<primitive> pipeline;
	auto s = stream(stream::kind::lazy);
	auto l = lrn_backward(lrn_prim_desc, *src, *diff_dst, *workspace,
	*diff_src);
	pipeline.push_back(l);
	s.submit(pipeline).wait();

	check_lrn_bwd<data_t>(p, *src, *diff_dst, *diff_src);
	return 0;
}