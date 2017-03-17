#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "local_response_normalization.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
LocalResponseNormalization<T>::LocalResponseNormalization()
{

}

template<typename T>
LocalResponseNormalization<T>::~LocalResponseNormalization()
{

}

template<typename T>
LocalResponseNormalization<T>::LocalResponseNormalization(
	T* x, int x_d1, int x_d2, int x_d3, int x_d4,
    T* y, int y_d1, int y_d2, int y_d3, int y_d4,
    int n, int k, double alpha, double beta)
{
    google::SetLogDestination(google::GLOG_INFO,"./lrnMyInfo");

    LOG(INFO) << "LocalResponseNormalization";
    LOG(INFO) << "    xdim=(" << x_d1 << "," << x_d2 << "," << x_d3 << "," << x_d4 << ")";
    LOG(INFO) << "    ydim=(" << y_d1 << "," << y_d2 << "," << y_d3 << "," << y_d4 << ")";
    LOG(INFO) << "    n =(" << n <<  ")";
    LOG(INFO) << "    k =(" << k <<  ")";

	p.alpha = alpha;
	p.beta = beta;
	p.aprop_kind = prop_kind::forward_training;
	p.local_size = 5;
	p.data_format = memory::format::nchw;
	p.diff_data_format = memory::format::nchw;
	p.aalgorithm = algorithm::lrn_across_channels;

	lrn_src_tz = {x_d1, x_d2, x_d3, x_d4};
	lrn_dst_tz = {y_d1, y_d2, y_d3, y_d4};

	eng.reset(new engine(engine::kind::cpu, 0));

	src_desc.reset(new memory::desc({ lrn_src_tz },
	   memory_data_type<T>(), p.data_format));
	dst_desc.reset(new memory::desc({ lrn_dst_tz},
	    memory_data_type<T>(), p.data_format));
	diff_src_desc.reset(new memory::desc({ lrn_src_tz },
	    memory_data_type<T>(), p.diff_data_format));
	diff_dst_desc.reset(new memory::desc({ lrn_dst_tz },
	    memory_data_type<T>(), p.diff_data_format));

	src.reset(new memory({{{lrn_src_tz}, memory_data_type<T>(),p.data_format}, *eng}, x));
	dst.reset(new memory({{{lrn_dst_tz}, memory_data_type<T>(),p.data_format}, *eng}, y));
	diff_src.reset(new memory({{{lrn_src_tz}, memory_data_type<T>(),p.diff_data_format}, *eng}, x));
	diff_dst.reset(new memory({{{lrn_dst_tz},memory_data_type<T>(),p.diff_data_format}, *eng}, y));

	is_training = p.aprop_kind == prop_kind::forward_training;

}


template<typename T>
int LocalResponseNormalization<T>::forward(){
	auto lrn_desc = lrn_forward::desc(p.aprop_kind, p.aalgorithm, *src_desc,
	        p.local_size, p.alpha, p.beta);
	lrn_fwd_prim_desc.reset(new lrn_forward::primitive_desc(lrn_desc, *eng));


	// src.reset(new memory({*src_desc, *eng}));
	// dst.reset(new memory({*dst_desc, *eng}));

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
template<typename T>
int LocalResponseNormalization<T>::backward(){
	auto lrn_desc = lrn_backward::desc(p.aalgorithm,
		*src_desc, *diff_dst_desc,p.local_size, p.alpha, p.beta);

	// diff_src.reset(new memory({*diff_src_desc, *eng}));
	// diff_dst.reset(new memory({*diff_dst_desc, *eng}));
	auto lrn_prim_desc = lrn_backward::primitive_desc(lrn_desc, *eng,
		*lrn_fwd_prim_desc);

	// Execute
	std::vector<primitive> pipeline;
	auto s = stream(stream::kind::lazy);
	auto l = lrn_backward(lrn_prim_desc, *src, *diff_dst, *workspace,
		*diff_src);
	pipeline.push_back(l);
	s.submit(pipeline).wait();

	return 0;
}

template class LocalResponseNormalization<float>;
template class LocalResponseNormalization<double>;