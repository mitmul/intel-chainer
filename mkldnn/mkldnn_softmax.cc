#include <glog/logging.h>
#include <iostream>
#include <vector>
#include "mkldnn.hpp"
#include "mkldnn_softmax.h"
#include "utils.h"

using namespace mkldnn;

extern engine cpu_engine;

template<typename T>
void Softmax<T>::update_user_mem(T* x, T* y)
{
    src_user = x;
    dst_user = y;
}

enum { SOFTMAX_SRC_USER_DATA_UPDATA = 0, SOFTMAX_DST_USER_DATA_UPDATA};

template<typename T>
void Softmax<T>::update_user_data(T* mem, int mem_type, int size)
{
    switch (mem_type) {
    case SOFTMAX_SRC_USER_DATA_UPDATA:
        memcpy(mem, src_user, size);
	break;
    case SOFTMAX_DST_USER_DATA_UPDATA:
        memcpy(dst_user, mem, size);
	break;
    default:
        LOG(INFO) << "No such mem_type " << mem_type << ".";
	break;
    };
}

// Class Softmax<T>
template<typename T> Softmax<T>*
Softmax<T>::softmax_create_forward(T* x, T* y, int* dims, int ndim, int axis)
{
    Softmax<T>* inst = NULL;

#define SOFTMAX_2D 2
#define SOFTMAX_4D 4
    if (SOFTMAX_2D == ndim) {
	// StreamFactory get map
        if (!(inst =
	    (Softmax_2D<T>*)StreamFactory::getInstance().getSoftmax2DFwdStream(dims[0], dims[1], axis))) {
            // New Softmax inst
            inst = new Softmax_2D<T>(dims, axis);

            // Setup forward
            inst->setup_forward();

	    // StreamFactory set map
            StreamFactory::getInstance().setSoftmax2DFwdStream(dims[0], dims[1], axis, inst);

            LOG(INFO) << "New Softmax2D Instance: " << (void*)&inst << " !!!";
	} else {
            LOG(INFO) << "Reuse Softmax2D Instance: " << (void*)&inst << " !!!";
	}

        // Update current user memory
        // FIXME: Do not drop memory util finishing forward() in Python layer
	inst->update_user_mem(x, y);
    } else if (SOFTMAX_4D == ndim) {
	// StreamFactory get map
        if (!(inst =
	    (Softmax_4D<T>*)StreamFactory::getInstance().getSoftmax4DFwdStream(dims[0], dims[1], dims[2], dims[3], axis))) {
            // New Softmax inst
            inst = new Softmax_4D<T>(dims, axis);

            // Setup forward
            inst->setup_forward();

	    // StreamFactory set map
            StreamFactory::getInstance().setSoftmax4DFwdStream(dims[0], dims[1], dims[2], dims[3], axis, inst);

            LOG(INFO) << "New Softmax4D Instance: " << (void*)&inst << " !!!";
        } else {
            LOG(INFO) << "Reuse Softmax4D Instance: " << (void*)&inst << " !!!";
	}

        // Update current user memory
        // FIXME: Do not drop memory util finishing forward() in Python layer
	inst->update_user_mem(x, y);
    } else {
        ; //Not supported;
    }

    return inst;
};

// Class Softmax_2D<T>
template<typename T>
int Softmax_2D<T>::get_res_size()
{
    return sizeof(T) * dims[0] * dims[1];
}

template<typename T>
int Softmax_2D<T>::setup_forward()
{
    LOG(INFO) << "Softmax_2D::setup_forward";

    // (1) Init persistent memory
    auto src = new T[get_res_size()];;
    auto dst = new T[get_res_size()];;
    memory::dims src_tz = {dims[0], dims[1]};
    memory::dims dst_tz = {dims[0], dims[1]};

    // (2) Prepare user memory primitive and memory desc
    src_mem.reset(new memory({{{src_tz}, memory_data_type<T>(),
                                      memory::format::nc}, cpu_engine}, src));
    dst_mem.reset(new memory({{{dst_tz}, memory_data_type<T>(),
                                      memory::format::nc}, cpu_engine}, dst));

    src_md.reset(new memory::desc({src_tz}, memory_data_type<T>(),
                                   memory::format::nc));

    // (3) Create softmax primitive desc
    softmax_desc.reset(new softmax_forward::desc(prop_kind::forward_scoring, *src_md, axis));
    softmax_pd.reset(new softmax_forward::primitive_desc(*softmax_desc, cpu_engine));

    // (4) Create src primitive and src_reorder primitive
    //
    // (No Reorder in Softmax)
    //
    // primitive *src_mem = user_src_memory, *src_reorder = NULL;
    // if (memory::primitive_desc(fwd_pd->src_primitive_desc()) !=
    //     user_src_memory->get_primitive_desc()){
    //     src_mem = new memory(fwd_pd->src_primitive_desc());
    //     src_reorder = new reorder(*user_src_memory, *src_mem);
    // }

    // (5) Create dst primitive and dst_reorder primitive
    //
    // (No Reorder in Softmax)
    //
    // primitive *dst_mem = user_dst_memory, *dst_reorder = NULL;
    // if (memory::primitive_desc(fwd_pd->dst_primitive_desc()) !=
    //     user_dst_memory->get_primitive_desc()) {
    //     dst_mem = new memory(fwd_pd->dst_primitive_desc());
    //     dst_reorder = new reorder(*dst_mem, *user_dst_memory);
    // }

    // (6) Create softmax primitive
    softmax.reset(new softmax_forward(*softmax_pd, *src_mem, *dst_mem));

    // (7) Construct net
    primitives_.push_back(*softmax);
    stream_.reset(new stream(stream::kind::lazy));

    return 0;
}

template<typename T>
int Softmax_2D<T>::setup_backward()
{
    LOG(INFO) << "Softmax_2D::setup_backward";
    return 0;
}

template<typename T>
int Softmax_2D<T>::forward()
{
    LOG(INFO) << "Softmax_2D::forward";

    // Copy user source data to persistent memory
    this->update_user_data(src, SOFTMAX_SRC_USER_DATA_UPDATA, get_res_size());

    // Submit stream
    stream_->submit(primitives_).wait();

    // Copy softmax outputs (persistent memory) to user destination memory
    this->update_user_data(dst, SOFTMAX_DST_USER_DATA_UPDATA, get_res_size());

    return 0;
}

template<typename T>
int Softmax_2D<T>::backward()
{
    LOG(INFO) << "Softmax_2D::backward";
    return 0;
}

// Class Softmax_4D
template<typename T>
int Softmax_4D<T>::get_res_size()
{
    return sizeof(T) * dims[0] * dims[1] * dims[2] * dims[3];
}

template<typename T>
int Softmax_4D<T>::setup_forward()
{
    LOG(INFO) << "Softmax_4D::setup_forward";
    return 0;
}

template<typename T>
int Softmax_4D<T>::setup_backward()
{
    LOG(INFO) << "Softmax_4D::setup_backward";
    return 0;
}


template<typename T>
int Softmax_4D<T>::forward()
{
    LOG(INFO) << "Softmax_4D::forward";
    return 0;
}

template<typename T>
int Softmax_4D<T>::backward()
{
    LOG(INFO) << "Softmax_4D::backward";
    return 0;
}

template class Softmax<float>;
template class Softmax_2D<float>;
template class Softmax_4D<float>;
