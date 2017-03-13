#include <glog/logging.h>
#include "mkldnn.hpp"

//
// map C type with mkldnn's
// float -> memory::data_type::f32
// int -> memory::data_type::s32
template<typename T>
static mkldnn::memory::data_type memory_data_type() {
    if (typeid(T) == typeid(float))
        return mkldnn::memory::data_type::f32;
    else if (typeid(T) == typeid(int))
        return mkldnn::memory::data_type::s32;

    LOG(ERROR) << "Not support type";
    return mkldnn::memory::data_type::data_undef;
}
