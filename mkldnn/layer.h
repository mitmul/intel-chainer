#ifndef _LAYER_H_
#define _LAYER_H_

#include <mkldnn.hpp>
#include <vector>

template <typename T>
class Layer {
public:
    int forward(){};
    int backward(){};

    int setup_forward(){};
    int setup_backward(){};

    mkldnn::stream* stream_;
    std::vector<mkldnn::primitive> primitives_;

};

#endif // _LAYER_H_
