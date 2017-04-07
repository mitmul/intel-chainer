#ifndef _LAYER_H_
#define _LAYER_H_

#include <mkldnn.hpp>
#include <vector>

template <typename T>
class Layer {
public:
    virtual ~Layer() {}
    virtual int forward(){ return 0; };
    virtual int backward(){ return 0; };

    virtual int setup_forward(){ return 0; };
    virtual int setup_backward(){ return 0; };

protected:
    mkldnn::stream* forward_stream_;
    mkldnn::stream* backward_stream_;
    std::vector<mkldnn::primitive> forward_primitives_;
    std::vector<mkldnn::primitive> backward_primitives_;
    bool forward_first_use_ = true;
    bool backward_first_use_ = true;
    bool backward_first_setup_ = true;
};

#endif // _LAYER_H_


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s