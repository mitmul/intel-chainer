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
    mkldnn::stream* stream_;
    std::vector<mkldnn::primitive> primitives_;
    bool first_use = true;

};

#endif // _LAYER_H_
