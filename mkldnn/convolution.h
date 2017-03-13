#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include <mkldnn.hpp>
#include <vector>

template <typename T>
class Convolution2D {
public:
    Convolution2D(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                  T* W, int W_d1, int W_d2, int W_d3, int W_d4,
                  T* b, int b_d1,
                  T* y, int y_d1, int y_d2, int y_d3, int y_d4,
                  int s1, int s2,
                  int p1, int p2);
    Convolution2D(T* x, int x_d1, int x_d2, int x_d3, int x_d4,
                  T* W, int W_d1, int W_d2, int W_d3, int W_d4,
                  T* y, int y_d1, int y_d2, int y_d3, int y_d4,
                  int s1, int s2,
                  int p1, int p2);

    int forward();
    int backward();

private:
    mkldnn::stream* stream_;
    std::vector<mkldnn::primitive> primitives_;

};

#endif // _CONVOLUTION_H_
