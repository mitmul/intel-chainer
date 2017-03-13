%module mkldnnpy

%{
    #define SWIG_FILE_WITH_INIT
    #include "common.h"
    #include "convolution.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply ( float* IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* x, int x_d1, int x_d2, int x_d3, int x_d4 )}
%apply ( float* IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* W, int W_d1, int W_d2, int W_d3, int W_d4 )}
%apply ( float* IN_ARRAY1, int DIM1)
    {( float* b, int b_d1)}
%apply ( float* INPLACE_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* y, int y_d1, int y_d2, int y_d3, int y_d4 )}

%include "common.h"
%include "convolution.h"

%template(Convolution2D_F32) Convolution2D<float>;
