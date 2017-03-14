%module mkldnn

%{
    #define SWIG_FILE_WITH_INIT
    #include "common.h"
    #include "convolution.h"
    #include "stream_factory.h"
%}

%include "numpy.i"

%init %{
    import_array();
    global_init();
%}

%apply ( float* IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* x, int x_d1, int x_d2, int x_d3, int x_d4 )}
%apply ( float* IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4 )}
%apply ( float* IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* W, int W_d1, int W_d2, int W_d3, int W_d4 )}
%apply ( float* IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* gW, int gW_d1, int gW_d2, int gW_d3, int gW_d4 )}
%apply ( float* IN_ARRAY1, int DIM1)
    {( float* b, int b_d1)}
%apply ( float* IN_ARRAY1, int DIM1)
    {( float* gb, int gb_d1)}
%apply ( float* INPLACE_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* y, int y_d1, int y_d2, int y_d3, int y_d4 )}
%apply ( float* INPLACE_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4 )}

%include "common.h"
%include "convolution.h"
%include "stream_factory.h"

%template(Convolution2D_F32) Convolution2D<float>;
