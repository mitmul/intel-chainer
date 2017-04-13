%module mkldnn

%{
    #define SWIG_FILE_WITH_INIT
    #include "common.h"
    #include "layer_factory.h"
    #include "layer.h"
    #include "linear.h"
    #include "pooling.h"
    #include "max_pooling.h"
    #include "avg_pooling.h"
    #include "conv.h"
    #include "relu4d.h"
    #include "relu.h"
    #include "softmax.h"
    #include "lrn.h"
    #include "softmax_cross_entropy.h"
    #include "concat.h"
    #include "sum.h"
%}

%include "numpy.i"

%init %{
    import_array();
    global_init();
%}

%apply ( float* IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* x, int x_d1, int x_d2, int x_d3, int x_d4 )}
%apply ( float* INPLACE_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* gx, int gx_d1, int gx_d2, int gx_d3, int gx_d4 )}
%apply ( float* IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* W, int W_d1, int W_d2, int W_d3, int W_d4 )}
%apply ( float* INPLACE_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* gW, int gW_d1, int gW_d2, int gW_d3, int gW_d4 )}
%apply ( float* IN_ARRAY1, int DIM1 )
    {( float* b, int b_d1 ),
     /* Softmax_2D/4D, Ravel 2D or 4D nparray to 1D to unify interface */
     ( float* x, int dummy_x )}
%apply ( float* INPLACE_ARRAY1, int DIM1 )
    {( float* gb, int gb_d1 ),
     ( float* y, int dummy_y ),
     ( float* gx, int dummy_gx )}
%apply ( int* IN_ARRAY1, int DIM1 )
    {( int* dims, int ndim ),
     ( int* label, int nlabel )}
%apply ( float* INPLACE_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* y, int y_d1, int y_d2, int y_d3, int y_d4 )}
%apply ( float* IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* gy, int gy_d1, int gy_d2, int gy_d3, int gy_d4 )}
    /* Relu_1D interface */
%apply ( float* IN_ARRAY1, int DIM1)
    {( float* x, int x_d1)}
%apply ( float* INPLACE_ARRAY1, int DIM1)
    {( float* y, int y_d1)}
%apply ( float* IN_ARRAY1, int DIM1)
    {( float* gy, int gy_d1)}
%apply ( float* INPLACE_ARRAY1, int DIM1)
    {( float* gx, int gx_d1)}
%apply ( int* IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( int* ws, int ws_d1, int ws_d2, int ws_d3, int ws_d4 )}
%apply ( float* IN_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4 )
    {( float* ws, int ws_d1, int ws_d2, int ws_d3, int ws_d4 )}
    /* linear_2D interface*/
%apply ( float* IN_ARRAY2, int DIM1, int DIM2 )
    {( float* x, int x_d1, int x_d2 )}
%apply ( float* IN_ARRAY2, int DIM1, int DIM2 )
    {( float* gx, int gx_d1, int gx_d2 )}
%apply ( float* IN_ARRAY2, int DIM1, int DIM2 )
    {( float* W, int W_d1, int W_d2 )}
%apply ( float* IN_ARRAY2, int DIM1, int DIM2 )
    {( float* gW, int gW_d1, int gW_d2 )}
%apply ( float* IN_ARRAY1, int DIM1)
    {( float* b, int b_d1)}
%apply ( float* IN_ARRAY1, int DIM1)
    {( float* gb, int gb_d1)}
%apply ( float* INPLACE_ARRAY2, int DIM1, int DIM2 )
    {( float* y, int y_d1, int y_d2 )}
%apply ( float* INPLACE_ARRAY2, int DIM1, int DIM2 )
    {( float* gy, int gy_d1, int gy_d2 )}
%apply ( int N, double K, double ALPHA, double BETA )
    {( int n, double k, double alpha, double beta )}
%apply ( float* IN_ARRAY1, int DIM1)
    {( float* ws, int ws_d)}

%include "common.h"
%include "layer_factory.h"
%include "layer.h"
%include "linear.h"
%include "pooling.h"
%include "max_pooling.h"
%include "avg_pooling.h"
%include "conv.h"
%include "relu4d.h"
%include "relu.h"
%include "softmax.h"
%include "lrn.h"
%include "softmax_cross_entropy.h"
%include "concat.h"
%include "sum.h"

/*
* Support Concat to get a variable size tuple
* in: input tuple
* num_concats: length of tuple
* concat data:
*/
/* Get the ndarray tuple */
%typemap(in) (int num_concats, char** data, int* n, int* c, int* h, int *w) {
    int i;
    if (!PyTuple_Check($input)) {
        PyErr_SetString(PyExc_ValueError, "Expecting a Tuple");
        return NULL;
    }

    $1 = PyTuple_Size($input);
    /* malloc the concat data struct */
    $2 = (char**)malloc(($1)*sizeof(char*));
    $3 = (int*)malloc(($1)*sizeof(int));
    $4 = (int*)malloc(($1)*sizeof(int));
    $5 = (int*)malloc(($1)*sizeof(int));
    $6 = (int*)malloc(($1)*sizeof(int));
    for (i = 0; i < $1; i++) {
        PyObject* x = PyTuple_GetItem($input, i);
        if (!PyArray_Check(x)) {
            PyErr_SetString(PyExc_ValueError, "Item must be array");
            return NULL;
        }
        void* data= array_data(x);
        ($2)[i] = (char*)data;
        int ndims = array_numdims(x);
        if (ndims != 4) {
            PyErr_SetString(PyExc_ValueError, "Only support 4 dimensions now");
            return NULL;
        }
        npy_intp *dims = array_dimensions(x);
        ($3)[i] = dims[0];
        ($4)[i] = dims[1];
        ($5)[i] = dims[2];
        ($6)[i] = dims[3];
    }
}
/* free the list*/
%typemap(freearg) (int num_concats, char** data, int* n, int* c, int* h, int* w) {
    free($2);
    free($3);
    free($4);
    free($5);
    free($6);
}
%apply (int num_concats, char** data, int* n, int* c, int* h, int* w) {
    (int num_concats, char** data, int* n, int* c, int* h, int* w),
    (int num_sum, char** data, int* n, int* c, int* h, int* w)
}

%template(Layer_F32) Layer<float>;
%template(Convolution2D_F32) Convolution2D<float>;
%template(Pooling_F32) Pooling<float>;
%template(MaxPooling_F32) MaxPooling<float>;
%template(Layer_F32) Layer<float>;
%template(Relu4D_F32) Relu4D<float>;
%template(Relu_F32) Relu<float>;
%template(AvgPooling_F32) AvgPooling<float>;
%template(Softmax_F32) Softmax<float>;
%template(LocalResponseNormalization_F32) LocalResponseNormalization<float>;
%template(Linear_F32) MKLDNNLinear<float>;
%template(SoftmaxCrossEntropy_F32) SoftmaxCrossEntropy<float>;
%template(Concat_F32) Concat<float>;
%template(Sum_F32) Sum<float>;
