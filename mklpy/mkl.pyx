import numpy as np
cimport numpy as np
cimport cmkl_dnn

# define our C wrapper
cdef extern from "mkl_ops.h": 
    int _conv_forward(float image_pbuf[], 
                     float weight_pbuf[],
                     size_t image_shape[], 
                     size_t weight_shape[],
                     int stride[], 
                     int pad[],
                     float *output)

    int _conv_backward(float image_pbuf[],
                       float weight_pbuf[],
                       float gz_pbuf[],
                       size_t image_shape[],
                       size_t weight_shape[],
                       int stride[],
                       int pad[],
                       float gradImage_pbuf[],
                       float gradWeight_pbuf[],
                       float gradBias_pbuf[])

    int _get_conv_output_shape(int ndim,
                               size_t* image_shape,
                               size_t* weight_shape,
                               int* stride,
                               int* pad,
                               size_t* output_shape)

## forward convolution
def conv_forward( np.ndarray[np.float32_t, mode="c", ndim=4] image_pbuf,
                  np.ndarray[np.float32_t, mode="c", ndim=4] weight_pbuf,
                  np.ndarray[np.float32_t, mode="c", ndim=4] output, 
                  image_shape=None,
                  weight_shape=None,
                  stride=None,
                  pad=None):
    cdef float* c_image
    cdef float* c_weight
    cdef size_t c_image_shape[4]
    cdef size_t c_weight_shape[4]
    cdef int c_stride[2]
    cdef int c_pad[2]

    if image_pbuf is None or weight_pbuf is None or output is None:
        raise ValueError('Input image, weight and gz should not be None.')

    if image_shape is None:
        for i in range(4):
            c_image_shape[i] = image_pbuf.shape[i]
    else:
        assert isinstance(image_shape, (tuple, list))
        assert len(image_shape) == 4
        for i in range(4):
            c_image_shape[i] = image_shape[i]

    if weight_shape is None:
        for i in range(4):
            c_weight_shape[i] = weight_pbuf.shape[i]
    else:
        assert isinstance(weight_shape, (tuple, list))
        assert len(weight_shape) == 4
        for i in range(4):
            c_weight_shape[i] = weight_shape[i]

    c_image = <float*> image_pbuf.data
    c_weight = <float*> weight_pbuf.data

    if stride == None:
        stride = (1, 1)

    if pad == None:
        pad = (0, 0)

    if len(stride) != 2 or len(pad) != 2:
        raise ValueError('Stride and pad should be 2-elements list or tuple.')

    for i in range(2):
        c_stride[i] = stride[i]
        c_pad[i] = pad[i]
        
    conv_f_ret = _conv_forward( <float*> c_image, <float*> c_weight,
                               <size_t*> c_image_shape, <size_t*> c_weight_shape, <int*> c_stride, <int*> c_pad,
                               <float*> output.data )

    return conv_f_ret


## Backward convolution
## By default: stride=(1,1), pad=(0,0)
def conv_backward( np.ndarray[np.float32_t, mode='c', ndim=4] image_pbuf,
                   np.ndarray[np.float32_t, mode='c', ndim=4] weight_pbuf,
                   np.ndarray[np.float32_t, mode='c', ndim=4] gz_pbuf,
                   np.ndarray[np.float32_t, mode='c', ndim=4] gradImage_pbuf,
                   np.ndarray[np.float32_t, mode='c', ndim=4] gradWeight_pbuf,
                   np.ndarray[np.float32_t, mode='c', ndim=1] gradBias_pbuf,
                   image_shape=None,
                   weight_shape=None,
                   stride=None,
                   pad=None):

    cdef float* c_image
    cdef float* c_weight
    cdef float* c_gz
    cdef size_t c_image_shape[4]
    cdef size_t c_weight_shape[4]
    cdef int c_stride[2]
    cdef int c_pad[2]

    if image_pbuf is None or weight_pbuf is None or gz_pbuf is None:
        raise ValueError('Input image, weight and gz should not be None.')

    if image_shape is None:
        for i in range(4):
            c_image_shape[i] = image_pbuf.shape[i]
    else:
        assert isinstance(image_shape, (tuple, list))
        assert len(image_shape) == 4
        for i in range(4):
            c_image_shape[i] = image_shape[i]

    if weight_shape is None:
        for i in range(4):
            c_weight_shape[i] = weight_pbuf.shape[i]
    else:
        assert isinstance(weight_shape, (tuple, list))
        assert len(weight_shape) == 4
        for i in range(4):
            c_weight_shape[i] = weight_shape[i]

    c_image = <float*> image_pbuf.data
    c_weight = <float*> weight_pbuf.data
    c_gz = <float*> gz_pbuf.data

    if stride is None:
        stride = (1, 1)

    if pad is None:
        pad = (0, 0)

    if len(stride) != 2 or len(pad) != 2:
        raise ValueError('Stride and pad should be 2-elements list or tuple.')
    
    for i in range(2):
        c_stride[i] = stride[i]
        c_pad[i] = pad[i]
     
    conv_b_ret = _conv_backward(c_image, c_weight, c_gz,
                                c_image_shape, c_weight_shape, c_stride, c_pad,
                                <float*>gradImage_pbuf.data, <float*>gradWeight_pbuf.data, <float*>gradBias_pbuf.data)

    return conv_b_ret



def get_conv_output_shape(image_shape, weight_shape, stride=None, pad=None):
    cdef int nd
    cdef size_t c_image_shape[4]
    cdef size_t c_weight_shape[4]
    cdef int c_stride[2]
    cdef int c_pad[2]
    
    nd = len(image_shape)

    if (nd != 4) or (len(image_shape) != len(weight_shape)):
        raise ValueError('Need input 4-D image shape and weight shape.')

    if stride == None:
        stride = (1, 1)

    if pad == None:
        pad = (0, 0)

    for i in range(4):
        c_image_shape[i] = image_shape[i]
        c_weight_shape[i] = weight_shape[i]
    
    for i in range(2):
        c_stride[i] = stride[i]
        c_pad[i] = pad[i]

    cdef size_t output_shape[4]
    _get_conv_output_shape(nd, c_image_shape, c_weight_shape, c_stride, c_pad, output_shape)

    return tuple(output_shape)
