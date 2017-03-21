import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
from mkldnn import mkldnn as mkl
from mkldnn import switch

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _mode = libcudnn.CUDNN_ACTIVATION_RELU


class ReLU(function.Function):

    """Rectified Linear Unit."""
    # TODO(beam2d): Implement in-place version.

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward_cpu(self, x):
        if switch.enable_relu:
            y = numpy.empty(x[0].shape, dtype=numpy.float32)
            if x[0].ndim == 4:
                mkl.Relu4D_F32.do_forward(x[0], y)
            else:
                #print (x[0].flags)
                #print (x[0].shape)
                in_x = x[0].ravel()
                out_y = y.ravel()
                mkl.Relu_F32.do_forward(in_x, out_y)
            return utils.force_array(y),
        else:
            return utils.force_array(numpy.maximum(x[0], 0, dtype=x[0].dtype)),

    def forward_gpu(self, x):
        if (cuda.cudnn_enabled and self.use_cudnn and
                x[0].flags.c_contiguous and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            y = cudnn.activation_forward(x[0], _mode)
            self.y = y
        else:
            y = cuda.cupy.maximum(x[0], 0)
        return y,

    def backward_cpu(self, x, gy):
        if switch.enable_relu:
            gx = numpy.empty(x[0].shape, dtype=numpy.float32)
            if x[0].ndim == 4:
                #self.mkldnn_relu_4d.backward(x[0], gy[0], gx)
                mkl.Relu4D_F32.do_backward(x[0], gy[0], gx)
            else:
                #print (x[0].flags)
                #print (x[0].shape)
                in_x = x[0].ravel()
                in_gy = gy[0].ravel()
                out_gx = gx.ravel()
                mkl.Relu_F32.do_backward(in_x, in_gy, out_gx)
            return utils.force_array(gx),
        else:
            return utils.force_array(gy[0] * (x[0] > 0)),

    def backward_gpu(self, x, gy):
        if (cuda.cudnn_enabled and self.use_cudnn and
                x[0].flags.c_contiguous and gy[0].flags.c_contiguous and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            gx = cudnn.activation_backward(x[0], self.y, gy[0], _mode)
        else:
            gx = cuda.elementwise(
                'T x, T gy', 'T gx',
                'gx = x > 0 ? gy : (T)0',
                'relu_bwd')(x[0], gy[0])
        return gx,


def relu(x, use_cudnn=True):
    """Rectified Linear Unit function.

     .. math::`f(x)=\\max(0, x)`.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_n)`-shaped float array.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_n)`-shaped float array.

    .. admonition:: Example

        >>> x = np.random.uniform(-1, 1, (3, 4, 5)).astype('f')
        >>> np.any(x < 0)
        True
        >>> y = F.relu(x)
        >>> np.any(y.data < 0)
        False
        >>> y.shape
        (3, 4, 5)

    """
    return ReLU(use_cudnn)(x)
