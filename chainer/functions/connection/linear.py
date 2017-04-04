import numpy

from chainer import function
from chainer.utils import type_check
from mkldnn import mkldnn
from mkldnn import switch


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class LinearFunction(function.Function):
    def __init__(self, linear_link = None):
        if switch.enable_linear and linear_link is None:
            assert "linear_link can not be None in mkldnn enabled mode"
        self.linear_link = linear_link

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, inputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        b = inputs[2] if len(inputs) == 3 else None
        if switch.enable_linearF(inputs):
            y = numpy.empty(shape=(x.shape[0], W.shape[0]), dtype=W.dtype);
            if b is not None:
                mkldnn.Linear_F32.do_forward(x, W, b, y)
            else:
                mkldnn.Linear_F32.do_forward(x, W, y);
            return y,
        else:
            y = x.dot(W.T).astype(x.dtype, copy=False)
            if b is not None:
                y += b
            return y,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        """
        For MKLDNN backward, only support float32
        """
        if switch.enable_linearF(inputs):
            gW = numpy.empty(shape=W.shape, dtype=W.dtype)
            gx = numpy.empty(shape=x.shape, dtype=W.dtype)
            if b is not None:
                gb = numpy.empty(shape=b.shape, dtype=W.dtype)
                mkldnn.Linear_F32.do_backward(x, W, b, gy, gW, gx, gb)
                return gx.reshape(inputs[0].shape), gW, gb
            else:
                mkldnn.Linear_F32.do_backward(x, W, gy, gW, gx)
                return gx.reshape(inputs[0].shape), gW
        else:
            gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
            gW = gy.T.dot(x).astype(W.dtype, copy=False)
            if b is not None:
                gb = gy.sum(0)
                return gx, gW, gb
            else:
                return gx, gW

def linear(x, W, b=None, linear_link=None):
    """Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
     .. math:: Y = xW^\\top + b.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which is a :math:`(s_B, s_1, \
            s_2, ..., s_n)`-shaped float array. Its first dimension
            :math:`(s_B)` is assumed to be the *minibatch dimension*. The
            other dimensions are treated as concatenated one dimension whose
            size must be :math:`(s_1 * ... * s_n = N)`.
        W (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Weight variable of shape :math:`(M, N)`,
            where :math:`(N = s_1 * ... * s_n)`.
        b (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Bias variable (optional) of shape
            :math:`(M,)`.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(s_B, M)`.

    .. seealso:: :class:`~chainer.links.Linear`

    .. admonition:: Example

        >>> x = np.random.uniform(0, 1, (3, 4)).astype('f')
        >>> W = np.random.uniform(0, 1, (5, 4)).astype('f')
        >>> b = np.random.uniform(0, 1, (5,)).astype('f')
        >>> y = F.linear(x, W, b)
        >>> y.shape
        (3, 5)

    """
    if b is None:
        return LinearFunction(linear_link)(x, W)
    else:
        return LinearFunction(linear_link)(x, W, b)
