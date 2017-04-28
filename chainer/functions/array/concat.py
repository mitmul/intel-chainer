import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check
from mkldnn import mkldnn
from mkldnn import switch


class Concat(function.Function):

    """Concatenate multiple tensors towards specified axis."""

    # concat along the channel dimension by default
    def __init__(self, axis=1):
        if not isinstance(axis, int):
            raise TypeError('axis must be int')

        self.axis = axis
        self.mkldnn_concat = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check.expect(in_types[0].ndim >
                          type_check.Variable(self.axis, 'axis'))

        type_check.expect(
            -in_types[0].ndim <= self.axis,
            self.axis < in_types[0].ndim
        )
        ndim = in_types[0].ndim.eval()
        axis = self.axis % ndim
        for i in six.moves.range(1, in_types.size().eval()):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            for d in six.moves.range(0, ndim):
                if d == axis:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def forward(self, xs):
        if switch.enable_concatF((xs,)) and self.axis == 1 and xs[0].ndim == 4 and all(isinstance(xi, numpy.ndarray) for xi in xs):
            out_c = 0
            xs_new = ()

            need_copy = False
            for xi in xs:
                if xi.flags.contiguous is False:
                    need_copy = True
            """
            tuple's element value is not c continuous,
            need to copy it so that native can get correct value from buffer address
            """
            for xi in xs:
                out_c += xi.shape[1]
                if need_copy:
                    tmp = xi.copy().astype(numpy.float32)
                    xs_new += (tmp,)

            self.mkldnn_concat = mkldnn.Concat_F32()

            """
            only support channel dim concat
            """
            y = numpy.empty(shape=(xs[0].shape[0], out_c, xs[0].shape[2], xs[0].shape[3]), dtype=xs[0].dtype)
            if need_copy:
                self.mkldnn_concat.forward(xs_new, y, self.axis)
            else:
                self.mkldnn_concat.forward(xs, y, self.axis)
            return y,
        else:
            xp = cuda.get_array_module(*xs)
            y = xp.concatenate(xs, axis=self.axis)
            return y,

    def backward(self, xs, gy):
        if len(xs) == 1:
            return gy
        if switch.enable_concatF((xs, gy)) and self.axis == 1 and xs[0].ndim == 4 and all(isinstance(xi, numpy.ndarray) for xi in xs):
            # x should have same shape as xs
            xs_new = ()
            for xi in xs:
                temp = numpy.empty(shape=xi.shape, dtype=xi.dtype)
                xs_new += (temp,)
            self.mkldnn_concat.backward(xs_new, gy[0], self.axis)
            return xs_new
        else:
            xp = cuda.get_array_module(*xs)
            sizes = numpy.array([x.shape[self.axis] for x in xs[:-1]]).cumsum()
            x = xp.split(gy[0], sizes, axis=self.axis)
            return x


def concat(xs, axis=1):
    """Concatenates given variables along an axis.

    Args:
        xs (tuple of Variables): Variables to be concatenated.
        axis (int): Axis that the input arrays are concatenated along.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Concat(axis=axis)(*xs)
