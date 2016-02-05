import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'in_shape': (3, 2), 'out_shape': (1, 3, 2), 'axis': 0},
    {'in_shape': (3, 2), 'out_shape': (3, 1, 2), 'axis': 1},
    {'in_shape': (3, 2), 'out_shape': (3, 2, 1), 'axis': 2},

    {'in_shape': (3, 2), 'out_shape': (3, 2, 1), 'axis': -1},
    {'in_shape': (3, 2), 'out_shape': (3, 1, 2), 'axis': -2},
    {'in_shape': (3, 2), 'out_shape': (1, 3, 2), 'axis': -3},
)
class TestExpandDims(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.in_shape) \
                             .astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, self.out_shape) \
                              .astype(numpy.float32)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.expand_dims(x, self.axis)
        self.assertEqual(y.data.shape, self.out_shape)
        y_expect = numpy.expand_dims(cuda.to_cpu(x_data), self.axis)
        numpy.testing.assert_array_equal(cuda.to_cpu(y.data), y_expect)

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            functions.ExpandDims(self.axis), x_data, y_grad)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward(self):
        self.check_forward(cuda.to_gpu(self.x))

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def test_invalid_dim(self):
        x = chainer.Variable(self.x)
        with self.assertRaises(chainer.utils.type_check.InvalidType):
            functions.expand_dims(x, self.x.ndim + 1)
        with self.assertRaises(chainer.utils.type_check.InvalidType):
            functions.expand_dims(x, -self.x.ndim - 2)


testing.run_module(__name__, __file__)
