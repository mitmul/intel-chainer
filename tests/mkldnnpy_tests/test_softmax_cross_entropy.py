import numpy as np
import unittest
import chainer.testing as testing
import chainer.testing.condition as condition
from chainer import functions as F
from mkldnn import switch


class TestSoftmaxCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.x_2d = np.random.rand(2, 3).astype('f')
        self.label = np.random.rand(2).astype('i')

    def tearDown(self):
        self.x_2d = None
        self.label = None

    def check_softmax_cross_entropy(self):
        switch.enable_softmax_cross_entropy = True
        y_2d = F.softmax_cross_entropy(self.x_2d, self.label, use_cudnn=False)
        switch.enable_softmax_cross_entropy = False
        y_2d_expect = F.softmax_cross_entropy(self.x_2d, self.label, use_cudnn=False)
        testing.assert_allclose(y_2d.data, y_2d_expect.data)

    @condition.retry(3)
    def test_cpu(self):
        self.check_softmax_cross_entropy()


testing.run_module(__name__, __file__)
