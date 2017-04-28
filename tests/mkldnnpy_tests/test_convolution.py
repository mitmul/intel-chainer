import numpy as np
import unittest
import chainer.links as L
import chainer.testing as testing
import chainer.testing.condition as condition
from mkldnn import switch


class TestConvolution(unittest.TestCase):
    def setUp(self):
        self.n = 1
        self.c = 16
        self.h = 32
        self.w = 32
        self.in_size = self.c
        self.out_size = 64
        self.ker_size = 3
        self.x = np.random.rand(self.n, self.c, self.h, self.w).astype('f')
        self.W = np.random.rand(self.out_size, self.in_size, self.ker_size, self.ker_size).astype('f')
        self.chainer_conv = L.Convolution2D(self.in_size, self.out_size, self.ker_size, stride=1, pad=1,
                                            initialW=self.W,
                                            use_cudnn=False)

    def tearDown(self):
        self.x = None
        self.y = None
        self.W = None

    def check_convolution(self):
        switch.enable_conv = True
        result = self.chainer_conv(self.x)
        switch.enable_conv = False
        result_expect = self.chainer_conv(self.x)
        testing.assert_allclose(result.data, result_expect.data)

    @condition.retry(3)
    def test_cpu(self):
        self.check_convolution()


testing.run_module(__name__, __file__)
