import numpy as np
import unittest
import chainer.functions as F
import chainer.testing as testing
import chainer.testing.condition as condition
from mkldnn import switch


class TestMaxpool2(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(1, 1, 4, 4).astype('f'),
        self.gy = np.random.rand(1, 1, 4, 4).astype('f'),

    def tearDown(self):
        self.x = None
        self.gy = None

    def check_maxpool2(self):
        for _ in range(2):
            switch.enable_max_pooling = True
            f = F.MaxPooling2D(3, stride=1, pad=1, use_cudnn=False)
            y = f.forward_cpu(self.x)
            gx = f.backward_cpu(self.x, self.gy)
            switch.enable_max_pooling = False
            f = F.MaxPooling2D(3, stride=1, pad=1, use_cudnn=False)
            y_expect = f.forward_cpu(self.x)
            gx_expect = f.backward_cpu(self.x, self.gy)
            testing.assert_allclose(np.asarray(y), np.asarray(y_expect))
            testing.assert_allclose(np.asarray(gx), np.asarray(gx_expect))

    @condition.retry(3)
    def test_cpu(self):
        self.check_maxpool2()


testing.run_module(__name__, __file__)
