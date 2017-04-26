import numpy as np
import unittest
import chainer.functions as F
import chainer.testing as testing
import chainer.testing.condition as condition
from mkldnn import switch


class TestAvgPool(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(1, 1, 4, 4).astype('f')

    def tearDown(self):
        self.x = None

    def check_avgpool(self):
        switch.enable_avg_pooling = True
        y = F.average_pooling_2d(self.x, 3, stride=1, pad=1)
        switch.enable_avg_pooling = False
        y_expect = F.average_pooling_2d(self.x, 3, stride=1, pad=1)
        testing.assert_allclose(y.data, y_expect.data)

    @condition.retry(3)
    def test_cpu(self):
        self.check_avgpool()


testing.run_module(__name__, __file__)
