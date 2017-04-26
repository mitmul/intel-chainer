import numpy as np
import unittest
import chainer.testing as testing
import chainer.testing.condition as condition
from chainer import functions as F
from mkldnn import switch


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.dims = [(2, 3), (1, 6)]
    
    def tearDown(self):
        self.dims = None
    
    def check_softmax(self):
        for dim1, dim2 in self.dims:
            x_2d = np.random.rand(dim1, dim2).astype('f')
            switch.enable_softmax = True
            y_2d = F.softmax(x_2d, use_cudnn=False)
            switch.enable_softmax = False
            y_2d_expect = F.softmax(x_2d, use_cudnn=False)
            testing.assert_allclose(y_2d.data, y_2d_expect.data)
        
    
    @condition.retry(3)
    def test_cpu(self):
        self.check_softmax()


testing.run_module(__name__, __file__)
