import numpy as np
import unittest
import chainer.functions as F
import chainer.testing as testing
import chainer.testing.condition as condition
from mkldnn import switch


@testing.parameterize(*testing.product({
    'dtype': [np.float32],
    'chanel': [2, 4, 8, 16, 24]
}))
class TestLocalResponseNormalizationValidation(unittest.TestCase):
    def setUp(self):
        n = 5
        k = 1
        alpha = 1e-4
        beta = .75
        self.x = np.random.uniform(-1, 1, (2, self.chanel, 3, 2)).astype(self.dtype)
        self.gy = np.random.uniform(-1, 1, (2, self.chanel, 3, 2)).astype(self.dtype)
        self.check_forward_optionss = {}
        self.check_backward_optionss = {}
        if self.chanel >= 8:
            self.check_forward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
            self.check_backward_optionss = {'atol': 5e-3, 'rtol': 5e-3}
        self.lrn = F.LocalResponseNormalization(n, k, alpha, beta)
    
    def check_forward(self, x_data):
        switch.enable_lrn = True
        y = self.lrn.forward_cpu((x_data,))
        self.assertEqual(y[0].dtype, self.dtype)
        switch.enable_lrn = False
        y_expect = self.lrn.forward_cpu((x_data,))
        testing.assert_allclose(y_expect[0], y[0], **self.check_forward_optionss)
    
    def check_backward(self, x_data, y_grad):
        switch.enable_lrn = True
        gx = self.lrn.backward_cpu((x_data,), (y_grad,))
        switch.enable_lrn = False
        gx_expect = self.lrn.backward_cpu((x_data,), (y_grad,))
        testing.assert_allclose(gx_expect[0], gx[0], **self.check_backward_optionss)
    
    @condition.retry(3)
    def test_cpu(self):
        self.check_forward(self.x)
        self.check_backward(self.x, self.gy)
    
    @testing.attr.xeon
    @condition.retry(3)
    def test_xeon_cpu(self):
        print "test xeon"
        pass
    
    @testing.attr.xeon_phi
    @condition.retry(3)
    def test_xeon_phi_cpu(self):
        print "test xeon phi"
        pass


testing.run_module(__name__, __file__)
