from mkldnn import mkldnn as mkl
import numpy as np
from mkldnn import switch

import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

@testing.parameterize(*testing.product({
    'dtype': [numpy.float32],
    'chanel':[2,4,8,16,24]
}))

class TestLocalResponseNormalizationValidation(unittest.TestCase):
    def setUp(self):
                n = 5
                k = 1
                alpha = 1e-4
                beta = .75
                self.x = numpy.random.uniform(-1, 1, (2, self.chanel, 3, 2)).astype(self.dtype)
                self.gy = numpy.random.uniform(-1, 1, (2, self.chanel, 3, 2)).astype(self.dtype)
                self.check_forward_optionss = {}
                self.check_backward_optionss = {}
                if self.chanel >= 16:
                        self.check_forward_optionss = {'atol': 1e-4, 'rtol': 1e-3}
                        self.check_backward_optionss = {'atol': 5e-3, 'rtol': 5e-3}
                self.lrn = functions.LocalResponseNormalization(n,k,alpha,beta)
    def check_forward(self, x_data):
                switch.enable_lrn = True
                y = self.lrn.forward_cpu((x_data,))
                self.assertEqual(y[0].dtype, self.dtype)
                switch.enable_lrn = False
                y_expect = self.lrn.forward_cpu((x_data,))
                testing.assert_allclose(y_expect[0], y[0], **self.check_forward_optionss)
    def check_backward(self, x_data, y_grad):
                switch.enable_lrn = True
                gx = self.lrn.backward_cpu((x_data,),(y_grad,))
                switch.enable_lrn = False
                gx_expect = self.lrn.backward_cpu((x_data,),(y_grad,))
                testing.assert_allclose(gx_expect[0], gx[0], **self.check_backward_optionss)
    @condition.retry(3)
    def test_cpu(self):
        self.check_forward(self.x)
        self.check_backward(self.x, self.gy)
    @attr.xeon
    @condition.retry(3)
    def test_xeon_cpu(self):
            print ("test xeon")
            pass
    @attr.xeon_phi
    @condition.retry(3)
    def test_xeon_phi_cpu(self):
            print ("test xeon phi")
            pass

testing.run_module(__name__, __file__)
# mkl.enableMkldnn = false
# print mkl.enabled()
# x = np.ones((128,3,32,32), dtype=np.float32)
# y = np.empty(shape=(1,3,2240,2240),dtype=np.float32)
# data = np.ndarray((1, 3, 2240, 2240), dtype=np.float32)
# data.fill(333.33)
# print data
# print y
# print data.shape
# f_lrn = F.local_response_normalization(data,1,3)
# mkl.setMkldnnEnable(False)
# print mkl.enabled()
# lrn = mkl.LocalResponseNormalization_F32(5,2,1e-4,.75)
# lrn.forward(data,y)
# print "mkl y = " + str(y)
# f_lrn = F.LocalResponseNormalization(1,3)
# switch.enable_lrn = False
# print f_lrn.forward_cpu(data)
# n = 5
# k = 1
# alpha = 1e-4
# beta = .75
# data = np.ndarray((10, 3, 2240, 2240), dtype=np.float32)
# data.fill(333.33)
# datay = np.ndarray((10, 3, 2240, 2240), dtype=np.float32)
# datay.fill(333.33)

# data = np.ndarray((2, 8, 5, 5), dtype=np.float32)
# data.fill(213)
# datay = np.ndarray((2, 8, 5, 5), dtype=np.float32)
# datay.fill(123)

# x = np.asarray(data),
# gy = np.asarray(datay),

# y = np.empty((5, 2, 1, 1),dtype=np.float32)
# gx = np.empty((5, 2, 1, 1),dtype=np.float32)
# print x
# lrn = F.LocalResponseNormalization(n,k,alpha,.75)
# mkl.enableGoogleLogging()
# switch.enable_lrn = True
# my = lrn.forward_cpu(x)
# lrn.forward_cpu(x)
# mklgx = lrn.backward_cpu(x,gy)
# print "mkl y = " + str(lrn.forward_cpu(x))
# print "mkl gx = " + str(lrn.backward_cpu(x,gy))
# lrn = F.local_response_normalization(data,5,2)

# switch.enable_lrn = False
# ny = lrn.forward_cpu(x)
# numpygx = lrn.backward_cpu(x,gy)


# print "my - ny" + str(my[0] - ny[0])
# print mklgx

# re = numpygx[0]- mklgx[0]
# print re
# print "mklgx-numpygx=" + str(numpygx[0]- mklgx[0])
# print "numpy y = " + str(lrn.forward_cpu(x))
# lrn.forward_cpu(x)
# print "numpy gx = " + str(lrn.backward_cpu(x,gy))
# print "mkl - numpy = " + str(my[0] - ny[0])
