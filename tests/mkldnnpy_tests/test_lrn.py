from mkldnn import mkldnn as mkl
import numpy as np
import chainer.functions as F
from mkldnn import switch 
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
n = 5
k = 1
alpha = 1e-4
beta = .75
# data = np.ndarray((10, 3, 2240, 2240), dtype=np.float32)
# data.fill(333.33)
# datay = np.ndarray((10, 3, 2240, 2240), dtype=np.float32)
# datay.fill(333.33)

data = np.ndarray((2, 4, 5, 5), dtype=np.float32)
data.fill(33)
datay = np.ndarray((2, 4, 5, 5), dtype=np.float32)
datay.fill(33)

x = np.asarray(data),
gy = np.asarray(datay),

# y = np.empty((5, 2, 1, 1),dtype=np.float32)
# gx = np.empty((5, 2, 1, 1),dtype=np.float32)
# print x
lrn = F.LocalResponseNormalization(n,k,alpha,.75)

switch.enable_lrn = True
my = lrn.forward_cpu(x)
# lrn.forward_cpu(x)
mklgx = lrn.backward_cpu(x,gy)
# print "mkl y = " + str(lrn.forward_cpu(x))
# print "mkl gx = " + str(lrn.backward_cpu(x,gy))
# lrn = F.local_response_normalization(data,5,2)

switch.enable_lrn = False
ny = lrn.forward_cpu(x)
numpygx=lrn.backward_cpu(x,gy)

print "mklgx-numpygx=" + str(numpygx[0]-mklgx[0])
# print "numpy y = " + str(lrn.forward_cpu(x))
# lrn.forward_cpu(x)
# print "numpy gx = " + str(lrn.backward_cpu(x,gy))
print "mkl - numpy = " + str(my[0] - ny[0])