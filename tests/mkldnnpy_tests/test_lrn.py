from mkldnn import mkldnn as mkl
import numpy as np
import chainer.functions as F

# mkl.enableMkldnn = false
print mkl.enabled()
# x = np.ones((128,3,32,32), dtype=np.float32)
y = np.empty(shape=(128,64,32,32),dtype=np.float32)
data = np.ndarray((1, 3, 2240, 2240), dtype=np.float32)
data.fill(333.33)
# print x
print y
print data.shape
f_lrn = F.local_response_normalization(data,1,3)
# mkl.setMkldnnEnable(true)
# print mkl.enabled()
# lrn = mkl.LocalResponseNormalization_F32(x,y,5,2,1e-4,.75)
# print lrn.forward()
# print y
# print f_lrn