from mkldnn import mkldnn as mkl
import numpy as np

print mkl.enabled()
x = np.ones((128,3,32,32), dtype=np.float32)
y = np.empty(shape=(128,64,32,32),dtype=np.float32)
# print x
# print y
lrn = mkl.LocalResponseNormalization_F32(x,y,5,2,1e-4,.75)
print lrn.forward()
print y