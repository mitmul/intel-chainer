from mkldnn import mkldnnpy
import numpy as np

x = np.ones((128,3,32,32), dtype=np.float32)
W = np.zeros((64,3,3,3), dtype=np.float32)
b = np.zeros((64,),dtype=np.float32)
y = np.empty(shape=(128,64,32,32),dtype=np.float32)

conv = mkldnnpy.Convolution2D_F32(x, W, y, 1, 1, 1, 1)

ret = conv.forward()

print(ret)


