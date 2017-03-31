from mkldnn import mkldnn
import numpy as np

print(mkldnn.enabled())

x = np.ones((128,3,32,32), dtype=np.float32)
W = np.zeros((64,3,3,3), dtype=np.float32)
b = np.zeros((64,),dtype=np.float32)
y = np.empty(shape=(128,64,32,32),dtype=np.float32)

conv = mkldnn.Convolution2D_F32()

ret = conv.forward(x,W,b,y,1,1,1,1,1,1)

print(ret)


