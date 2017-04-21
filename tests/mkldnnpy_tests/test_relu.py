from mkldnn import mkldnn
import chainer.functions as F
import numpy as np

x = np.ones((128,3,32,32), dtype=np.float32)
W = np.zeros((64,3,3,3), dtype=np.float32)
b = np.zeros((64,),dtype=np.float32)
y = np.empty(shape=(128,64,32,32),dtype=np.float32)

#conv = mkldnnpy.Convolution2D_F32(x, W, y, 1, 1, 1, 1)

#ret = conv.forward()
#x = np.linspace((-5, 5, 48), dtype=np.float32).reshape(2, 2, 3, 4)
x1 = np.arange(-24,24).reshape(2, 2, 3, 4).astype(np.float32)
x2 = x1
x = tuple([x1, x2])
print (x[0])

#relu_f = F.relu(x)
f_relu = F.ReLU(False)
res = f_relu.forward_cpu(x)
print(res)

print ('--------------gy---------------')
gy = x;
res = f_relu.backward_cpu(x, gy)
print(res)

