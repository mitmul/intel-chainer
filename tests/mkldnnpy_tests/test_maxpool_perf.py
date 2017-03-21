import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import math
import time

#input = [   [ 1, 2, 3 ],
#            [ 4, 5, 6 ],
#            [ 7, 8, 9 ] ]

repeat = 1;
(n, c, h, w) = (64, 256, 256, 256)
x = np.ones((n, c, h, w), dtype=np.float32)
f = F.MaxPooling2D(3, stride=2, pad = 0, use_cudnn=False)

x = x,
y = f.forward_cpu(x)
start = time.time()
for i in range (repeat):
    y = f.forward_cpu(x)
print "average forward %f seconds"%((time.time()-start)/repeat)

gy = np.ndarray(y[0].shape, dtype=np.float32)
gy.fill(0.001)
gy = gy,
gx = f.backward_cpu(x, gy)
start = time.time()
for i in range(repeat):
    gx = f.backward_cpu(x, gy)
print "average backward %f seconds"%((time.time()-start)/repeat)
