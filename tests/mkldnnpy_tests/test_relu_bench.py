import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import time

data = np.ndarray((1, 3, 2240, 2240), dtype=np.float32)
data.fill(333.33)
gy = np.ndarray((1, 3, 2240, 2240), dtype=np.float32)
gy.fill(333.33)

total_forward = 0
count = 0
niter = 13
n_dry = 3



for i in range(niter):
    x = np.asarray(data),

    start = time.time()
    #model.forward(x)
    f_relu = F.ReLU(False)
    f_relu.forward_cpu(x)
    f_relu.backward_cpu(x, gy)
    end = time.time()
    if i > n_dry - 1:
        count += 1
        total_forward += (end-start)*1000


print("Average Forward: ", total_forward/count, "ms")

