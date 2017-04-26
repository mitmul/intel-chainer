import chainer.functions as F
import numpy as np
import time

data = np.ndarray((10, 3, 2240, 2240), dtype=np.float32)
data.fill(333.33)
datay = np.ndarray((10, 3, 2240, 2240), dtype=np.float32)
datay.fill(333.33)

total_forward = 0
count = 0
niter = 15
n_dry = 3

for i in range(niter):
    x = np.asarray(data),
    gy = np.asarray(datay),

    start = time.time()
    # model.forward(x)
    f_relu = F.ReLU(False)
    f_relu.forward_cpu(x)
    f_relu.backward_cpu(x, gy)
    end = time.time()
    if i > n_dry - 1:
        count += 1
        total_forward += (end-start)*1000


print("Average Forward: ", total_forward/count, "ms")

data = np.ndarray((2240, 2240), dtype=np.float32)
data.fill(333.33)
datay = np.ndarray((2240, 2240), dtype=np.float32)
datay.fill(333.33)

total_forward = 0
count = 0
niter = 5
n_dry = 3

for i in range(niter):
    x = np.asarray(data),
    gy = np.asarray(datay),

    start = time.time()
    # model.forward(x)
    f_relu = F.ReLU(False)
    f_relu.forward_cpu(x)
    f_relu.backward_cpu(x, gy)
    end = time.time()
    if i > n_dry - 1:
        count += 1
        total_forward += (end-start)*1000


print("Average Forward: ", total_forward/count, "ms")
