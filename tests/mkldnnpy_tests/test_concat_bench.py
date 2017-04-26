import chainer.functions as F
import numpy as np
import time

batchsize = 1

data1 = np.random.rand(batchsize, 2, 224, 224).astype(np.float32)
data2 = np.random.rand(batchsize, 4, 224, 224).astype(np.float32)
data3 = np.random.rand(batchsize, 8, 224, 224).astype(np.float32)
data4 = np.random.rand(batchsize, 16, 224, 224).astype(np.float32)
gy = np.random.rand(batchsize, 30, 224, 224).astype(np.float32)

total_forward = 0
total_backward = 0
count = 0
niter = 4
n_dry = 3

for i in range(niter):
    f_concat = F.Concat()

    start = time.time()
    f_concat.forward((data1, data2, data3, data4))
    end = time.time()
    if i > n_dry - 1:
        count += 1
        total_forward += (end-start)*1000

    start = time.time()
    f_concat.backward((data1, data2, data3, data4), (gy,))
    end = time.time()
    if i > n_dry - 1:
        total_backward += (end-start)*1000


print("Average Forward: ", total_forward/count, "ms")
print("Average Backward: ", total_backward/count, "ms")
print("Average Total: ", (total_forward + total_backward)/count, "ms")
