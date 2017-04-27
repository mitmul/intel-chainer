import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import time

from chainer import Variable

batch = 1
total_backward = 0
total_forward = 0
count = 0

niter = 5
n_dry = 3

#data = np.ndarray((batch, 3, 224, 224), dtype=np.float32)
#data.fill(333.33)
data = np.random.rand(batch, 8, 3, 3).astype(np.float32)
y_grad = np.ones((batch, 32, 3, 3), dtype=np.float32)

#incept = L.Inception(3,  64,  96, 128, 16,  32,  32)
incept = L.Inception(8,  8,  96, 8, 16,  8,  8)
x = Variable(data)

for i in range(niter):
    print "iter:", i
    start = time.time()
    y = incept(x)
    end = time.time()
    if i > n_dry - 1:
        count += 1
        total_forward += (end-start)*1000

    y.grad = y_grad
    start = time.time()
    y.backward()
    end = time.time()
    if i > n_dry - 1:
            total_backward += (end-start)*1000

print("Average Forward: ", total_forward/count, "ms")
print("Average Backward: ", total_backward/count, "ms")
print("Average Total: ", (total_forward + total_backward)/count, "ms")

