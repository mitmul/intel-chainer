import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable

import numpy as np
import time

batch = 1000
total_backward=0
total_forward=0
count = 0
niter=10
n_dry=3

data = np.ndarray((batch, 1000), dtype=np.float32)
data.fill(333.33)

y_grad = np.ones((batch, 1000), dtype=np.float32)

linear = L.Linear(1000, 1000)

x = Variable(data)

for i in range(niter):
    print "iter:" ,i
    x = np.asarray(data)
    start = time.time()
    y = linear(x)  
    end = time.time()
    if i > n_dry - 1:
        count+=1
        total_forward += (end-start) * 1000
    y.grad = y_grad
    start = time.time()
    y.backward()
    end = time.time()
    if i > n_dry -1:
        total_backward += (end-start) * 1000

print("Average Forward: ", total_forward/count, "ms")
print("Average Backward: ", total_backward/count, "ms")
print("Average Total: ", (total_forward + total_backward)/count, "ms")

