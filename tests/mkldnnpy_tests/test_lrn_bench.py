import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import time
from mkldnn import mkldnn as mkl

data = np.ndarray((10, 3, 2240, 2240), dtype=np.float32)
data.fill(333.33)
datay = np.ndarray((10, 3, 2240, 2240), dtype=np.float32)
datay.fill(333.33)
y = np.empty(shape=(10,3,2240,2240),dtype=np.float32)
gx = np.empty(shape=(10,3,2240,2240),dtype=np.float32)
total_forward = 0
count = 0
niter = 5
n_dry = 3

n = 5
k = 2
alpha = 1e-4
beta = .75

for i in range(niter):
    x = np.asarray(data),
    gy = np.asarray(datay),

    y = np.empty(shape=(10,3,2240,2240),dtype=np.float32)
    gx = np.empty(shape=(10,3,2240,2240),dtype=np.float32)

    start = time.time()
    #model.forward(x)
    lrn = mkl.LocalResponseNormalization_F32(n,k,alpha,.75)
    lrn.forward(x[0],y)
    lrn.backward(x[0],gy[0],gx)
    # lrn = F.LocalResponseNormalization(n,k,alpha,.75)
    # lrn.forward_cpu(x)
    # lrn.backward_cpu(x,gy)
    end = time.time()
    if i > n_dry - 1:
        count += 1
        total_forward += (end-start)*1000


print("Mkldnn Average Forward: ", total_forward/count, "ms")


for i in range(niter):
    x = np.asarray(data),
    gy = np.asarray(datay),

    y = np.empty(shape=(10,3,2240,2240),dtype=np.float32)
    gx = np.empty(shape=(10,3,2240,2240),dtype=np.float32)

    start = time.time()
    #model.forward(x)
    # lrn = mkl.LocalResponseNormalization_F32(n,k,alpha,.75)
    # lrn.forward(x[0],y)
    # lrn.backward(x[0],gy[0],gx)
    lrn = F.LocalResponseNormalization(n,k,alpha,.75)
    # lrn = F.local_response_normalization(data,5,2)
    lrn.forward_cpu(x)
    lrn.backward_cpu(x,gy)
    end = time.time()
    if i > n_dry - 1:
        count += 1
        total_forward += (end-start)*1000


print("Numpy Average Forward: ", total_forward/count, "ms")

# data = np.ndarray((2240, 2240), dtype=np.float32)
# data.fill(333.33)
# datay = np.ndarray((2240, 2240), dtype=np.float32)
# datay.fill(333.33)

# total_forward = 0
# count = 0
# niter = 5
# n_dry = 3



# for i in range(niter):
#     x = np.asarray(data),
#     gy = np.asarray(datay),


#     start = time.time()
#     #model.forward(x)
#     f_relu = F.ReLU(False)
#     f_relu.forward_cpu(x)
#     f_relu.backward_cpu(x, gy)
#     end = time.time()
#     if i > n_dry - 1:
#         count += 1
#         total_forward += (end-start)*1000


# print("Average Forward: ", total_forward/count, "ms")

