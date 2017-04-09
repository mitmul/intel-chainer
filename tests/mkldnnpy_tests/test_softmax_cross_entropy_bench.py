import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import time
from mkldnn import mkldnn
from mkldnn import switch

switch.enable_softmax_cross_entropy = True

# Accuracy Test
mkldnn.set_mkldnn_enable(True)

print "With mkldnn"
x = np.arange(1.0, 13.0, 2.0, dtype=np.float32).reshape(2,3)
label = np.array([0, 2], dtype=np.int32)
print "x	****************** "
print x
print "label	****************** "
print label

sce = F.SoftmaxCrossEntropy(use_cudnn=False, normalize=True, cache_score=True)
loss = sce.forward_cpu((x, label))
gx = sce.backward_cpu((x, label), (1, 1))

print "loss	******************* "
print loss
print "gx	******************* "
print gx

mkldnn.set_mkldnn_enable(False)

print " "
print "Without mkldnn"
x = np.arange(1.0, 13.0, 2.0, dtype=np.float32).reshape(2,3)
label = np.array([0, 2], dtype=np.int32)
print "x	****************** "
print x
print "label	****************** "
print label

sce = F.SoftmaxCrossEntropy(use_cudnn=False, normalize=True, cache_score=True)
loss = sce.forward_cpu((x, label))
gx = sce.backward_cpu((x, label), (1, 1))

print "loss	******************* "
print loss
print "gx	******************* "
print gx

# Performance Test
mkldnn.set_mkldnn_enable(True)
total = 0
count = 0
n_dry = 3
niter = 100
for i in range(niter):
    x = np.ndarray((10, 1000), dtype=np.float32)
    x.fill(3.33)
    label = np.array([8, 2, 3, 5, 2, 6, 4, 3, 9, 2], dtype=np.int32)

    start = time.time()
    sce = F.SoftmaxCrossEntropy(use_cudnn=False, normalize=True, cache_score=True)
    loss = sce.forward_cpu((x, label))
    gx = sce.backward_cpu((x, label), (1, 1))
    end = time.time()

    if i > n_dry - 1:
        count += 1
        total += (end-start)*1000


print("Average with mkldnn : ", total/count, "ms")
print("Total with mkldnn : ", total, "ms")

mkldnn.set_mkldnn_enable(False)
total = 0
count = 0
n_dry = 3
niter = 100
for i in range(niter):
    x = np.ndarray((10, 1000), dtype=np.float32)
    x.fill(3.33)
    label = np.array([8, 2, 3, 5, 2, 6, 4, 3, 9, 2], dtype=np.int32)

    start = time.time()
    sce = F.SoftmaxCrossEntropy(use_cudnn=False, normalize=True, cache_score=True)
    loss = sce.forward_cpu((x, label))
    gx = sce.backward_cpu((x, label), (1, 1))
    end = time.time()

    if i > n_dry - 1:
        count += 1
        total += (end-start)*1000


print("Average without mkldnn : ", total/count, "ms")
print("Total without mkldnn : ", total, "ms")
