import chainer.functions as F
import numpy as np
import time
from mkldnn import switch


def test_lrn(caculate, switchOn=True):
    total_forward = 0
    total_backward = 0
    data = np.ndarray((10, 3, 2240, 2240), dtype=np.float32)
    data.fill(333.33)
    datay = np.ndarray((10, 3, 2240, 2240), dtype=np.float32)
    datay.fill(333.33)
    # y = np.empty(shape=(10, 3, 2240, 2240), dtype=np.float32)
    # gx = np.empty(shape=(10, 3, 2240, 2240), dtype=np.float32)

    total_forward = 0
    count = 0
    niter = 5
    n_dry = 3

    n = 5
    k = 2
    alpha = 1e-4
    beta = .75

    switch.enable_lrn = switchOn
    for i in range(niter):
        x = np.asarray(data),
        gy = np.asarray(datay),

        # y = np.empty(shape=(10, 3, 2240, 2240), dtype=np.float32)
        # gx = np.empty(shape=(10, 3, 2240, 2240), dtype=np.float32)

        start = time.time()

        lrn = F.LocalResponseNormalization(n, k, alpha, beta)
        lrn.forward_cpu(x)
        end = time.time()
        if i > n_dry - 1:
            count += 1
            total_forward += (end-start)*1000

        start = time.time()
        lrn.backward_cpu(x, gy)
        end = time.time()
        if i > n_dry - 1:
            total_backward += (end-start)*1000

    # print("Mkldnn Average Forward: ", total_forward/count, "ms")
    print(caculate, " Average Forward: ", total_forward/count, "ms")
    print(caculate, " Average Backward: ", total_backward/count, "ms")
    print(caculate, " Average Total: ", (total_forward + total_backward)/count, "ms")

test_lrn("mkldnn")
test_lrn("numpy", False)
