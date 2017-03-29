import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import math

#input = [   [ 1, 2, 3 ],
#            [ 4, 5, 6 ],
#            [ 7, 8, 9 ] ]

(n, c, h, w) = (1, 16, 32, 32)
(in_size, out_size, ker_size) = (c, 64, 3)

x = np.zeros((n, c, h, w), dtype=np.float32)

for i in range(n):
    for j in range(c):
        for k in range(h):
            for l in range(w):
                x[i, j, k, l] = math.sin(i+j*2+k*3+l*4)
                                        # break symmetry

w = np.zeros((out_size, in_size, ker_size, ker_size), dtype=np.float32)
for i in range(out_size):
    for j in range(in_size):
        for k in range(ker_size):
            for l in range(ker_size):
                w[i, j, k, l] = math.cos(i+j*2+k*3+l*4)


conv = L.Convolution2D(in_size, out_size, ker_size, stride=1, pad=1, initialW=w, use_cudnn=False);
print "x="
print x
y1 = conv(x);

print "y1="
print y1.data
