import numpy as np
import math
from mkldnn import mkldnn
from chainer.utils import conv

n=1
c=1
h=4
w=4
stride=1
padding=1
ker=3
x = np.zeros((n, c, h, w), dtype=np.float32)
x2 = np.zeros((n, c, h, w), dtype=np.float32)
gy = np.zeros((n, c, h, w), dtype=np.float32)

for i in range(n):
    for j in range(c):
        for k in range(h):
            for l in range(w):
                x[i, j, k, l] = math.sin(i+j+k+l)
                x2[i, j, k, l] = math.sin(i+j+k+l)+1

for i in range(n):
    for j in range(c):
        for k in range(h):
            for l in range(w):
                gy[i, j, k, l] = 0.001;

print "x="
print x

y_h = conv.get_conv_outsize(h, k, stride, padding)
y_w = conv.get_conv_outsize(w, k, stride, padding)
y   = np.empty((n, c, y_h, y_w), dtype=x.dtype)
y2   = np.empty((n, c, y_h, y_w), dtype=x.dtype)
gx  = np.empty((n, c, h, w), dtype=x.dtype)
ws  = np.empty((n, c, y_h, y_w), dtype=np.int32)
ws2  = np.empty((n, c, y_h, y_w), dtype=np.int32)

mkldnn.MaxPooling_F32_do_forward(x, y, ws, stride, stride,
                    padding, padding, ker, ker);
print "y="
print y

mkldnn.MaxPooling_F32_do_forward(x2, y2, ws2, stride, stride,
                    padding, padding, ker, ker);
print "y2="
print y2
mkldnn.MaxPooling_F32_do_backward(gy, x, gx, ws, stride, stride,
                    padding, padding, ker, ker);
print "gx="
print gx
