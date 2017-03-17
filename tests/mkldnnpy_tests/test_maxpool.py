import numpy as np
import math
from mkldnn import mkldnn
from chainer.utils import conv

#input = [   [ 1, 2, 3 ],
#            [ 4, 5, 6 ],
#            [ 7, 8, 9 ] ]

n=1
c=1
h=4
w=4
stride=1
padding=1
ker=3
x = np.zeros((n, c, h, w), dtype=np.float32)

for i in range(n):
    for j in range(c):
        for k in range(h):
            for l in range(w):
                x[i, j, k, l] = math.sin(i+j+k+l)

print "x="
print x
forward_obj = mkldnn.MaxPooling_F32_get_forward_object(x, stride, stride,
                    padding, padding, ker, ker);

y_h = conv.get_conv_outsize(h, k, stride, padding)
y_w = conv.get_conv_outsize(w, k, stride, padding)
y = np.empty((n, c, y_h, y_w), dtype=x.dtype)

forward_obj.forward(x, y)
print "y="
print y
