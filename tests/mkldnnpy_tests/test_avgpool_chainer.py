import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import math

#input = [   [ 1, 2, 3 ],
#            [ 4, 5, 6 ],
#            [ 7, 8, 9 ] ]

x = np.zeros((1, 1, 4, 4), dtype=np.float32)

for i in range(1):
    for j in range(1):
        for k in range(4):
            for l in range(4):
                x[i, j, k, l] = math.sin(i+j*2+k*3+l*4)
                                        # break symmetry

print "x="
print x
y1 = F.average_pooling_2d(x, 3, stride=1, pad=1)
y2 = F.average_pooling_2d(x, 3, stride=1, pad=1)

print "y1="
print y1.data
print "y2="
print y2.data
