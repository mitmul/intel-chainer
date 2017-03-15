import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import math

#input = [   [ 1, 2, 3 ],
#            [ 4, 5, 6 ],
#            [ 7, 8, 9 ] ]

x = np.zeros((1, 3, 6, 6), dtype=np.float32)

for i in range(1):
    for j in range(3):
        for k in range(6):
            for l in range(6):
                x[i, j, k, l] = math.sin(i+j+k+l)

y = F.max_pooling_2d(x, 3, stride=1, pad=1)

print "x="
print x
print "y="
print y.data
