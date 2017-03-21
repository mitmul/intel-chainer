import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import math
import time

#input = [   [ 1, 2, 3 ],
#            [ 4, 5, 6 ],
#            [ 7, 8, 9 ] ]

(n, c, h, w) = (128, 256, 64, 64)
x = np.ones((n, c, h, w), dtype=np.float32)

start = time.time();
y1 = F.max_pooling_2d(x, 3, stride=2, pad=0)
print "first pass %f seconds"%(time.time()-start)
start = time.time();
y2 = F.max_pooling_2d(x, 3, stride=2, pad=0)
print "second pass %f seconds"%(time.time()-start)
