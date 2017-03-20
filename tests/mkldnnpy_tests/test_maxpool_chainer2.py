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
gy = np.ndarray((1, 1, 4, 4), dtype=np.float32)
gy.fill(0.01)

x=x,
gy=gy,

print "first pass"
print "x="
print x[0]
f = F.MaxPooling2D(3, stride=1, pad =1, use_cudnn=False)
y=f.forward_cpu(x)
print "y="
print y[0]
gx=f.backward_cpu(x, gy)
print "gx="
print gx[0]


print "second pass"
print "x="
print x[0]
y=f.forward_cpu(x)
print "y="
print y[0]
gx = f.backward_cpu(x, gy)
print "gx="
print gx[0]
