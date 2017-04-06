import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import math
from mkldnn import switch

#input = [   [ 1, 2, 3 ],
#            [ 4, 5, 6 ],
#            [ 7, 8, 9 ] ]

(n,c,h,w)=(2,3,4,3)
x = np.arange(n*c*h*w, dtype=np.float32).reshape(n, c, h, w)
np.random.shuffle(x)
x = 2*x/x.size -1
print "x="
print x


switch.enable_max_pooling = True
y1 = F.max_pooling_2d(x, 3, stride=2, pad=1, cover_all=False)
switch.enable_max_pooling = False
y2 = F.max_pooling_2d(x, 3, stride=2, pad=1, cover_all=False)

ydiff = y1-y2
print "ydiff(cover_all==False)="
print ydiff.data

switch.enable_max_pooling = True
y1 = F.max_pooling_2d(x, 3, stride=2, pad=1, cover_all=True)
switch.enable_max_pooling = False
y2 = F.max_pooling_2d(x, 3, stride=2, pad=1, cover_all=True)

ydiff = y1-y2
print "ydiff(cover_all==True)="
print ydiff.data
