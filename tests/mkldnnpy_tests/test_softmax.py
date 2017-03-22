from mkldnn import mkldnn
import numpy as np
import chainer.functions as F

print(mkldnn.enabled())

# x = np.array([[2.0, 3.0, 3.0], [1.0, 1.0, 2.0]])
# x_4d = np.ones((2,3,3,4), dtype=np.float32)
# x = np.ones((2,3), dtype=np.float32)
# y_4d = np.ones((2,3,3,4), dtype=np.float32)
# y = np.ones((2,3), dtype=np.float32)

print "TEST1"
x_2d = np.arange(6.0, dtype=np.float32).reshape(2,3)
print "x_2d shape: ", x_2d.shape
print x_2d
y_2d = np.ones((2,3), dtype=np.float32)
print y_2d

softmax1 = mkldnn.Softmax_F32_softmax_create_forward(x_2d.ravel(), y_2d.ravel(), x_2d.shape, 1);
ret = softmax1.forward()
print y_2d

print "TEST2"
x_2d = np.arange(1.0, 13.0, 2.0, dtype=np.float32).reshape(2,3)
print "x_2d shape: ", x_2d.shape
print x_2d
y_2d = np.ones((2,3), dtype=np.float32)
print y_2d

softmax2 = mkldnn.Softmax_F32_softmax_create_forward(x_2d.ravel(), y_2d.ravel(), x_2d.shape, 1);
ret = softmax2.forward()
print y_2d

print "TEST3"
x_2d1 = np.arange(1.0, 13.0, 2.0, dtype=np.float32).reshape(1,6)
print "x_2d1 shape: ", x_2d1.shape
print x_2d1
y_2d1 = np.ones((1,6), dtype=np.float32)
print y_2d1

softmax3 = mkldnn.Softmax_F32_softmax_create_forward(x_2d1.ravel(), y_2d1.ravel(), x_2d1.shape, 1);
ret = softmax3.forward()
print y_2d1

print "TEST4"
x_2d = np.arange(1.0, 13.0, 2.0, dtype=np.float32).reshape(2,3)
print "x_2d shape: ", x_2d.shape
print x_2d
y_2d = np.ones((2,3), dtype=np.float32)
print y_2d

softmax_cross_entropy1 = mkldnn.SoftmaxCrossEntropy_F32_softmax_cross_entropy_create_forward(x_2d.shape)
ret = softmax_cross_entropy1.forward(x_2d.ravel(), y_2d.ravel(), x_2d.shape)
print y_2d

print "TEST5"
x_2d1 = np.arange(1.0, 13.0, 2.0, dtype=np.float32).reshape(1,6)
print "x_2d1 shape: ", x_2d1.shape
print x_2d1
y_2d1 = np.ones((1,6), dtype=np.float32)
print y_2d1

softmax_cross_entropy1 = mkldnn.SoftmaxCrossEntropy_F32_softmax_cross_entropy_create_forward(x_2d.shape)
ret = softmax_cross_entropy1.forward(x_2d.ravel(), y_2d.ravel(), x_2d.shape)
print y_2d

print "TEST6"
gx_2d = np.arange(1.0, 13.0, 2.0, dtype=np.float32).reshape(2,3)
print "gx_2d shape: ", gx_2d.shape
print gx_2d
label = np.array([0, 2], dtype=np.int32)
print label

softmax_cross_entropy1 = mkldnn.SoftmaxCrossEntropy_F32_softmax_cross_entropy_create_backward(x_2d.shape)
ret = softmax_cross_entropy1.backward(gx_2d.ravel(), label.ravel(), gx_2d.shape)
print gx_2d

print "TEST7"
x_2d = np.arange(1.0, 13.0, 2.0, dtype=np.float32).reshape(2,3)
print "x_2d shape: ", x_2d.shape
print x_2d
label = np.array([0, 2], dtype=np.int32)
print label

y_2d = F.softmax_cross_entropy(x_2d, label, use_cudnn=False)
print y_2d.data

print(ret)


