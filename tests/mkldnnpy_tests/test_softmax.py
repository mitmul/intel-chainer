from mkldnn import mkldnn
import numpy as np

print(mkldnn.enabled())

# x = np.array([[2.0, 3.0, 3.0], [1.0, 1.0, 2.0]])
# x_4d = np.ones((2,3,3,4), dtype=np.float32)
# x = np.ones((2,3), dtype=np.float32)
# y_4d = np.ones((2,3,3,4), dtype=np.float32)
# y = np.ones((2,3), dtype=np.float32)

x_2d = np.arange(6.0, dtype=np.float32).reshape(2,3)
print "x_2d shape: ", x_2d.shape
print x_2d
y_2d = np.ones((2,3), dtype=np.float32)
print y_2d

softmax1 = mkldnn.Softmax_F32_softmax_create_forward(x_2d.ravel(), y_2d.ravel(), x_2d.shape, 1);
ret = softmax1.forward()
print y_2d
ret = softmax1.backward()

x_2d = np.arange(1.0, 13.0, 2.0, dtype=np.float32).reshape(2,3)
print "x_2d shape: ", x_2d.shape
print x_2d
y_2d = np.ones((2,3), dtype=np.float32)
print y_2d

softmax2 = mkldnn.Softmax_F32_softmax_create_forward(x_2d.ravel(), y_2d.ravel(), x_2d.shape, 1);
ret = softmax2.forward()
print y_2d
ret = softmax2.backward()

x_2d1 = np.arange(1.0, 13.0, 2.0, dtype=np.float32).reshape(1,6)
print "x_2d1 shape: ", x_2d1.shape
print x_2d1
y_2d1 = np.ones((1,6), dtype=np.float32)
print y_2d1

softmax3 = mkldnn.Softmax_F32_softmax_create_forward(x_2d1.ravel(), y_2d1.ravel(), x_2d1.shape, 1);
ret = softmax3.forward()
print y_2d1
ret = softmax3.backward()

print(ret)


