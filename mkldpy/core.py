import numpy as np

class ndarray:
    def __init__(self, shape, dtype=float, memptr=None, order='C'):
        self._shape = shape
	self.dtype = dtype
	self.data = memptr
	self.base = None
	self._c_contiguous = True
	self._f_contiguous = False
	self.array = np.ndarray(shape, dtype = self.dtype);

    def fill(self, value):
        self.array.fill(value)
