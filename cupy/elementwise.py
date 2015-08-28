import string

import numpy
import six

import cupy
from cupy import carray
from cupy import cuda
from cupy import util


six_range = six.moves.range
six_zip = six.moves.zip


def _get_simple_elementwise_kernel(
        params, operation, name, preamble,
        loop_prep='', after_loop='', options=()):
    module_code = string.Template('''
    ${preamble}
    extern "C" __global__ void ${name}(${params}) {
      ${loop_prep};
      CUPY_FOR(i, _ind.size()) {
        _ind.set(i);
        ${operation};
      }
      ${after_loop};
    }
    ''').substitute(
        params=params,
        operation=operation,
        name=name,
        preamble=preamble,
        loop_prep=loop_prep,
        after_loop=after_loop)
    module = carray.compile_with_cache(module_code, options)
    return module.get_function(name)


_typenames = {
    numpy.dtype('float64'): 'double',
    numpy.dtype('float32'): 'float',
    numpy.dtype('float16'): 'float16',
    numpy.dtype('int64'): 'long long',
    numpy.dtype('int32'): 'int',
    numpy.dtype('int16'): 'short',
    numpy.dtype('int8'): 'signed char',
    numpy.dtype('uint64'): 'unsigned long long',
    numpy.dtype('uint32'): 'unsigned int',
    numpy.dtype('uint16'): 'unsigned short',
    numpy.dtype('uint8'): 'unsigned char',
    numpy.dtype('bool'): 'bool',
}


_scalar_type = (int, float, bool) + tuple(t.type for t in _typenames.keys())


def _get_typename(dtype):
    if dtype is None:
        raise ValueError('dtype is None')
    return _typenames[numpy.dtype(dtype)]


def _check_args(args):
    dev = cuda.Device()
    cp_array = cupy.ndarray
    scalar_type = _scalar_type
    for arg in args:
        if isinstance(arg, cp_array):
            if not arg.data.device == dev:
                raise ValueError('Array device must be same as the current '
                                 'device: array device = %d while current = %d'
                                 % (arg.data.device.id, dev.id))
        elif not isinstance(arg, scalar_type):
            raise TypeError('Unsupported type %s' % type(arg))


def _get_args_info(args):
    return tuple([(type(a), a.dtype, a.ndim) for a in args])


def _get_kernel_params(params, args_info):
    ret = []
    for p, a in six_zip(params, args_info):
        type, dtype, ndim = a
        is_array = type is cupy.ndarray
        if type is carray.Indexer:
            t = 'CIndexer<%d>' % ndim
        else:
            t = _get_typename(dtype)
            if is_array:
                t = 'CArray<%s, %d>' % (t, ndim)
        ret.append('%s%s %s%s' % ('const ' if p.is_const else '',
                                  t,
                                  '_raw_' if is_array and not p.raw else '',
                                  p.name))
    return ', '.join(ret)


def _reduce_dims(args, params, indexer):
    ndim = indexer.ndim
    if ndim <= 1:
        return args, indexer
    is_array_flags = [not p.raw and isinstance(a, cupy.ndarray)
                      for a, p in six_zip(args, params)]
    args_strides = [a.strides
                    for a, f in six_zip(args, is_array_flags) if f]
    shape = list(indexer.shape)
    for i in six_range(1, len(shape)):
        j = i - 1
        for strides in args_strides:
            if strides[i] * shape[i] != strides[j]:
                break
        else:
            shape[i] *= shape[j]
            shape[j] = 1

    axis = None
    for i, sh in enumerate(shape):
        if sh != 1:
            if axis is None:
                axis = i
            else:
                break
    else:
        if axis is not None:
            indexer.shape = new_shape = shape[axis],
            args = list(args)
            for i, arg in enumerate(args):
                if is_array_flags[i]:
                    args[i] = arg = arg.view()
                    arg._shape = new_shape
                    arg._strides = arg._strides[axis],
        return args, indexer

    indexer.shape = new_shape = tuple([dim for dim in shape if dim != 1])
    args = list(args)
    for i, arg in enumerate(args):
        if is_array_flags[i]:
            args[i] = arg = arg.view()
            arg._shape = new_shape
            arg._strides = tuple(
                [st for st, sh in six_zip(arg.strides, shape)
                 if sh != 1])
    return args, indexer


class ParameterInfo(object):

    def __init__(self, str, is_const):
        self.name = None
        self.dtype = None
        self.ctype = None
        self.raw = False
        self.is_const = is_const
        s = tuple(i for i in str.split() if len(i) != 0)
        if len(s) < 2:
            raise Exception('Syntax error: %s' % str)

        t, self.name = s[-2:]
        if t == 'CIndexer':
            pass
        elif len(t) == 1:
            self.ctype = t
        else:
            self.dtype = numpy.dtype(t)
            if self.dtype.name != t:
                raise ValueError('Wrong type %s' % t)
            self.ctype = _get_typename(self.dtype)

        for i in s[:-2]:
            if i == 'raw':
                self.raw = True
            else:
                raise Exception('Unknown keyward "%s"' % i)


@util.memoize()
def _get_param_info(s, is_const):
    if len(s) == 0:
        return ()
    return tuple([ParameterInfo(i, is_const) for i in s.strip().split(',')])


@util.memoize()
def _decide_params_type(in_params, out_params, in_args_dtype, out_args_dtype):
    type_dict = {}
    if out_args_dtype:
        assert len(out_params) == len(out_args_dtype)
        for p, a in six_zip(out_params, out_args_dtype):
            if a is None:
                raise TypeError('Output arguments must be cupy.ndarray')
            if p.dtype is not None:
                if a != p.dtype:
                    raise TypeError(
                        'Type is mismatched. %s %s %s' % (p.name, a, p.dtype))
            elif p.ctype in type_dict:
                t = type_dict[p.ctype]
                if t != a:
                    raise TypeError(
                        'Type is mismatched. %s %s %s %s' % (
                            p.name, a, t, p.ctype))
            else:
                type_dict[p.ctype] = a

    assert len(in_params) == len(in_args_dtype)
    unknown_ctype = []
    for p, a in six_zip(in_params, in_args_dtype):
        if a is None:
            if p.dtype is None:
                unknown_ctype.append(p.ctype)
        else:
            if p.dtype is not None:
                if a != p.dtype:
                    raise TypeError(
                        'Type is mismatched. %s %s %s' % (p.name, a, p.dtype))
            elif p.ctype in type_dict:
                t = type_dict[p.ctype]
                if t != a:
                    raise TypeError(
                        'Type is mismatched. %s %s %s %s' % (
                            p.name, a, t, p.ctype))
            else:
                type_dict[p.ctype] = a

    in_types = tuple([type_dict[p.ctype] if p.dtype is None else p.dtype
                      for p in in_params])
    out_types = tuple([type_dict[p.ctype] if p.dtype is None else p.dtype
                       for p in out_params])
    return in_types, out_types, tuple(type_dict.items())


def _broadcast(args, params, size_error=True):
    brod = cupy.broadcast(
        *[a if not p.raw and isinstance(a, cupy.ndarray) else None
          for p, a in six_zip(params, args)])
    if size_error and all(i is None for i in brod.values):
        raise ValueError('Loop size is Undecided')
    value = [b if a is None else a
             for a, b in six_zip(brod.values, args)]
    return value, brod.shape


def _get_out_args(out_args, out_types, out_shape):
    if not out_args:
        return [cupy.empty(out_shape, t) for t in out_types]

    for a in out_args:
        if not isinstance(a, cupy.ndarray):
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        if a.shape != out_shape:
            raise ValueError('Out shape is mismatched')
    return out_args


def _get_out_args_with_params(out_args, out_types, out_shape, out_params):
    if not out_args:
        for p in out_params:
            if p.raw:
                raise ValueError('Output array size is Undecided')
        return [cupy.empty(out_shape, t) for t in out_types]

    for a, p in six_zip(out_args, out_params):
        if not isinstance(a, cupy.ndarray):
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        if a.shape != out_shape and not p.raw:
            raise ValueError('Out shape is mismatched')
    return out_args


@util.memoize(for_each_device=True)
def _get_elementwise_kernel(
        args_info, types, params, operation, name,
        preamble, **kwargs):
    kernel_params = _get_kernel_params(params, args_info)
    types_preamble = '\n'.join(
        'typedef %s %s;' % (_get_typename(v), k) for k, v in types)
    preamble = types_preamble + '\n' + preamble

    op = []
    for p, a in six_zip(params, args_info):
        if not p.raw and a[0] == cupy.ndarray:
            if p.is_const:
                fmt = 'const {t} {n} = _raw_{n}[_ind.get()];'
            else:
                fmt = '{t} &{n} = _raw_{n}[_ind.get()];'
            op.append(fmt.format(t=p.ctype, n=p.name))
    op.append(operation)
    operation = '\n'.join(op)
    return _get_simple_elementwise_kernel(
        kernel_params, operation, name,
        preamble, **kwargs)


class ElementwiseKernel(object):

    """User-defined elementwise kernel.

    This class can be used to define an elementwise kernel with or without
    broadcasting.

    The kernel is compiled at an invocation of the
    :meth:`~ElementwiseKernel.__call__` method,
    which is cached for each device.
    The compiled binary is also cached into a file under the
    ``$HOME/.cupy/kernel_cache/`` directory with a hashed file name. The cached
    binary is reused by other processes.

    Args:
        in_params (str): Input argument list.
        out_params (str): Output argument list.
        operation (str): The body in the loop written in CUDA-C/C++.
        name (str): Name of the kernel function. It should be set for
            readability of the performance profiling.
        reduce_dims (bool): If False, the shapes of array arguments are
            kept within the kernel invocation. The shapes are reduced
            (i.e., the arrays are reshaped without copy to the minimum
            ndims) by default. It may make the kernel fast by reducing the
            index calculations.
        options (list): Options passed to the nvcc command.
        preamble (str): Fragment of the CUDA-C/C++ code that is inserted at the
            top of the cu file.
        loop_prep (str): Fragment of the CUDA-C/C++ code that is inserted at
            the top of the kernel function definition and above the ``for``
            loop.
        after_loop (str): Fragment of the CUDA-C/C++ code that is inserted at
            the bottom of the kernel function definition.

    """
    def __init__(self, in_params, out_params, operation,
                 name='kernel', reduce_dims=True, preamble='', **kwargs):
        self.in_params = _get_param_info(in_params, True)
        self.out_params = _get_param_info(out_params, False)
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        param_rest = _get_param_info('CIndexer _ind', False)
        self.params = self.in_params + self.out_params + param_rest
        self.operation = operation
        self.name = name
        self.reduce_dims = reduce_dims
        self.preamble = preamble
        self.kwargs = kwargs
        names = [p.name for p in self.in_params + self.out_params]
        if 'i' in names:
            raise ValueError("Can not use 'i' as a parameter name")

    def __call__(self, *args, **kwargs):
        """Compiles and invokes the elementwise kernel.

        The compilation runs only if the kernel is not cached. Note that the
        kernels with different argument dtypes or ndims are not compatible. It
        means that single ElementwiseKernel object may be compiled into
        multiple kernel binaries.

        Args:
            args: Argumens of the kernel.
            size (int): Range size of the indices. If specified, the variable
                ``n`` is set to this value. Otherwise, the result of
                broadcasting is used to determine the value of ``n``.

        Returns:
            Arrays are returned according to the ``out_params`` argument of the
            ``__init__`` method.

        """
        n = kwargs.pop('size', None)
        np_array = numpy.ndarray
        cp_array = cupy.ndarray

        if not (len(args) == self.nin or
                len(args) == self.nin + self.nout):
            raise TypeError('Wrong number of arguments for %s' % self.name)
        for i in args:
            if isinstance(i, np_array):
                raise TypeError('Unsupported type %s' % type(i))
        _check_args(args)

        values, shape = _broadcast(args, self.params, n is None)
        in_args = values[:self.nin]
        out_args = values[self.nin:]
        in_ndarray_types = tuple(
            [a.dtype if isinstance(a, cp_array) else None for a in in_args])
        out_ndarray_types = tuple(
            [a.dtype if isinstance(a, cp_array) else None for a in out_args])

        in_types, out_types, types = _decide_params_type(
            self.in_params, self.out_params,
            in_ndarray_types, out_ndarray_types)

        ret = out_args = _get_out_args_with_params(
            out_args, out_types, shape, self.out_params)
        if len(ret) == 1:
            ret = ret[0]

        if n is None:
            indexer = carray.Indexer(shape)
        else:
            indexer = carray.Indexer((n,))

        if indexer.size == 0:
            return ret

        inout_args = [x if isinstance(x, cp_array) else t.type(x)
                      for x, t in six_zip(in_args, in_types)]
        inout_args += out_args

        if self.reduce_dims:
            inout_args, indexer = _reduce_dims(
                inout_args, self.params, indexer)
        inout_args.append(indexer)

        args_info = _get_args_info(inout_args)
        kern = _get_elementwise_kernel(
            args_info, types, self.params, self.operation,
            self.name, self.preamble, **self.kwargs)
        kern.linear_launch(indexer.size, inout_args)
        return ret


@util.memoize(for_each_device=True)
def _get_ufunc_kernel(in_types, out_types, routine, args_info, out_raw_types,
                      params, name, preamble):
    kernel_params = _get_kernel_params(params, args_info)

    types = []
    op = []
    for i, x in enumerate(in_types):
        types.append('typedef %s in%d_type;' % (_get_typename(x), i))
        if args_info[i][0] is cupy.ndarray:
            op.append(
                'const in{0}_type in{0} = _raw_in{0}[_ind.get()];'.format(i))

    for i, x in enumerate(out_types):
        types.append('typedef %s out%d_type;' % (_get_typename(x), i))
        op.append('{1} &out{0} = _raw_out{0}[_ind.get()];'.format(
            i, _get_typename(out_raw_types[i])))

    op.append(routine)
    operation = '\n'.join(op)

    types.append(preamble)
    preamble = '\n'.join(types)

    return _get_simple_elementwise_kernel(
        kernel_params, operation, name, preamble)


def _guess_routine_from_in_types(ops, in_types):
    for op in ops:
        for t0, t1 in six_zip(in_types, op[0]):
            if not numpy.can_cast(t0, t1):
                break
        else:
            return op
    return None


def _guess_routine_from_dtype(ops, dtype):
    for op in ops:
        for t in op[1]:
            if t != dtype:
                break
        else:
            return op
    return None


def _guess_routine(name, cache, ops, in_args, dtype):
    key = dtype
    if dtype is None:
        key = tuple([numpy.dtype(type(i))
                     if isinstance(i, (int, float, bool)) else i.dtype
                     for i in in_args])

    op = cache.get(key, ())
    if op is ():
        if dtype is None:
            op = _guess_routine_from_in_types(ops, key)
        else:
            op = _guess_routine_from_dtype(ops, key)
        cache[key] = op

    if op:
        return op
    raise TypeError('Wrong type of arguments for %s' % name)


class ufunc(object):

    """Universal function.

    Attributes:
        name (str): The name of the universal function.
        nin (int): Number of input arguments.
        nout (int): Number of output arguments.
        nargs (int): Number of all arguments.

    """
    def __init__(self, name, nin, nout, ops, preamble='', doc=''):
        self.name = name
        self.nin = nin
        self.nout = nout
        self.nargs = nin + nout
        self._ops = ops
        self._preamble = preamble
        self.__doc__ = doc
        _in_params = tuple(
            ParameterInfo('T in%d' % i, True)
            for i in six_range(nin))
        _out_params = tuple(
            ParameterInfo('T out%d' % i, False)
            for i in six_range(nout))
        self._params = _in_params + _out_params + (
            ParameterInfo('CIndexer _ind', False),)
        self._routine_cache = {}

    def __repr__(self):
        return "<ufunc '%s'>" % self.name

    @property
    def types(self):
        """A list of type signatures.

        Each type signature is represented by type character codes of inputs
        and outputs separated by '->'.

        """
        types = []
        for in_types, out_types, _ in self._ops:
            in_str = ''.join([t.char for t in in_types])
            out_str = ''.join([t.char for t in out_types])
            types.append('%s->%s' % (in_str, out_str))
        return types

    def __call__(self, *args, **kwargs):
        """Applies the universal function to arguments elementwise.

        Args:
            args: Input arguments. Each of them can be a cupy.ndarray object or
                a scalar. The output arguments can be omitted or be specified
                by the ``out`` argument.
            out (cupy.ndarray): Output array. It outputs to new arrays
                default.
            dtype: Data type specifier.

        Returns:
            Output array or a tuple of output arrays.

        """
        out = kwargs.get('out', None)
        dtype = kwargs.get('dtype', None)
        if dtype is not None:
            dtype = numpy.dtype(dtype)

        if not (len(args) == self.nin or len(args) == self.nargs):
            raise TypeError('Wrong number of arguments for %s' % self.name)

        in_args = list(args[:self.nin])
        out_args = list(args[self.nin:])
        if out is not None:
            if len(out_args) != 0:
                raise ValueError("cannot specify 'out' as both "
                                 "a positional and keyword argument")
            out_args = [out]

        _check_args(in_args + out_args)

        in_types, out_types, routine = _guess_routine(
            self.name, self._routine_cache, self._ops, in_args, dtype)

        broad = cupy.broadcast(*(in_args + out_args))

        ret = out_args = _get_out_args(out_args, out_types, broad.shape)
        if len(ret) == 1:
            ret = ret[0]

        if 0 in broad.shape:
            return ret

        inout_args = [x if isinstance(x, cupy.ndarray) else t.type(x)
                      for x, t
                      in six_zip(broad.values, in_types)]
        inout_args += out_args
        inout_args, indexer = _reduce_dims(
            inout_args, self._params, carray.Indexer(broad.shape))
        inout_args.append(indexer)
        args_info = _get_args_info(inout_args)
        out_raw_types = tuple([x.dtype for x in out_args])
        kern = _get_ufunc_kernel(
            in_types, out_types, routine,
            args_info, out_raw_types,
            self._params, self.name, self._preamble)

        kern.linear_launch(indexer.size, inout_args)
        return ret


def create_ufunc(name, ops, routine=None, preamble='', doc=''):
    _ops = []
    for t in ops:
        if not isinstance(t, tuple):
            typ = t
            rt = routine
        else:
            typ, rt = t

        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = tuple([numpy.dtype(t) for t in in_types])
        out_types = tuple([numpy.dtype(t) for t in out_types])
        _ops.append((in_types, out_types, rt))

    return ufunc(name, len(_ops[0][0]), len(_ops[0][1]), _ops, preamble, doc)


_id = 'out0 = in0'

copy = create_ufunc(
    'cupy_copy',
    ['?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d'],
    _id)


copy_where = create_ufunc(
    'cupy_copy_where',
    ['??->?', 'b?->b', 'B?->B', 'h?->h', 'H?->H', 'i?->i', 'I?->I', 'l?->l',
     'L?->L', 'q?->q', 'Q?->Q', 'e?->e', 'f?->f', 'd?->d'],
    'if (in1) out0 = in0')


_divmod = create_ufunc(
    'cupy_divmod',
    ['bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
     'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d'],
    'out0_type a = _floor_divide(in0, in1); out0 = a; out1 = in0 - a * in1')
