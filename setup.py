#!/usr/bin/env python

from setuptools import setup
from setuptools.extension import Extension
#from Cython.Build import cythonize

import numpy

setup_requires = []
install_requires = [
    'filelock',
    'nose',
    'numpy>=1.9.0',
    'protobuf',
    'six>=1.9.0',
    'glog',
]

#extensions = [
#    Extension(
#        "chainer.cpu_accl.c_numeric_c",
#        ["chainer/cpu_accl/c_numeric_c.pyx"],
#        include_dirs=['/usr/local/include'],
#        #include_dirs=['/opt/intel/mklml_lnx_2017.0.1.20161005/include/'],
#        libraries=['mklml_intel', 'iomp5', 'mkldnn'],
#        library_dirs=['/usr/local/lib/'],
#        #library_dirs=['/opt/intel/mklml_lnx_2017.0.1.20161005/lib/'],
#    ),
#]

incdir = numpy.get_include()

extensions = [
    #Extension(
    #    "mkldnnpy.dnn_raw",
    #    sources=["mkldnnpy/dnn_raw.c", "mkldnnpy/dnn_raw.i"]
    #),
    #Extension(
    #    "mkldnnpy._cos_module",
    #    sources=["mkldnnpy/cos_module.c", "mkldnnpy/cos_module.i"]
    #),
    #Extension(
    #    "mkldnnpy._sin_module",
    #    sources=["mkldnnpy/sin_module.c", "mkldnnpy/sin_module.i"]
    #),
    Extension(
        "_mkldnn",
        sources=[
                "mkldnn/mkldnn_conv.cc",
                "mkldnn/common.cc",
                "mkldnn/stream_factory.cc",
                "mkldnn/pooling.cc",
                "mkldnn/mkldnn.i"
                ],
        swig_opts=["-c++"],
        extra_compile_args=["-std=c++11"],
        include_dirs=[incdir],
        libraries=['glog', 'stdc++', 'mkldnn'],
    )
]

setup(
    name='chainer',
    version='2.0.0a1',
    description='A flexible framework of neural networks',
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='http://chainer.org/',
    license='MIT License',
    packages=['chainer',
              'chainer.dataset',
              'chainer.datasets',
              'chainer.functions',
              'chainer.functions.activation',
              'chainer.functions.array',
              'chainer.functions.caffe',
              'chainer.functions.connection',
              'chainer.functions.evaluation',
              'chainer.functions.loss',
              'chainer.functions.math',
              'chainer.functions.noise',
              'chainer.functions.normalization',
              'chainer.functions.pooling',
              'chainer.functions.theano',
              'chainer.functions.util',
              'chainer.function_hooks',
              'chainer.iterators',
              'chainer.initializers',
              'chainer.links',
              'chainer.links.activation',
              'chainer.links.caffe',
              'chainer.links.caffe.protobuf2',
              'chainer.links.caffe.protobuf3',
              'chainer.links.connection',
              'chainer.links.loss',
              'chainer.links.model',
              'chainer.links.model.vision',
              'chainer.links.normalization',
              'chainer.links.theano',
              'chainer.optimizers',
              'chainer.serializers',
              'chainer.testing',
              'chainer.training',
              'chainer.training.extensions',
              'chainer.training.triggers',
              'chainer.utils',
              #'chainer.cpu_accl',
              'mkldnn',
              ],
    #ext_modules = cythonize(extensions),
    ext_modules = extensions,
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=['mock',
                   'nose'],
)
