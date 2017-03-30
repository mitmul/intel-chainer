#!/usr/bin/env python

from setuptools import setup
from setuptools.extension import Extension
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

extensions = [
    Extension(
        "mkldnn._mkldnn",
        sources=[
                "mkldnn/relu4d.cc",
                "mkldnn/relu.cc",
                "mkldnn/mkldnn_conv.cc",
                "mkldnn/mkldnn_concat.cc",
                "mkldnn/common.cc",
                "mkldnn/cpu_info.cc",
                "mkldnn/layer_factory.cc",
                "mkldnn/linear.cc",
                "mkldnn/local_response_normalization.cc",
                "mkldnn/pooling.cc",
                "mkldnn/max_pooling.cc",
                "mkldnn/avg_pooling.cc",
                "mkldnn/mkldnn_softmax.cc",
                "mkldnn/softmax_cross_entropy.cc",
                "mkldnn/mkldnn.i"
                ],
        swig_opts=["-c++"],
        extra_compile_args=["-std=c++11", "-fopenmp"],
        include_dirs=["mkldnn/incl/", numpy.get_include()],
        libraries=['glog', 'stdc++', 'boost_system', 'mkldnn', 'm'],
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
              'mkldnn',
              'mkldpy',
              ],
    ext_modules = extensions,
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=['mock',
                   'nose'],
)
