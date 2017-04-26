import numpy
from . import mkldnn
is_from_chain = False
enable_conv = True
enable_max_pooling = True
enable_avg_pooling = True
enable_lrn = True
enable_relu = True
enable_softmax = False
enable_linear = True
enable_softmax_cross_entropy = False
enable_concat = True
enable_acc_grad = True
supportTypes = (numpy.float32,)


def SupportedInput(tul):
    isSupportType = True
    for x in tul:
        if len(x) == 0:
            continue
        if x[0].dtype not in supportTypes:
            isSupportType = False
            break
        else:
            isSupportType = True
    return isSupportType


def enable_convF(tul):
        return mkldnn.enabled() and SupportedInput(tul) and enable_conv


def enable_max_poolingF(tul):
        return mkldnn.enabled() and SupportedInput(tul) and enable_max_pooling


def enable_avg_poolingF(tul):
        return mkldnn.enabled() and SupportedInput(tul) and enable_avg_pooling


def enable_lrnF(tul):
        return mkldnn.enabled() and SupportedInput(tul) and enable_lrn


def enable_reluF(tul):
        return mkldnn.enabled() and SupportedInput(tul) and enable_relu


def enable_softmaxF(tul):
        return mkldnn.enabled() and SupportedInput(tul) and enable_softmax


def enable_linearF(tul):
        return mkldnn.enabled() and SupportedInput(tul) and enable_linear


def enable_softmax_cross_entropyF(tul):
        return mkldnn.enabled() and SupportedInput(tul) and enable_softmax_cross_entropy


def enable_concatF(tul):
        return mkldnn.enabled() and SupportedInput(tul) and enable_concat


def enable_acc_gradF(tul):
        return mkldnn.enabled() and SupportedInput(tul) and enable_acc_grad
