from nose.plugins import attrib

gpu = attrib.attr('gpu')
cudnn = attrib.attr('gpu', 'cudnn')
slow = attrib.attr('slow')
xeon = attrib.attr('xeon')
xeon_phi = attrib.attr('xeon_phi')

def multi_gpu(gpu_num):
    return attrib.attr(gpu=gpu_num)
