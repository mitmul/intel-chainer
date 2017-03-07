# define original API of MKL
cdef extern from "mkl_dnn.h":
    ctypedef struct dnnError_t:
        pass
    ctypedef struct dnnLayout_t:
        pass
    ctypedef struct dnnPrimitive_t:
        pass
    ctypedef struct dnnResourceType_t:
        pass
    ctypedef struct dnnPrimitiveAttributes_t:
        pass
    ctypedef struct dnnAlgorithm_t:
        pass
    ctypedef struct dnnBorder_t:
        pass
    dnnError_t dnnLayoutCreate_F32(dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[])
    dnnError_t dnnAllocateBuffer_F32(void **pPtr, dnnLayout_t layout)
    dnnError_t dnnReleaseBuffer_F32(void *ptr)
    int dnnLayoutCompare_F32(const dnnLayout_t l1, const dnnLayout_t l2)
    dnnError_t dnnConversionCreate_F32(dnnPrimitive_t* pConversion, const dnnLayout_t _from, const dnnLayout_t _to)
    dnnError_t dnnConvolutionCreateForward_F32(dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType)
    dnnError_t dnnConvolutionCreateForwardBias_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType)
    dnnError_t dnnLayoutCreateFromPrimitive_F32(
        dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type)
    dnnError_t dnnConversionExecute_F32(
        dnnPrimitive_t conversion, void *_from, void *_to)
    dnnError_t dnnExecute_F32(
        dnnPrimitive_t primitive, void *resources[])
    dnnError_t dnnLayoutDelete_F32(
        dnnLayout_t layout)
