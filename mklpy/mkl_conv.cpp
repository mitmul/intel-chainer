#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>

#include "mkl_dnn.h"
#include "mkl_ops.h"

static int init_conversion(dnnPrimitive_t *cv, float **ptr_out,
                                 dnnLayout_t lt_pr, dnnLayout_t lt_us, float *ptr_us)
{
    int err = E_SUCCESS;
    *ptr_out = NULL;
    if (!dnnLayoutCompare_F32(lt_pr, lt_us)) {
        CHECK_ERR( dnnConversionCreate_F32(cv, lt_us, lt_pr), err );
        CHECK_ERR( dnnAllocateBuffer_F32((void**)ptr_out, lt_pr), err );
    } else {
        *ptr_out = ptr_us;
    }
    return err;

bail_out:
    if (*ptr_out) dnnReleaseBuffer_F32(*ptr_out);
    return err;
}


int _conv_forward(float* image_pbuf, 
                  float* weight_pbuf,
                  size_t* image_shape, 
                  size_t* weight_shape,
                  int* stride, 
                  int* pad,
                  float* z_pbuf)
{
    int err;

    int has_bias = 0;
    float* bias_pbuf = NULL;
    if (bias_pbuf != NULL)
        has_bias = 1;

    assert (image_pbuf != NULL);
    assert (weight_pbuf != NULL);
    assert (z_pbuf != NULL);

    int n_image = image_shape[0];
    int c_image = image_shape[1];
    int h_image = image_shape[2];
    int w_image = image_shape[3];

    int k_weight = weight_shape[0];
    int c_weight = weight_shape[1];
    int h_weight = weight_shape[2];
    int w_weight = weight_shape[3];

    int h_stride = stride[0];
    int w_stride = stride[1];

    int h_pad = pad[0];
    int w_pad = pad[1];

    assert (c_image == c_weight);

    // calculate output shape
    int n_z = n_image;
    int c_z = k_weight;
    int h_z = (h_image + 2 * h_pad - h_weight) / h_stride + 1;
    int w_z = (w_image + 2 * w_pad - w_weight) / w_stride + 1;

    // double gflops = (double)(n_image * k_weight * h_z * w_z) * (c_image * h_weight * w_weight * 2) / (1e9);

    // calculate shape info for MKL API
    size_t imageSize[DIMENSION] = {w_image, h_image, c_image, n_image};
    size_t imageStrides[DIMENSION] = {1, w_image, h_image * w_image, h_image * w_image * c_image};

    size_t weightSize[DIMENSION] = {w_weight, h_weight, c_weight, k_weight};
    size_t weightStrides[DIMENSION] = {1, w_weight, h_weight * w_weight, h_weight * w_weight * c_weight};
    
    size_t outputSize[DIMENSION] = {w_z, h_z, c_z, n_z};
    size_t outputStrides[DIMENSION] = {1, w_z, w_z * h_z, w_z * h_z * c_z};

    size_t convStride[DIMENSION - 2] = {w_stride, h_stride};
    int convPadding[DIMENSION - 2] = {-w_pad, -h_pad};

    size_t biasSize[1];
    size_t biasStrides[1];

    if (has_bias) {
        biasSize[0] = outputSize[2];
        biasStrides[0] = 1;
    }

    float* convRes[dnnResourceNumber] = {NULL};

    dnnPrimitive_t pConvFwd = NULL;
    dnnLayout_t image_layout_user, weight_layout_user, bias_layout_user, z_layout_user;
    dnnLayout_t image_layout_internal, weight_layout_internal, bias_layout_internal, z_layout_internal;
    dnnPrimitive_t image_user_to_internal = NULL,
                   weight_user_to_internal = NULL,
                   bias_user_to_internal = NULL;

    // Create user layout for all the inputs and output

    CHECK_ERR( dnnLayoutCreate_F32(&image_layout_user, DIMENSION, imageSize, imageStrides) , err );
    CHECK_ERR( dnnLayoutCreate_F32(&weight_layout_user, DIMENSION, weightSize, weightStrides), err );
    if (has_bias) {
        CHECK_ERR( dnnLayoutCreate_F32(&bias_layout_user, 1, biasSize, biasStrides) , err );
    }
    CHECK_ERR( dnnLayoutCreate_F32(&z_layout_user, DIMENSION, outputSize, outputStrides), err );

    // Create convolution forward primitive
    if (has_bias) {
        CHECK_ERR( dnnConvolutionCreateForwardBias_F32(&pConvFwd, NULL,
                   dnnAlgorithmConvolutionDirect, DIMENSION, imageSize,
                   outputSize, weightSize, convStride, convPadding,
                   dnnBorderZeros), err);
    } else {
        CHECK_ERR( dnnConvolutionCreateForward_F32(&pConvFwd, NULL,
                   dnnAlgorithmConvolutionDirect, DIMENSION, imageSize,
                   outputSize, weightSize, convStride, convPadding,
                   dnnBorderZeros), err);
    }
#ifdef _MKL_DEBUG_
    MKL_PRINT_PTR(pConvFwd);
#endif

    // Create internal layout for all inputs and output
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&image_layout_internal, pConvFwd, dnnResourceSrc) , err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&weight_layout_internal, pConvFwd, dnnResourceFilter), err );
    if (has_bias) {
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&bias_layout_internal, pConvFwd, dnnResourceBias) , err );
    }
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&z_layout_internal, pConvFwd, dnnResourceDst) , err );

    // Do the layout conversion for all the inputs if needed
    CHECK_ERR( init_conversion(&image_user_to_internal, &convRes[dnnResourceSrc], image_layout_internal, image_layout_user, image_pbuf) , err );
    if (image_user_to_internal) {
#ifdef _MKL_DEBUG_
        printf("image convert\n");
        printf("image_buf: %ld \n", (size_t)image_pbuf);
        printf("dnnResourceSrc: %ld \n", (size_t)convRes[dnnResourceSrc]);
#endif
        CHECK_ERR( dnnConversionExecute_F32(image_user_to_internal, (void*)image_pbuf, convRes[dnnResourceSrc]), err );
#ifdef _MKL_DEBUG_
        printf("[%s:%d], image convert to internal!\n", __FUNCTION__, __LINE__);
        MKL_PRINT_PTR(convRes[dnnResourceSrc]);
#endif
    }

    CHECK_ERR( init_conversion(&weight_user_to_internal, &convRes[dnnResourceFilter], weight_layout_internal, weight_layout_user, weight_pbuf), err );
    if (weight_user_to_internal) {
        CHECK_ERR( dnnConversionExecute_F32(weight_user_to_internal, (void*)weight_pbuf, convRes[dnnResourceFilter]), err );
#ifdef _MKL_DEBUG_
        printf("[%s:%d], weight convert to internal!\n", __FUNCTION__, __LINE__);
        MKL_PRINT_PTR(convRes[dnnResourceFilter]);
#endif
    }

    if (has_bias) {
        CHECK_ERR( init_conversion(&bias_user_to_internal, &convRes[dnnResourceBias], bias_layout_internal, bias_layout_user, bias_pbuf), err );
        if (bias_user_to_internal) {
            CHECK_ERR( dnnConversionExecute_F32(bias_user_to_internal, (void*)bias_pbuf, convRes[dnnResourceBias]) , err );
#ifdef _MKL_DEBUG_
            printf("[%s:%d], bias convert to internal!\n", __FUNCTION__, __LINE__);
            MKL_PRINT_PTR(convRes[dnnResourceBias]);
#endif
        }
    }

    struct timeval start, end;
    // Allocate internal layout buffer for output
    if (!dnnLayoutCompare_F32(z_layout_internal, z_layout_user)) {
        dnnPrimitive_t z_internal_to_user = NULL;
        CHECK_ERR( dnnAllocateBuffer_F32((void**)&convRes[dnnResourceDst], z_layout_internal), err );
#ifdef _MKL_DEBUG_
        gettimeofday(&start, NULL);
#endif
        CHECK_ERR( dnnExecute_F32(pConvFwd, (void**)convRes), err );
#ifdef _MKL_DEBUG_
        gettimeofday(&end, NULL);
        printf("A: dnnExecute_F32 %.8f s \n", ((end.tv_sec - start.tv_sec)*1.0 + (float)(end.tv_usec - start.tv_usec)/1000000.0));
#endif
        CHECK_ERR( dnnConversionCreate_F32(&z_internal_to_user, z_layout_internal, z_layout_user), err );
        CHECK_ERR( dnnConversionExecute_F32(z_internal_to_user, (void*)convRes[dnnResourceDst], (void*)z_pbuf), err );
        dnnDelete_F32(z_internal_to_user);
    } else {
        convRes[dnnResourceDst] = z_pbuf;
#ifdef _MKL_DEBUG_
        gettimeofday(&start, NULL);
#endif
        CHECK_ERR( dnnExecute_F32(pConvFwd, (void**)convRes), err );
#ifdef _MKL_DEBUG_
        gettimeofday(&end, NULL);
        printf("B: dnnExecute_F32 %.8f s \n", ((end.tv_sec - start.tv_sec)*1.0 + (float)(end.tv_usec - start.tv_usec)/1000000.0));
#endif
    }

bail_out:
    dnnDelete_F32(pConvFwd);
    dnnDelete_F32(image_user_to_internal);
    dnnDelete_F32(weight_user_to_internal);
    dnnDelete_F32(bias_user_to_internal);

    dnnLayoutDelete_F32(image_layout_user);
    dnnLayoutDelete_F32(weight_layout_user);
    dnnLayoutDelete_F32(z_layout_user);
    dnnLayoutDelete_F32(image_layout_internal);
    dnnLayoutDelete_F32(weight_layout_internal);
    if (has_bias)
    {
        dnnLayoutDelete_F32(bias_layout_internal);
        dnnLayoutDelete_F32(bias_layout_user);
    }
    dnnLayoutDelete_F32(z_layout_internal);

    dnnReleaseBuffer_F32(convRes[dnnResourceDiffSrc]);
    dnnReleaseBuffer_F32(convRes[dnnResourceDiffFilter]);
    dnnReleaseBuffer_F32(convRes[dnnResourceDiffBias]);

    return err;
}


int _conv_backward(float* image_pbuf, float* weight_pbuf, float* gz_pbuf,
                   size_t* image_shape, size_t* weight_shape,
                   int* stride, int* pad,
                   float* gradImage_pbuf,   // output param
                   float* gradWeight_pbuf,  // output param
                   float* gradBias_pbuf)    // output param
{
    int err;
    float * bias_pbuf = NULL; 
    int has_bias = 0;
    if (bias_pbuf != NULL)
        has_bias = 1;

    assert (image_pbuf != NULL);
    assert (weight_pbuf != NULL);
    // assert (bias_pbuf != NULL);
    assert (gz_pbuf != NULL);
    assert (gradImage_pbuf != NULL);
    assert (gradWeight_pbuf != NULL);
    assert (gradBias_pbuf != NULL);

    int n_image = image_shape[0];
    int c_image = image_shape[1];
    int h_image = image_shape[2];
    int w_image = image_shape[3];

    int k_weight = weight_shape[0];
    int c_weight = weight_shape[1];
    int h_weight = weight_shape[2];
    int w_weight = weight_shape[3];

    int h_stride = stride[0];
    int w_stride = stride[1];

    int h_pad = pad[0];
    int w_pad = pad[1];

    assert (c_image == c_weight);
    // calculate output shape
    int n_z = n_image;
    int c_z = k_weight;
    int h_z = (h_image + 2 * h_pad - h_weight) / h_stride + 1;
    int w_z = (w_image + 2 * w_pad - w_weight) / w_stride + 1;

    // double gflops = (double)(n_image * k_weight * h_z * w_z) * (c_image * h_weight * w_weight * 2) / (1e9);

    // calculate shape info for MKL API
    size_t outputSize[DIMENSION] = {w_z, h_z, c_z, n_z};
    size_t outputStrides[DIMENSION] = {1, w_z, h_z * w_z, h_z * w_z * c_z};

    size_t imageSize[DIMENSION] = {w_image, h_image, c_image, n_image};
    size_t imageStrides[DIMENSION] = {1, w_image, h_image * w_image, h_image * w_image * c_image};

    size_t weightSize[DIMENSION] = {w_weight, h_weight, c_weight, k_weight};
    size_t weightStrides[DIMENSION] = {1, w_weight, h_weight * w_weight, h_weight * w_weight * c_weight};
    
    size_t convStride[DIMENSION - 2] = {w_stride, h_stride};
    int convPadding[DIMENSION - 2] = {-w_pad, -h_pad};

    size_t biasSize[1];
    size_t biasStrides[1];

    if (has_bias) {
        biasSize[0] = outputSize[2];
        biasStrides[0] = 1;
    }

    float *convBwdImageRes[dnnResourceNumber] = {NULL};
    float *convBwdWeightRes[dnnResourceNumber] = {NULL};
    float *convBwdBiasRes[dnnResourceNumber] = {NULL};

    dnnPrimitive_t pConvBwdImage = NULL;
    dnnPrimitive_t pConvBwdWeight = NULL;
    dnnPrimitive_t pConvBwdBias = NULL;

    dnnLayout_t image_layout_user = NULL,
                weight_layout_user = NULL,
                bias_layout_user = NULL,
                gz_layout_user = NULL;

    dnnLayout_t image_layout_internal_bwdd = NULL,
                weight_layout_internal_bwdd = NULL,
                gz_layout_internal_bwdd = NULL,

                image_layout_internal_bwdf = NULL,
                weight_layout_internal_bwdf = NULL,
                gz_layout_internal_bwdf = NULL,

                bias_layout_internal_bwdb = NULL,
                gz_layout_internal_bwdb = NULL;

    dnnPrimitive_t weight_user_to_internal_bwdd = NULL,
                   gz_user_to_internal_bwdd = NULL,
                   image_internal_to_user_bwdd = NULL,

                   image_user_to_internal_bwdf = NULL,
                   gz_user_to_internal_bwdf = NULL,
                   weight_internal_to_user_bwdf = NULL,

                   gz_user_to_internal_bwdb = NULL,
                   bias_internal_to_user = NULL;

    // Create user layout for all the inputs and output
    CHECK_ERR( dnnLayoutCreate_F32(&image_layout_user, DIMENSION, imageSize, imageStrides) , err );
    CHECK_ERR( dnnLayoutCreate_F32(&weight_layout_user, DIMENSION, weightSize, weightStrides), err );
    if (has_bias) {
        CHECK_ERR( dnnLayoutCreate_F32(&bias_layout_user, 1, biasSize, biasStrides) , err );
    }
    CHECK_ERR( dnnLayoutCreate_F32(&gz_layout_user, DIMENSION, outputSize, outputStrides), err );

    { /* ~ gradImage ~ */
    // Create convolution backward image primitive
    CHECK_ERR( dnnConvolutionCreateBackwardData_F32(&pConvBwdImage, NULL,
               dnnAlgorithmConvolutionDirect, DIMENSION, imageSize,
               outputSize, weightSize, convStride, convPadding,
               dnnBorderZeros), err);

    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&image_layout_internal_bwdd, pConvBwdImage, dnnResourceDiffSrc), err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&weight_layout_internal_bwdd, pConvBwdImage, dnnResourceFilter), err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&gz_layout_internal_bwdd, pConvBwdImage, dnnResourceDiffDst) , err );

    // Do the layout conversion for weight and gz if needed
    CHECK_ERR( init_conversion(&weight_user_to_internal_bwdd, &convBwdImageRes[dnnResourceFilter],
               weight_layout_internal_bwdd, weight_layout_user, weight_pbuf), err );
    if (weight_user_to_internal_bwdd) {
        CHECK_ERR( dnnConversionExecute_F32(weight_user_to_internal_bwdd, (void*)weight_pbuf, convBwdImageRes[dnnResourceFilter]), err );
    }

    CHECK_ERR( init_conversion(&gz_user_to_internal_bwdd, &convBwdImageRes[dnnResourceDiffDst],
               gz_layout_internal_bwdd, gz_layout_user, gz_pbuf), err );
    if (gz_user_to_internal_bwdd) {
        CHECK_ERR( dnnConversionExecute_F32(gz_user_to_internal_bwdd, (void*)gz_pbuf, convBwdImageRes[dnnResourceDiffDst]) , err );
    }

    // Allocate internal layout buffer for gradImages 
    CHECK_ERR( dnnAllocateBuffer_F32((void**)&convBwdImageRes[dnnResourceDiffSrc], image_layout_internal_bwdd), err );

    // Do the convolution backward w.r.t image 
	CHECK_ERR( dnnExecute_F32(pConvBwdImage, (void**)convBwdImageRes), err );

    // Convert backward image to user layout
    if (!dnnLayoutCompare_F32(image_layout_user, image_layout_internal_bwdd))
    {
        dnnConversionCreate_F32(&image_internal_to_user_bwdd, image_layout_internal_bwdd, image_layout_user);
        CHECK_ERR(dnnConversionExecute_F32(image_internal_to_user_bwdd, convBwdImageRes[dnnResourceDiffSrc], (void*)gradImage_pbuf), err);
    }
    else
    {
        memcpy((void*)gradImage_pbuf, convBwdImageRes[dnnResourceDiffSrc], dnnLayoutGetMemorySize_F32(image_layout_user));
    }
    } /* ~ gradImage ~ */

    { /* ~ gradWeight ~ */
    // Create convolution backward weight primitive
    CHECK_ERR( dnnConvolutionCreateBackwardFilter_F32 (&pConvBwdWeight, NULL,
                dnnAlgorithmConvolutionDirect, DIMENSION, imageSize,
                outputSize, weightSize, convStride, convPadding,
                dnnBorderZeros), err);

    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&image_layout_internal_bwdf, pConvBwdWeight, dnnResourceSrc), err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&weight_layout_internal_bwdf, pConvBwdWeight, dnnResourceDiffFilter), err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&gz_layout_internal_bwdf, pConvBwdWeight, dnnResourceDiffDst) , err );

    // Do the layout conversion for image and gz if needed
    CHECK_ERR( init_conversion(&image_user_to_internal_bwdf, &convBwdWeightRes[dnnResourceSrc],
               image_layout_internal_bwdf, image_layout_user, image_pbuf), err );
    if (image_user_to_internal_bwdf) {
        CHECK_ERR( dnnConversionExecute_F32(image_user_to_internal_bwdf, (void*)image_pbuf, convBwdWeightRes[dnnResourceSrc]), err );
    }
    
    CHECK_ERR( init_conversion(&gz_user_to_internal_bwdf, &convBwdWeightRes[dnnResourceDiffDst],
               gz_layout_internal_bwdf, gz_layout_user, gz_pbuf), err );
    if (gz_user_to_internal_bwdf) {
        CHECK_ERR( dnnConversionExecute_F32(gz_user_to_internal_bwdf, (void*)gz_pbuf, convBwdWeightRes[dnnResourceDiffDst]) , err );
    }

    // Allocate internal layout buffer for gradWeight
    CHECK_ERR( dnnAllocateBuffer_F32((void**)&convBwdWeightRes[dnnResourceDiffFilter], weight_layout_internal_bwdf), err );
    // Do the convolution forward
	CHECK_ERR( dnnExecute_F32(pConvBwdWeight, (void**)convBwdWeightRes), err );

    // Do the layout conversion for gradWeight if needed.
    if (!dnnLayoutCompare_F32(weight_layout_user, weight_layout_internal_bwdf))
    {
        dnnConversionCreate_F32(&weight_internal_to_user_bwdf, weight_layout_internal_bwdf, weight_layout_user);
        CHECK_ERR(dnnConversionExecute_F32(weight_internal_to_user_bwdf, convBwdWeightRes[dnnResourceDiffFilter], (void*)gradWeight_pbuf), err);
    }
    else
    {
        memcpy((void*)gradWeight_pbuf, convBwdWeightRes[dnnResourceDiffFilter], dnnLayoutGetMemorySize_F32(weight_layout_user));
    }
    } /* ~ gradWeight ~ */

    { /* ~ gradBias ~ */
    if (has_bias) {
        // Create convolution backward bias primitive
        CHECK_ERR( dnnConvolutionCreateBackwardBias_F32 (&pConvBwdBias, NULL,
                   dnnAlgorithmConvolutionDirect, DIMENSION, outputSize), err);

        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&bias_layout_internal_bwdb, pConvBwdBias, dnnResourceDiffBias), err );
        CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&gz_layout_internal_bwdb, pConvBwdBias, dnnResourceDiffDst) , err );

        // Do the layout conversion for gz if needed
        CHECK_ERR( init_conversion(&gz_user_to_internal_bwdb, &convBwdBiasRes[dnnResourceDiffDst],
                   gz_layout_internal_bwdb, gz_layout_user, gz_pbuf), err );
        if (gz_user_to_internal_bwdb) {
            CHECK_ERR( dnnConversionExecute_F32(gz_user_to_internal_bwdb, (void*)gz_pbuf, convBwdBiasRes[dnnResourceDiffDst]) , err );
        }

        // Allocate internal layout buffer for bias 
        CHECK_ERR( dnnAllocateBuffer_F32((void**)&convBwdBiasRes[dnnResourceDiffBias], bias_layout_internal_bwdb), err );

        // Do the convolution forward
        CHECK_ERR( dnnExecute_F32(pConvBwdBias, (void**)convBwdBiasRes), err );

        // Do the layout conversion for gradBias if needed.
        CHECK_ERR( init_conversion(&bias_internal_to_user, &gradBias_pbuf, bias_layout_user,
                  bias_layout_internal_bwdb, convBwdBiasRes[dnnResourceDiffBias]) , err );
        if (bias_internal_to_user) {
            CHECK_ERR( dnnConversionExecute_F32(bias_internal_to_user, convBwdBiasRes[dnnResourceDiffBias], (void*)gradBias_pbuf), err );
        }
    }
    } /* ~ gradBias ~ */

bail_out:
    dnnDelete_F32(pConvBwdImage);
    dnnDelete_F32(pConvBwdWeight);
    dnnDelete_F32(pConvBwdBias);

    dnnLayoutDelete_F32(image_layout_user);
    dnnLayoutDelete_F32(weight_layout_user);
    dnnLayoutDelete_F32(bias_layout_user);
    dnnLayoutDelete_F32(gz_layout_user);

    dnnLayoutDelete_F32(image_layout_internal_bwdd);
    dnnLayoutDelete_F32(weight_layout_internal_bwdd);
    dnnLayoutDelete_F32(gz_layout_internal_bwdd);

    dnnLayoutDelete_F32(image_layout_internal_bwdf);
    dnnLayoutDelete_F32(weight_layout_internal_bwdf);
    dnnLayoutDelete_F32(gz_layout_internal_bwdf);

    dnnLayoutDelete_F32(bias_layout_internal_bwdb);
    dnnLayoutDelete_F32(gz_layout_internal_bwdb);

    dnnDelete_F32(weight_user_to_internal_bwdd);
    dnnDelete_F32(gz_user_to_internal_bwdd);
    dnnDelete_F32(image_internal_to_user_bwdd);

    dnnDelete_F32(image_user_to_internal_bwdf);
    dnnDelete_F32(gz_user_to_internal_bwdf);
    dnnDelete_F32(weight_internal_to_user_bwdf);

    dnnDelete_F32(gz_user_to_internal_bwdb);
    dnnDelete_F32(bias_internal_to_user);

    return err;
}



int _get_conv_output_shape(int ndim, size_t* image_shape, size_t* weight_shape,
                           int* stride, int* pad, size_t* output_shape)
{
    assert (ndim == 4);

    output_shape[0] = image_shape[0];
    output_shape[1] = weight_shape[0];
    output_shape[2] = (image_shape[2] + 2 * pad[0] - weight_shape[2]) / stride[0] + 1;
    output_shape[3] = (image_shape[3] + 2 * pad[1] - weight_shape[3]) / stride[1] + 1;
    
    return (0);
}


void fill_buf(float *pbuf, int size, int range)
{
    for (int i = 0; i < size; ++i) {
        pbuf[i] = (float)(rand() % range);
    }
}
