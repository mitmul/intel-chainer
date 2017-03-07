// Example:
int _conv_forward( float image_pbuf[], 
                  float weight_pbuf[],
                  size_t image_shape[],  
                  size_t weight_shape[],
                  int stride[], 
                  int pad[],
                  float *z_pbuf);

int _conv_backward(float* image_pbuf,
                   float* weight_pbuf,
                   float* gz_pbuf,
                   size_t* image_shape,
                   size_t* weight_shape,
                   int* stride,
                   int* pad,
                   float* gradImage_pbuf,
                   float* gradWeight_pbuf,
                   float* gradBias_pbuf);

int _get_conv_output_shape(int dim,
                           size_t* image_shape,
                           size_t* weight_shape,
                           int* stride,
                           int* pad,
                           size_t* output_shape);

#define CHECK_ERR(f, err) do { \
    (err) = (f); \
    if ((err) != E_SUCCESS) { \
        printf("[%s:%d] err (%d)\n", __FILE__, __LINE__, err); \
    } \
} while(0)

#define MKL_PRINT_PTR(var) do { \
    printf("[%s:%d], %s: %p\n", __FUNCTION__, __LINE__, #var, var); \
} while(0) 

#define DIMENSION (4)
#define NUM_LAYERS (5)
