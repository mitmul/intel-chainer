#ifndef _STREAM_FACTORY_
#define _STREAM_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "layer.h"
#include <unordered_map>

// Usage:
// When stream is created, call:
// LayerFactory::getInstance().setRELUFwdLayer(<input pointer>, <layer>)
// then when forward is needed, call
// layer = LayerFactory::getInstance().getRELUFwdLayer(<input pointer>)

class LayerFactory {
private:
    LayerFactory();

public:
    static LayerFactory& getInstance() {
        static LayerFactory instance;
        return instance;
    }

private:
    Layer<float>* getLayer(std::string       key);
    void          setLayer(std::string       key,
                            Layer<float>*     layer);

public:
    // relu stream
    Layer<float>* getRELUFwdLayer(void*           input,
                                   void*           output);
    void          setRELUFwdLayer(void*           input,
                                   void*           output,
                                   Layer<float>*   layer);
    Layer<float>* getRELUBwdLayer(void*           input,
                                   void*           output_diff,
                                   void*           input_diff);
    void          setRELUBwdLayer(void*           input,
                                   void*           output_diff,
                                   void*           input_diff,
                                   Layer<float>*   layer);

    // maxpool stream
    Layer<float>* getMaxPoolLayer(int             x_d1,
                                   int             x_d2,
                                   int             x_d3,
                                   int             x_d4,
                                   int             stride_y,
                                   int             stride_x,
                                   int             ksize_h,
                                   int             ksize_w,
                                   int             pad_l_h,
                                   int             pad_l_w,
                                   int             pad_r_h,
                                   int             pad_r_w);

    void          setMaxPoolLayer(int             x_d1,
                                   int             x_d2,
                                   int             x_d3,
                                   int             x_d4,
                                   int             stride_y,
                                   int             stride_x,
                                   int             ksize_h,
                                   int             ksize_w,
                                   int             pad_l_h,
                                   int             pad_l_w,
                                   int             pad_r_h,
                                   int             pad_r_w,
                                   Layer<float>*   layer);

    // avgpool stream
    Layer<float>* getAvgPoolFwdLayer(void*              input,
                                      void*              output,
                                      int                stride_y,
                                      int                stride_x,
                                      int                ksize_h,
                                      int                ksize_w,
                                      int                pad_l_h,
                                      int                pad_l_w,
                                      int                pad_r_h,
                                      int                pad_r_w);
    void            setAvgPoolFwdLayer(void*              input,
                                      void*              output,
                                      int                stride_y,
                                      int                stride_x,
                                      int                ksize_h,
                                      int                ksize_w,
                                      int                pad_l_h,
                                      int                pad_l_w,
                                      int                pad_r_h,
                                      int                pad_r_w,
                                      Layer<float>*      layer);

    void setAvgPoolBwdLayer(void*              input_diff,
                             void*              output_diff,
                             void*              workspace,
                             int                stride_y,
                             int                stride_x,
                             int                ksize_h,
                             int                ksize_w,
                             int                pad_l_h,
                             int                pad_l_w,
                             int                pad_r_h,
                             int                pad_r_w,
                             Layer<float>*      layer);

    Layer<float>* getAvgPoolBwdLayer(void*              input_diff,
                                      void*              output_diff,
                                      void*              workspace,
                                      int                stride_y,
                                      int                stride_x,
                                      int                ksize_h,
                                      int                ksize_w,
                                      int                pad_l_h,
                                      int                pad_l_w,
                                      int                pad_r_h,
                                      int                pad_r_w);

    // Local Response Normalization stream
    Layer<float>* getLRNFwdLayer(void*              input,
                                  void*              output,
                                  int                local_size,
                                  float              alpha,
                                  float              beta);
    void          setLRNFwdLayer(void*              input,
                                  void*              output,
                                  int                local_size,
                                  float              alpha,
                                  float              beta,
                                  Layer<float>*      layer);
    Layer<float>* getLRNBwdLayer(void*              input_diff,
                                  void*              output_diff,
                                  int                local_size,
                                  float              alpha,
                                  float              beta);
    void          setLRNBwdLayer(void*              input_diff,
                                  void*              output_diff,
                                  int                local_size,
                                  float              alpha,
                                  float              beta,
                                  Layer<float>*      layer);

    // Softmax Cross Entropy stream
    Layer<float>* getSoftmax2DFwdLayer(int               d1,
                                        int               d2,
                                        int               axis);
    void          setSoftmax2DFwdLayer(int               d1,
                                        int               d2,
                                        int               axis,
                                        Layer<float>*     layer);
    Layer<float>* getSoftmax4DFwdLayer(int               d1,
                                        int               d2,
                                        int               d3,
                                        int               d4,
                                        int               axis);
    void          setSoftmax4DFwdLayer(int               d1,
                                        int               d2,
                                        int               d3,
                                        int               d4,
                                        int               axis,
                                        Layer<float>*     layer);

    LayerFactory(LayerFactory const&)  = delete;
    void operator=(LayerFactory const&) = delete;

private:
    //LayerFactory(LayerFactory const&);
    //void operator=(LayerFactory const&);
    std::unordered_map<std::string, Layer<float>*> map;
};

#endif // _STREAM_FACTORY_
