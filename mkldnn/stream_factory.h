#ifndef _STREAM_FACTORY_
#define _STREAM_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include "layer.h"
#include <unordered_map>

// Usage:
// When stream is created, call:
// StreamFactory::getInstance().setRELUFwdStream(<input pointer>, <layer>)
// then when forward is needed, call
// layer = StreamFactory::getInstance().getRELUFwdStream(<input pointer>)

class StreamFactory {
private:
    StreamFactory();

public:
    static StreamFactory& getInstance() {
        static StreamFactory instance;
        return instance;
    }

private:
    Layer<float>* getStream(std::string       key);
    void          setStream(std::string       key,
                            Layer<float>*     layer);

public:
    // relu stream
    Layer<float>* getRELUFwdStream(void*           input,
                                   void*           output);
    void          setRELUFwdStream(void*           input,
                                   void*           output,
                                   Layer<float>*   layer);
    Layer<float>* getRELUBwdStream(void*           input,
                                   void*           output_diff,
                                   void*           input_diff);
    void          setRELUBwdStream(void*           input,
                                   void*           output_diff,
                                   void*           input_diff,
                                   Layer<float>*   layer);

    // maxpool stream
    Layer<float>* getMaxPoolStream(int             x_d1,
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

    void          setMaxPoolStream(int             x_d1,
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
    Layer<float>* getAvgPoolFwdStream(void*              input,
                                      void*              output,
                                      int                stride_y,
                                      int                stride_x,
                                      int                ksize_h,
                                      int                ksize_w,
                                      int                pad_l_h,
                                      int                pad_l_w,
                                      int                pad_r_h,
                                      int                pad_r_w);
    void            setAvgPoolFwdStream(void*              input,
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

    void setAvgPoolBwdStream(void*              input_diff,
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

    Layer<float>* getAvgPoolBwdStream(void*              input_diff,
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
    Layer<float>* getLRNFwdStream(void*              input,
                                  void*              output,
                                  int                local_size,
                                  float              alpha,
                                  float              beta);
    void          setLRNFwdStream(void*              input,
                                  void*              output,
                                  int                local_size,
                                  float              alpha,
                                  float              beta,
                                  Layer<float>*      layer);
    Layer<float>* getLRNBwdStream(void*              input_diff,
                                  void*              output_diff,
                                  int                local_size,
                                  float              alpha,
                                  float              beta);
    void          setLRNBwdStream(void*              input_diff,
                                  void*              output_diff,
                                  int                local_size,
                                  float              alpha,
                                  float              beta,
                                  Layer<float>*      layer);

    // Softmax Cross Entropy stream
    Layer<float>* getSoftmax2DFwdStream(int               d1,
                                        int               d2,
                                        int               axis);
    void          setSoftmax2DFwdStream(int               d1,
                                        int               d2,
                                        int               axis,
                                        Layer<float>*     layer);
    Layer<float>* getSoftmax4DFwdStream(int               d1,
                                        int               d2,
                                        int               d3,
                                        int               d4,
                                        int               axis);
    void          setSoftmax4DFwdStream(int               d1,
                                        int               d2,
                                        int               d3,
                                        int               d4,
                                        int               axis,
                                        Layer<float>*     layer);

    StreamFactory(StreamFactory const&)  = delete;
    void operator=(StreamFactory const&) = delete;

private:
    //StreamFactory(StreamFactory const&);
    //void operator=(StreamFactory const&);
    std::unordered_map<std::string, Layer<float>*> map;
};

#endif // _STREAM_FACTORY_
