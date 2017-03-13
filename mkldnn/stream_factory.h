#ifndef _STREAM_FACTORY_
#define _STREAM_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include <unordered_map>

class StreamFactory {
public:
    static StreamFactory& getInstance() {
        static StreamFactory instance;
        return instance;
    }
    mkldnn::stream* getStream(std::string       key);
    void            setStream(std::string       key,
                             mkldnn::stream*   stream);

    // relu stream
    mkldnn::stream* getRELUStream(void*           input);
    void            setRELUStream(void*           input,
                                  mkldnn::stream* stream);

#if 0
    // maxpool stream
    mkldnn::stream* getMaxPoolStream(void*              input,
                                     int                ksize_h,
                                     int                ksize_w,
                                     int                stride_y,
                                     int                stride_x,
                                     int                pad_h,
                                     int                pad_w);
    void            setMaxPoolStream(void*              input,
                                     int                ksize_h,
                                     int                ksize_w,
                                     int                stride_y,
                                     int                stride_x,
                                     int                pad_h,
                                     int                pad_w,
                                     mkldnn::stream*    stream);

    // avgpool stream
    mkldnn::stream* getAvgPoolStream(void*              input,
                                     int                ksize_h,
                                     int                ksize_w,
                                     int                stride_y,
                                     int                stride_x,
                                     int                pad_h,
                                     int                pad_w);
    void            setAvgPoolStream(void*              input,
                                     int                ksize_h,
                                     int                ksize_w,
                                     int                stride_y,
                                     int                stride_x,
                                     int                pad_h,
                                     int                pad_w,
                                     mkldnn::stream*    stream);

    // Local Response Normalization stream
    mkldnn::stream* getLRNStream(void*              input,
                                 int                n,
                                 float              k,
                                 float              alpha,
                                 float              beta);
    void            setLRNStream(void*              input,
                                 int                n,
                                 float              k,
                                 float              alpha,
                                 float              beta,
                                 mkldnn::stream*    stream);

    // Softmax Cross Entropy stream
    mkldnn::stream* getSoftmaxCrossEntropyStream(
                            void*               input,
                            void*               groundTruth,
                            bool                normalize,
                            void*               class_weight);
    void            setSoftmaxCrossEntropyStream(
                            void*               input,
                            void*               groundTruth,
                            bool                normalize,
                            void*               class_weight,
                            mkldnn::stream*     stream);
#endif

    StreamFactory(StreamFactory const&)  = delete;
    void operator=(StreamFactory const&) = delete;

private:
    StreamFactory();
    //StreamFactory(StreamFactory const&);
    //void operator=(StreamFactory const&);
    std::unordered_map<std::string, mkldnn::stream*> map;
};

#endif // _STREAM_FACTORY_
