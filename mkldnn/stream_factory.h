#ifndef _STREAM_FACTORY_
#define _STREAM_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include <unordered_map>

// Usage:
// When stream is created, call:
// StreamFactory::getInstance().setRELUFwdStream(<input pointer>, <stream>)
// then when forward is needed, call
// stream = StreamFactory::getInstance().getRELUFwdStream(<input pointer>)

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
    mkldnn::stream* getRELUFwdStream(void*           input,
                                     void*           output);
    void            setRELUFwdStream(void*           input,
                                     void*           output,
                                     mkldnn::stream* stream);
    mkldnn::stream* getRELUBwdStream(void*           input,
                                     void*           output_diff,
                                     void*           input_diff);
    void            setRELUBwdStream(void*           input,
                                     void*           output_diff,
                                     void*           input_diff,
                                     mkldnn::stream* stream);

    // maxpool stream
    mkldnn::stream* getMaxPoolFwdStream(void*              input,
                                        void*              output,
                                        int                stride_y,
                                        int                stride_x,
                                        int                ksize_h,
                                        int                ksize_w,
                                        int                pad_l_h,
                                        int                pad_l_w,
                                        int                pad_r_h,
                                        int                pad_r_w);
    void            setMaxPoolFwdStream(void*              input,
                                        void*              output,
                                        int                stride_y,
                                        int                stride_x,
                                        int                ksize_h,
                                        int                ksize_w,
                                        int                pad_l_h,
                                        int                pad_l_w,
                                        int                pad_r_h,
                                        int                pad_r_w,
                                        mkldnn::stream*    stream);
    mkldnn::stream* getMaxPoolBwdStream(void*              input_diff,
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
    void            setMaxPoolBwdStream(void*              input_diff,
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
                                        mkldnn::stream*    stream);

    // avgpool stream
    mkldnn::stream* getAvgPoolFwdStream(void*              input,
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
                                        mkldnn::stream*    stream);
    mkldnn::stream* getAvgPoolBwdStream(void*              input_diff,
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
    void            setAvgPoolBwdStream(void*              input,
                                        void*              output,
                                        void*              workspace,
                                        int                stride_y,
                                        int                stride_x,
                                        int                ksize_h,
                                        int                ksize_w,
                                        int                pad_l_h,
                                        int                pad_l_w,
                                        int                pad_r_h,
                                        int                pad_r_w,
                                        mkldnn::stream*    stream);

    // Local Response Normalization stream
    mkldnn::stream* getLRNFwdStream(void*              input,
                                    void*              output,
                                    int                local_size,
                                    float              alpha,
                                    float              beta);
    void            setLRNFwdStream(void*              input,
                                    void*              output,
                                    int                local_size,
                                    float              alpha,
                                    float              beta,
                                    mkldnn::stream*    stream);
    mkldnn::stream* getLRNBwdStream(void*              input_diff,
                                    void*              output_diff,
                                    int                local_size,
                                    float              alpha,
                                    float              beta);
    void            setLRNBwdStream(void*              input_diff,
                                    void*              output_diff,
                                    int                local_size,
                                    float              alpha,
                                    float              beta,
                                    mkldnn::stream*    stream);

    // Softmax Cross Entropy stream
    mkldnn::stream* getSoftmaxFwdStream(
                            void*               input,
                            void*               output,
                            int                 axis);
    void            setSoftmaxFwdStream(
                            void*               input,
                            void*               output,
                            int                 axis,
                            mkldnn::stream*     stream);

    StreamFactory(StreamFactory const&)  = delete;
    void operator=(StreamFactory const&) = delete;

private:
    StreamFactory();
    //StreamFactory(StreamFactory const&);
    //void operator=(StreamFactory const&);
    std::unordered_map<std::string, mkldnn::stream*> map;
};

#endif // _STREAM_FACTORY_
