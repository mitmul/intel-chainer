#ifndef _STREAM_FACTORY_
#define _STREAM_FACTORY_
#include <mkldnn.hpp>
#include <string>
#include <map>

class StreamFactory {
public:
    static StreamFactory& getInstance() {
        static StreamFactory instance;
        return instance;
    }
    mkldnn::stream* getStream(std::string       key);
    void            setStream(std::string       key,
                              mkldnn::stream*   stream);

    StreamFactory(StreamFactory const&)  = delete;
    void operator=(StreamFactory const&) = delete;

private:
    StreamFactory();
    //StreamFactory(StreamFactory const&);
    //void operator=(StreamFactory const&);
    std::map<std::string, mkldnn::stream*> map;
};

#endif // _STREAM_FACTORY_
