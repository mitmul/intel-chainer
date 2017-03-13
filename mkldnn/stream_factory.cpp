#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "stream_factory.h"

// helper functions to convert layer unique data to a string
static std::string pointer_to_string(void* ptr)
{
    std::ostringstream os;
    os << std::hex << static_cast<void*>(ptr);
    return os.str();
}
// end of helper functions

mkldnn::stream* StreamFactory::getStream(std::string key)
{
    auto stream_iter = map.find(key);
    if (stream_iter == map.end()) {
        return NULL;
    } else {
        return stream_iter->second;
    }
}

void StreamFactory::setStream(std::string key, mkldnn::stream* stream)
{
    auto stream_iter = map.find(key);
    if (stream_iter == map.end()) {
        map[key]=stream;
    } else {
        throw new std::invalid_argument("cannot set same key to a new stream");
    }
}

mkldnn::stream* StreamFactory::getRELUStream(void* input)
{
    std::string key = "relu_";

    key += pointer_to_string(input);
    return getStream(key);
}

void StreamFactory::setRELUStream(void* input, mkldnn::stream* stream)
{
    std::string key = "relu_";

    key += pointer_to_string(input);
    setStream(key, stream);
}
