#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "stream_factory.h"

mkldnn::stream* StreamFactory::getStream(std::string key)
{
    std::map<std::string, mkldnn:stream*>::const_iterator
            stream_iter = map.find(key);
    if (stream_iter = map.end()) {
        return NULL;
    } else {
        return stream_iter.second;
    }
}

void StreamFactory::setStream(std::string key, mkldnn::stream* stream)
{
    std::map<std::string, mkldnn:stream*>::const_iterator
            stream_iter = map.find(key);
    if (stream_iter = map.end()) {
        map.insert(key, stream);
    } else {
        throw new std::invalid_argument("cannot set same key to a new stream);
    }
}
