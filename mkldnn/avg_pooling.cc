#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "avg_pooling.h"
#include "utils.h"

using namespace mkldnn;

template class AvgPooling<float>;


// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s