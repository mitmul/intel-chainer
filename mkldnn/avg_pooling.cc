#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "avg_pooling.h"
#include "utils.h"

using namespace mkldnn;

template class AvgPooling<float>;
