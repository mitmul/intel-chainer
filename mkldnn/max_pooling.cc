#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "max_pooling.h"
#include "utils.h"

using namespace mkldnn;

template class MaxPooling<float>;
