#include <cstddef>
#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "common.h"
#include "cpu_info.h"

using namespace mkldnn;

engine cpu_engine(engine::cpu, 0);
bool enableMkldnn = true;

int global_init()
{
    google::InitGoogleLogging("mkldnnpy");
 //   google::SetCommandLineOption("minloglevel", "0"); // GLOG_minloglevel
 //   google::SetCommandLineOption("logtostderr", "1"); // GLOG_logtostderr

    LOG(INFO) << "Global Init";

    if (enabled()) {
    /*
     * 1. Set OpenMP thread num as core num
     * 2. Bind OpenMP thread to core
     */
        OpenMpManager::bindOpenMpThreads();
        OpenMpManager::printVerboseInformation();
    }
    return 0;
}

bool enabled()
{
    return enableMkldnn;
}

void setMkldnnEnable(bool isEnabled)
{
    enableMkldnn = isEnabled;
}