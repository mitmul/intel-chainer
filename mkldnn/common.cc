#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "common.h"

using namespace mkldnn;

engine cpu_engine(engine::cpu, 0);

int global_init()
{
    google::InitGoogleLogging("mkldnnpy");
    // google::SetCommandLineOption("minloglevel", "0"); // GLOG_minloglevel
    // google::SetCommandLineOption("logtostderr", "1"); // GLOG_logtostderr

    LOG(INFO) << "Global Init";

    return 0;
}

bool enabled()
{
    return true;
}

