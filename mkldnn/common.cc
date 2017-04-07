#include <cstddef>
#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "common.h"
#include "cpu_info.h"

using namespace mkldnn;

engine cpu_engine(engine::cpu, 0);
bool enableMkldnn = true;
unsigned char dummy[PAGE_SIZE] __attribute__((aligned(PAGE_SIZE)));
#define DUMMY_VAL 0xcc

int global_init()
{
    google::SetStderrLogging(1);
    google::InitGoogleLogging("mkldnnpy");

    LOG(INFO) << "Global Init";

    if (enabled()) {
    /*
     * 1. Set OpenMP thread num as core num
     * 2. Bind OpenMP thread to core
     */
        OpenMpManager::bindOpenMpThreads();
        OpenMpManager::printVerboseInformation();
    }

    for (int i=0; i<PAGE_SIZE; i++) {
        dummy[i]=DUMMY_VAL;
    }

    return 0;
}

bool enabled()
{
    return enableMkldnn;
}

bool checkDummy()
{
    for (int i=0; i<PAGE_SIZE; i++) {
        if (dummy[i] != DUMMY_VAL) {
            return false;
        }
    }
    return true;
}

void setMkldnnEnable(bool isEnabled)
{
    enableMkldnn = isEnabled;
}

void enableGoogleLogging()
{
   google::SetStderrLogging(0);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s