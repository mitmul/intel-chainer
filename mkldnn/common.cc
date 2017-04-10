#include <cstddef>
#include <glog/logging.h>
#include <iostream>
#include "mkldnn.hpp"
#include "common.h"
#include "cpu_info.h"

using namespace mkldnn;

engine cpu_engine(engine::cpu, 0);
static bool s_enable_mkldnn = true;
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
    return s_enable_mkldnn;
}

bool check_dummy()
{
    for (int i=0; i<PAGE_SIZE; i++) {
        if (dummy[i] != DUMMY_VAL) {
            return false;
        }
    }
    return true;
}

void set_mkldnn_enable(bool is_enabled)
{
    s_enable_mkldnn = is_enabled;
}

void enable_google_logging()
{
   google::SetStderrLogging(0);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
