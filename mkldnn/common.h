#ifndef _COMMON_H_
#define _COMMON_H_

#include <mkldnn.hpp>

int global_init();
bool enabled();
void setMkldnnEnable(bool isEnabled);
#endif // _COMMON_H_
