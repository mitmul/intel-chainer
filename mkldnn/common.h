#ifndef _COMMON_H_
#define _COMMON_H_

#include <mkldnn.hpp>

#define PAGE_SIZE 4096
int global_init();
bool enabled();
void setMkldnnEnable(bool isEnabled);
void enableGoogleLogging();
extern unsigned char dummy[PAGE_SIZE];
#endif // _COMMON_H_
