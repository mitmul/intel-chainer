/*  Example of wrapping sin function from math.h using SWIG. */

%module sin_module
%{
    /* the resulting C file should be built as a python extension */
    #define SWIG_FILE_WITH_INIT
    /*  Includes the header in the wrapper code */
    #include "sin_module.h"
%}
/*  Parse the header file to generate wrappers */
%include "sin_module.h"
