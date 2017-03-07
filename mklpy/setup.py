import os
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
# from Cython.Build import cythonize


mkl_dir = os.environ.get("MKLROOT", None)
if not mkl_dir:
    raise RuntimeError("MKLROOT is not set, "
                       "please make sure MKL is properly installed!")

ext_modules = [Extension(
    name="mklpy",
    sources=[
        "mkl.pyx",
        "mkl_conv.cpp",
        ],
    include_dirs=[numpy.get_include(), "%s/include" % (mkl_dir)],
    language="C++",
    libraries=["mklml_intel", "iomp5"],
    library_dirs=["%s/lib" % (mkl_dir)]
    )]

setup(
    name="mklpy",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    )
