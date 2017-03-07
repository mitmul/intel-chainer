# Intel Software Optimization for Chainer*
---

This repo is dedicated to improving Chainer performance on CPU, especially in Intel® Xeon® and Intel® Xeon Phi™ processors.

**Installation**

  * Quick Commands
```
# get latest code
git clone https://github.com/intel/chainer.git intel-chainer
cd intel-chainer

# install mklpy
cd ../mklpy
python setup.py build
python setup.py install --user  
 
# install Chainer
cd ..
python setup.py build
python setup.py install –user
```
---
>\* Other names and trademarks may be claimed as the property of others.

