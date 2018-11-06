## C++ Interface for classification

Reference : [demo](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification/predict-cpp)

### Build ```libmxnet.so``` from source 
* For all platforms, the first step is to build MXNet from source, with `USE_CPP_PACKAGE = 1`. Details are available on [MXNet website](https://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=CPU).
* Build cpp inference demo with mxnet cpp-package support.
* mkdir lib in project and make install libmxnet.so there

**We will go through with cpu versions, gpu versions of mxnet are similar but requires `USE_CUDA=1` and `USE_CUDNN=1 (optional)`. See [MXNet website](https://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=CPU) if interested.**

#### Linux (cpu + openblas)
We use Ubuntu as example in Linux section.

##### 1. Install build tools and git
```bash
sudo apt-get update
sudo apt-get install -y build-essential git
# install openblas
sudo apt-get install -y libopenblas-dev
# install opencv
sudo apt-get install -y libopencv-dev
# install cmake
sudo apt-get install -y cmake
```
 ##### 2. Download MXNet source and build shared library
```bash
cd ~
git clone --recursive https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CPP_PACKAGE=1
```
##### 3. (optional) Add MXNet to `LD_LIBRARY_PATH`
```bash
export LD_LIBRARY_PATH=~/incubator-mxnet/lib
```
##### 4. Build demo application
```bash
cd ~
git clone https://github.com/dmlc/gluon-cv.git
cd gluon-cv/scripts/deployment/cpp-inference
mkdir build
cd build
cmake .. -DMXNET_ROOT=~/incubator-mxnet
make -j $(nproc)
```

##### 5. (optional) Copy app to install directory
```bash
make install
# gluoncv-detect and libmxnet.so will be available at ~/gluon-cv/scripts/deployment/cpp-inference/install/
# you may want to add libmxnet.so to LD_LIBRARY_PATH
```
