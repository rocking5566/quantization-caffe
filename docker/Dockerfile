FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
LABEL maintainer caffe-maint@googlegroups.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy \
        python-tk \
        octave


RUN apt-get install -y cmake \
    g++ \
    unzip \
    vim \
    libgtk2.0-dev \
    pkg-config \
    libcanberra-gtk-module \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

RUN pip install --upgrade pip
RUN pip install Cython>=0.19.2 \
    numpy==1.16.3 \
    scipy==0.17.0 \
    scikit-image>=0.9.3 \
    matplotlib==2.2.4 \
    ipython>=3.0.0 \
    h5py>=2.2.0 \
    leveldb>=0.191 \
    networkx>=1.8.1 \
    nose>=1.3.0 \
    pandas==0.24.0 \
    python-dateutil==2.1 \
    protobuf>=2.5.0 \
    python-gflags>=2.0 \
    pyyaml>=3.10 \
    Pillow>=2.3.0 \
    six>=1.1.0 \
    lmdb


# ============= OpenCV =============
# OpenCV
# [Warning] - opencv-python does not include Video I/O (Can not open Webcam)
# Build opencv from source

ARG opencv_version=3.4.1

ENV OPENCV_DIR /opt/opencv

RUN mkdir -p $OPENCV_DIR

WORKDIR $OPENCV_DIR

RUN wget https://github.com/Itseez/opencv/archive/${opencv_version}.zip -O opencv-${opencv_version}.zip

RUN unzip opencv-${opencv_version}.zip && \
    cd opencv-${opencv_version} &&\
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_CUDA=OFF \
    -D WITH_EIGEN=OFF \
    -D WITH_OPENCL=OFF \
    -D BUILD_opencv_python3=YES \
  ..

RUN cd opencv-${opencv_version}/build && \
    make -j4 && \
    make install && \
    rm $OPENCV_DIR/opencv-${opencv_version}.zip

RUN pip install pycocotools future onnx==1.6.0 cityscapesScripts
RUN apt-get update && apt install -y fuse libnss3 libgtk-3-0

ENV CAFFE_ROOT=/workspace/caffe-quant
ENV BUILD_ROOT=/workspace/build

ENV PYCAFFE_ROOT $BUILD_ROOT/install/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $PYCAFFE_ROOT/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$PYCAFFE_ROOT/install/lib/" >> /etc/ld.so.conf.d/caffe.conf && ldconfig