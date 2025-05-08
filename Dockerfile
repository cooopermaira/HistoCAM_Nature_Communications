#FROM nvidia/cuda:12.6.0-base-ubuntu22.04 AS build
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS build
#FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python-is-python3 \
    python3-pip \
    apache2-dev \
    build-essential \
    cmake \
    cmake-curses-gui \
    curl \
    doxygen \
    gcc \
    git \
    ladspa-sdk \
    less \
    libapr1-dev \
    libaprutil1-dev \
    libasound2-dev \
    libcurl4-openssl-dev  \
    libfreetype6-dev \
    libglu1-mesa-dev \
    libgraphviz-dev \
    libjack-jackd2-dev \
    default-libmysqlclient-dev \
    libpq-dev \
    libpython3-dev \
    libssl-dev \
    libwebkit2gtk-4.0-dev \
    libx11-dev \
    libxcomposite-dev \
    libxcursor-dev \
    libxext-dev \
    libxinerama-dev \
    libxrandr-dev \
    libxrender-dev \
    mesa-common-dev \
    pkg-config \
    postgresql-server-dev-all \
    procps \
    software-properties-common \
    unixodbc-dev \
    vim \
    gcc-11 \
    g++-11 \
    && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 && \
    true

# The TORCH_CUDA_ARCH_LIST affects both how torch and how pathcam are built.
# Building with a full architecture list is MUCH slower (2000s versus 6000s in
# one test:
#     TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0"

# ENV CUDA_VERSION="11.8" \
#     OPENCV_VERSION="4.9.0" \
#     TORCH_VERSION="2.4.1" \
#     TORCH_CUDA_ARCH_LIST="7.5"

ENV CUDA_VERSION="12.6" \
    OPENCV_VERSION="4.11.0" \
    TORCH_VERSION="2.6.0" \
    TORCH_CUDA_ARCH_LIST="8.6;8.9"

#RUN apt-get update && \
#    apt-get install -y --no-install-recommends gpg-agent && \
#    curl -OLJ https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb && \
#    dpkg -i cuda-keyring_1.1-1_all.deb && \
#    apt-get update && \
#    apt-get install -y --no-install-recommends \
#    cuda-cudart-$(echo ${CUDA_VERSION} | sed 's/\./-/') \
#    cuda-cupti-$(echo ${CUDA_VERSION} | sed 's/\./-/') \
#    cuda-libraries-$(echo ${CUDA_VERSION} | sed 's/\./-/') \
#    cuda-libraries-dev-$(echo ${CUDA_VERSION} | sed 's/\./-/') \
#    cuda-nvcc-$(echo ${CUDA_VERSION} | sed 's/\./-/') \
#    cuda-nvml-dev-$(echo ${CUDA_VERSION} | sed 's/\./-/') \
#    cuda-nvrtc-$(echo ${CUDA_VERSION} | sed 's/\./-/') \
#    cuda-nvtx-$(echo ${CUDA_VERSION} | sed 's/\./-/') \
#    && \
#    true

ENV CUDA_PATH=/usr/local/cuda \
    CUDA_HOME=/usr/local/cuda-${CUDA_VERSION} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    PATH=/usr/local/cuda/bin:${PATH} \
    USE_CUDNN=1 \
    CAFFE2_USE_CUDNN=1

# This pulls a LOT of data
RUN cd /opt && \
    pip install pyyaml typing-extensions && \
    git clone --depth=1 --single-branch -b v$TORCH_VERSION --recurse-submodules -j `nproc` https://github.com/pytorch/pytorch && \
    true

# This is VERY slow
RUN cd /opt && \
    cd pytorch && \
    TOTAL_MEMORY=$(free -g | awk '/Mem:/ {print $2}') && \
    JOBS=$((TOTAL_MEMORY * 10 / 34)) && \
    if [ $JOBS -gt `nproc` ]; then JOBS=`nproc`; fi && \
    CMAKE_BUILD_PARALLEL_LEVEL=$JOBS JOBS=$JOBS MAX_JOBS=$JOBS python setup.py develop && \
    true

RUN mkdir -p /opt && \
    cd /opt && \
    git clone -b ${OPENCV_VERSION} https://github.com/opencv/opencv.git && \
    git clone -b ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git && \
    cd opencv && \
    mkdir _build && \
    cd _build && \
    cmake \
    -DBUILD_SHARED_LIBS=ON \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_opencv_python3=ON \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DCUDA_ARCH_BIN=${TORCH_CUDA_ARCH_LIST} \
    -DCUDA_ARCH_PTX="" \
    -DWITH_CUBLAS=ON \
    -DBUILD_CUDA_STUBS=ON \
    -DENABLE_FAST_MATH=ON \
    -DCUDA_FAST_MATH=ON \
    -DOPENCV_DNN_OPENVINO=ON \
    -DBUILD_opencv_cudacodec=OFF \
    .. && \
    cmake --build . -- -j15 && \
    cmake --build . --target install

RUN mkdir -p /opt/pathcam

# We expect this to be the pathcam repo with all submodules checked out
COPY . /opt/pathcam

RUN cd /opt/pathcam && \
    # Modify CMakeLists.txt so that we are allowed to override parameters \
    sed -i -E 's/^([[:space:]]*)(set|SET|Set)\(([[:alnum:]_]+)[[:space:]]+"([^"]+)"\)/\1\2(\3 "\4" CACHE STRING "\3")/; /\/.*"/ s/STRING/PATH/' CMakeLists.txt && \
    mkdir _build && \
    cd _build && \
    cmake \
    -DTorch_DIR=/opt/pytorch/torch/share/cmake/Torch \
    -DPython3_INCLUDE_DIR=/usr/include/python3.11 \
    -DPython3_LIBRARY_DIR=/usr/lib \
    -DPython3_LIB=/usr/lib/x86_64-linux-gnu/libpython3.11.so.1.0 \
    -DCMAKE_CUDA_COMPILER="${CUDA_PATH}/bin/nvcc" \
    -DCUDAToolkit_ROOT_DIR="${CUDA_PATH}" \
    -DCUDA_NVCC_EXECUTABLE="${CUDA_PATH}/bin/nvcc" \
    -DCUDA_INCLUDE_DIRS="${CUDA_PATH}/include" \
    -DCUDA_CUDART_LIBRARY="${CUDA_PATH}/lib64/libcudart.so" \
    .. && \
    cmake --build . -- -j15 && \
    cmake --build . --target install && \
    true

RUN apt-get install -y gdb
ENV LD_LIBRARY_PATH="/opt/pathcam/_build/lib:$LD_LIBRARY_PATH"

WORKDIR /opt/pathcam/_build/bin

# Sample use:
#  docker build --progress=plain --force-rm -t pathcam/pathcam -f Dockerfile .
#  docker run -e DISPLAY=:10.0 -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/data2/pathcam:/media/max/Data:ro -v /tmp:/output --gpus=all --rm -it pathcam/pathcam /opt/pathcam/_build/pathCamApp_artefacts/Debug/PathCam --config-file=/media/max/Data/1AT1/config_1AT1.xml
