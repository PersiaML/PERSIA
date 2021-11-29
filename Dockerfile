ARG DEVICE=cuda
ARG BASE_IMAGE=nvidia/cuda:11.2.0-devel-ubuntu20.04

FROM ${BASE_IMAGE} AS base
ARG PYTHON_VERSION=3.8
ARG PYTORCH_VERSION=1.8
ARG MAGMA_CUDA_VERSION=magma-cuda110
ARG DEVICE

RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl \
    build-essential \
    ca-certificates \
    git \
    libgfortran-8-dev \
    vim \
    zsh \
    wget \
    ssh \
    iputils-ping \
    procps \
    net-tools \
    apt-utils \
    rlwrap \
    ethtool \
    telnet \
    openjdk-11-jdk \
    openssh-server 

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} numpy scipy mkl mkl-include ninja cython typing && \
    /opt/conda/bin/conda install -y -c conda-forge mpi4py && \
    ln -s /usr/share/pyshared /opt/conda/lib/python${PYTHON_VERSION}/site-packages && \
    if [ "${DEVICE}" = "cuda" ]; then \
    /opt/conda/bin/conda install -y -c pytorch -c conda-forge ${MAGMA_CUDA_VERSION} pytorch=${PYTORCH_VERSION} torchvision; \
    /opt/conda/bin/pip3 install bagua-cuda113 --no-cache-dir; \
    else \ 
    /opt/conda/bin/conda install -y -c pytorch -c conda-forge pytorch=${PYTORCH_VERSION} torchvision cpuonly; \
    /opt/conda/bin/pip3 install scikit-learn --no-cache-dir; \
    fi && \
    /opt/conda/bin/conda install torchserve torch-model-archiver torch-workflow-archiver -c pytorch -y; \
    /opt/conda/bin/conda clean -yapf;

RUN mkdir -p /opt/hadoop/; \
    cd /opt/hadoop/; \
    wget https://dlcdn.apache.org/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz; \
    tar -zxvf hadoop-3.3.1.tar.gz; \
    rm hadoop-3.3.1.tar.gz; 

RUN /opt/conda/bin/pip install --no-cache-dir \ 
    remote-pdb \
    pytest \
    tqdm \
    pandas \
    tensorboard \
    ipython \
    captum \
    grpcio \
    protobuf \
    grpcio-tools && \
    apt-get purge --auto-remove && \
    apt-get clean

ENV PATH=/opt/conda/bin:/opt/hadoop/hadoop-3.3.1/bin/:$PATH
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
ENV LIBRARY_PATH="/usr/local/lib64:/usr/local/lib:/usr/lib"
ENV LD_LIBRARY_PATH="/opt/conda/lib/python${PYTHON_VERSION}/site-packages/torch/lib/:/opt/conda/lib/"

# alias for cpu builder image
FROM base AS cpu-builder-base 
# alias for gpu builder image
FROM base AS cuda-builder-base
ARG DEVICE

ENV USE_CUDA=1
ENV LIBRARY_PATH="${LIBRARY_PATH}:/usr/local/cuda/lib64/stubs/"

FROM ${DEVICE}-builder-base AS builder

ENV RUSTUP_HOME=/rust
ENV CARGO_HOME=/cargo
ENV PATH=/cargo/bin:/rust/bin:/opt/conda/bin:$PATH

RUN curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain stable -y --profile default --no-modify-path 

FROM builder AS persia-builder

WORKDIR /workspace
COPY . /workspace
RUN cd /workspace && pip3 install colorama setuptools setuptools-rust setuptools_scm \
    && python setup.py bdist_wheel --dist-dir=/root/dist && rm -rf /workspace

# Build bagua distributed training framework manully
# RUN if [ "${DEVICE}" = "cuda" ]; then \
#     rm -rf /etc/apt/sources.list.d; \
#     apt-get update; \
#     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends zlib1g-dev libhwloc-dev; \ 
#     git clone https://github.com/BaguaSys/bagua.git; \
#     cd bagua; \
#     pip3 install cmake setuptools-rust colorama tqdm wheel --no-cache-dir; \
#     git submodule update --init --recursive; \
#     python setup.py bdist_wheel --dist-dir=/root/dist; \
#     cd ..; \
#     rm -rf bagua; \
#     /opt/conda/bin/conda clean -yapf; \
#     fi 

ARG DEVICE
FROM base AS runtime

# Install the persia-runtime and bagua (Optional for cpu-runtime)
COPY --from=persia-builder /root/dist .
RUN pip3 install *.whl && rm -rf *.whl


# Copy examples
RUN mkdir -p /home/PERSIA/examples
COPY examples /home/PERSIA/examples
RUN cd /home/PERSIA/examples/src/adult-income/data/ && ./prepare_data.sh