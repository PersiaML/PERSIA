ARG BASE_IMAGE=nvidia/cuda:11.2.0-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION=3.8
ARG PYTORCH_VERSION=1.7
ARG MAGMA_CUDA_VERSION=magma-cuda110
ARG DEPLOY=false
ARG CPU_ONLY
ARG WITHOUT_BAGUA
ARG CI

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
    telnet \
    libhwloc-dev \
    software-properties-common \
    libssl-dev && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \ 
    apt remove --purge cmake -y && apt-get update && apt-get install -y --no-install-recommends cmake

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
    /opt/conda/bin/conda install -y -c conda-forge mpi4py && \
    if [ -z "${CPU_ONLY}" ]; then \
    echo "install pytorch in gpu mode and install the magma cuda requirement."; \
    /opt/conda/bin/conda install -y -c pytorch ${MAGMA_CUDA_VERSION};  \ 
    else \
    echo "install pytorch in cpu mode."; \
    fi

ENV RUSTUP_HOME=/rust
ENV CARGO_HOME=/cargo
ENV PATH=/cargo/bin:/rust/bin:/opt/conda/bin:$PATH

RUN curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain none -y --profile default --no-modify-path && \
    chown -R 1000:1000 /rust /cargo  && \
    rustup install nightly-2021-06-01 

RUN /opt/conda/bin/conda install -y -c pytorch -c conda-forge pytorch=${PYTORCH_VERSION} torchvision ${CPU_ONLY} && \
    /opt/conda/bin/conda clean -yapf && \
    /opt/conda/bin/pip install --no-cache-dir \ 
    pyyaml \
    remote-pdb \
    colorlog \
    pytest \
    tqdm \
    pandas \
    scikit-learn \
    ipython && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean  && \
    ln -s /usr/share/pyshared /opt/conda/lib/python${PYTHON_VERSION}/site-packages

RUN if [ -z "${WITHOUT_BAGUA}" ]; then \
    git clone https://github.com/BaguaSys/bagua.git; \
    /opt/conda/bin/pip install setuptools-rust colorama tqdm wheel; \
    cd bagua; \
    /opt/conda/bin/pip install . -v; \
    git clone https://github.com/BaguaSys/bagua-core.git;  \
    cd bagua-core;  \
    git submodule update --init --recursive; \
    pip3 install .; \
    fi

# install deploy requirements
RUN if [ "${DEPLOY}" = true ]; then \
    echo "install deploy image requirement, openjdk and torch serve"; \
    /opt/conda/bin/conda install torchserve torch-model-archiver torch-workflow-archiver -c pytorch -y; \
    /opt/conda/bin/conda clean -yapf; \
    /opt/conda/bin/pip install --no-cache-dir \ 
        captum \
        grpcio \
        protobuf \
        grpcio-tools; \
    apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-11-jdk; \
    apt-get purge --auto-remove; \
    apt-get clean; \
    fi

ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs/:/usr/local/lib64:/usr/local/lib:/usr/lib"
ENV LD_LIBRARY_PATH="/opt/conda/lib/${PYTHON_VERSION}/site-packages/torch/lib/:/opt/conda/lib/"

WORKDIR /workspace
COPY . /workspace
RUN cd /workspace && \
    if [ -z "${CPU_ONLY}" ]; then \
        echo "install persiaml in gpu mode"; \
        export USE_CUDA=1; \
    else \
        echo "install persiaml in cpu mode"; \
    fi && \
    pip3 install . -v --no-cache-dir && \
    rm -rf /workspace && \ 
    if [ -z "${CI}" ]; then \
        rm -rf /cargo /rust; \
    fi
