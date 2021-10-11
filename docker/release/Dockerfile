ARG BASE_IMAGE=nvidia/cuda:11.2.0-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION=3.8
ARG PYTORCH_VERSION=1.7
ARG MAGMA_CUDA_VERSION=magma-cuda110
ARG DEPLOY=false
ARG CPU_ONLY
ARG CI
ARG HDFS

RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl \
    build-essential \
    cmake \
    ca-certificates \
    git \
    libgfortran-8-dev \
    vim \
    zsh \
    ssh \
    iputils-ping \
    procps \
    net-tools \
    apt-utils \
    rlwrap \
    ethtool \
    telnet

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

RUN /opt/conda/bin/conda install -y -c pytorch -c conda-forge pytorch=${PYTORCH_VERSION} torchvision ${CPU_ONLY} && \
    /opt/conda/bin/conda clean -yapf && \
    /opt/conda/bin/pip install --no-cache-dir \ 
    pyyaml \
    remote-pdb \
    retrying \
    colorlog \
    pytest \
    tqdm \
    pandas \
    tensorboard \ 
    scikit-learn \
    ipython && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean && \
    update-alternatives --install /usr/bin/python3 python3 /opt/conda/bin/python${PYTHON_VERSION} 10 

# install hdfs requirement
RUN if [ "${HDFS}" = true ]; then \
    apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-11-jdk \
    openssh-server \
    wget; \
    mkdir -p /opt/hadoop/; \
    cd /opt/hadoop/; \
    wget https://dlcdn.apache.org/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz; \
    tar -zxvf hadoop-3.3.1.tar.gz; \
    rm hadoop-3.3.1.tar.gz; \
    cd -; \
    fi

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
ENV PATH=/opt/hadoop/hadoop-3.3.1/bin/:$PATH

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

ENV RUSTUP_HOME=/rust
ENV CARGO_HOME=/cargo
ENV PATH=/cargo/bin:/rust/bin:/opt/conda/bin:$PATH
ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs/:/usr/local/lib64:/usr/local/lib:/usr/lib"
ENV LD_LIBRARY_PATH="/opt/conda/lib/${PYTHON_VERSION}/site-packages/torch/lib/:/opt/conda/lib/"

WORKDIR /workspace
COPY . /workspace
RUN curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain none -y --profile default --no-modify-path && \
    chown -R 1000:1000 /rust /cargo  && \
    rustup install nightly-2021-06-01 && \
    cd /workspace && \
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
