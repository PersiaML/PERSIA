FROM persiaml/persia-ci:latest

ARG USE_CUDA

WORKDIR /workspace

RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-11-jdk

RUN /opt/conda/bin/conda install -y python=3.8 pip && \
        /opt/conda/bin/conda install torchserve torch-model-archiver torch-workflow-archiver -c pytorch -y && \
        /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/pip install --no-cache-dir \ 
    captum \
    grpcio \
    protobuf \
    grpcio-tools \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean && \
    update-alternatives --install /usr/bin/python3 python3 /opt/conda/bin/python3.8 10 

COPY . .

RUN  pip3 install . -v && rm -rf /workspace