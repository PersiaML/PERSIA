FROM persiaml/persia-ci:latest

ARG USE_CUDA

WORKDIR /workspace

RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends wget openssh-server

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/

RUN mkdir -p /opt/hadoop/ && \
    cd /opt/hadoop/ && \
    wget https://dlcdn.apache.org/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz && \
    tar -zxvf hadoop-3.3.1.tar.gz && \
    rm hadoop-3.3.1.tar.gz

ENV PATH=/opt/hadoop/hadoop-3.3.1/bin/:$PATH

COPY . .

RUN  pip3 install . -v && rm -rf /workspace