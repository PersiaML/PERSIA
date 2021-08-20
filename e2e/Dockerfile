FROM persiaml/persia-ci:latest

ARG GITHUB_TOKEN
ARG USE_CUDA

RUN git clone https://github.com/PersiaML/PersiaML && \
    git clone https://${GITHUB_TOKEN}@github.com/PersiaML/PersiaML-server

RUN cd PersiaML-server && \
    /opt/conda/bin/pip3 install colorama && python3 setup.py install && \
    cargo build --release --package persia-embedding-sharded-server && \
    mv target/release/persia-embedding-sharded-middleware /bin && \
    mv target/release/persia-embedding-sharded-server /bin/ && \
    rm -rf /target/release

RUN cd ../PersiaML && python3 setup.py install