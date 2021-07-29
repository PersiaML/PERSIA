FROM persiaml/persia-ci:latest

ARG GITHUB_TOKEN

RUN git clone https://github.com/PersiaML/PersiaML && \
    git clone https://${GITHUB_TOKEN}@github.com/PersiaML/PersiaML-server

RUN cd PersiaML-server && \
    cargo build --release --package persia-embedding-sharded-server && \
    mv target/release/persia-embedding-sharded-middleware /bin && \
    mv target/release/persia-embedding-sharded-server /bin/ && \
    rm -rf /target/release

RUN cd ../PersiaML && pip3 install .