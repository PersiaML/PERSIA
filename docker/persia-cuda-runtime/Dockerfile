FROM persiaml/persia-ci:latest

ARG USE_CUDA

WORKDIR /workspace

COPY . .

RUN  pip3 install . -v && rm -rf /workspace