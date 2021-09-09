FROM persiaml/persia-ci:latest

ARG USE_CUDA=1

WORKDIR /workspace

COPY . .

RUN  pip3 install . -v && rm -rf /workspace