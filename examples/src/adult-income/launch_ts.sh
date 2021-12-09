#!/bin/bash
set -x

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

torch-model-archiver \
    --model-name adult_income \
    --version 1.0 \
    --serialized-file $PERSIA_CKPT_DIR/jit_dense.pt \
    --handler $SCRIPTPATH/serve_handler.py \
    --export-path $PERSIA_CKPT_DIR/ -f

torchserve --start --ncs --model-store $PERSIA_CKPT_DIR \
    --models adult_income.mar \
    --ts-config $SCRIPTPATH/config/ts_config.properties &

mkdir -p /cache/proto/ && python -m grpc_tools.protoc \
    --proto_path=/proto/ \
    --python_out=/cache/proto/ \
    --grpc_python_out=/cache/proto/ \
    /proto/inference.proto

sleep 10s

python $SCRIPTPATH/serve_client.py && torchserve --stop
