#!/bin/bash
set -x

torch-model-archiver \
    --model-name adult_income \
    --version 1.0 \
    --serialized-file $PERSIA_CKPT_DIR/jit_dense.pt \
    --handler /workspace/serve_handler.py \
    --export-path $PERSIA_CKPT_DIR/ -f

torchserve --start --ncs --model-store $PERSIA_CKPT_DIR \
    --models adult_income.mar \
    --ts-config /workspace/config/ts_config.properties &

mkdir -p /cache/proto/ && python -m grpc_tools.protoc \
    --proto_path=/workspace/proto/ \
    --python_out=/cache/proto/ \
    --grpc_python_out=/cache/proto/ \
    /proto/inference.proto

sleep 10s

python /workspace/serve_client.py && torchserve --stop
