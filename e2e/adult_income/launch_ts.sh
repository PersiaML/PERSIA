#!/bin/bash
set -x

torch-model-archiver \
    --model-name adult_income \
    --version 1.0 \
    --serialized-file $INFER_CHECKPOINT_DIR/jit_dense.pt \
    --handler /workspace/serve_handler.py \
    --export-path $INFER_CHECKPOINT_DIR/ -f

torchserve --start --ncs --model-store $INFER_CHECKPOINT_DIR \
    --models adult_income.mar \
    --ts-config /workspace/config/ts_config.properties &

mkdir /cache/proto/ && python -m grpc_tools.protoc \
    --proto_path=/workspace/proto/ \
    --python_out=/cache/proto/ \
    --grpc_python_out=/cache/proto/ \
    /workspace/proto/inference.proto

sleep 10s

python /workspace/serve_client.py && torchserve --stop
