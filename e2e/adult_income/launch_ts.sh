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
    --ts-config config/ts_config.properties &

python -m grpc_tools.protoc \
    --proto_path=/workspace/proto/ \
    --python_out=/workspace/proto/ \
    --grpc_python_out=/workspace/proto/ \
    /workspace/proto/inference.proto

sleep 10s

python serve_client.py && torchserve --stop
