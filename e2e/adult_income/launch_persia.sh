#!/bin/bash
set -x

export NODE_RANK=$(($JOB_ID - 1))
export WORLD_SIZE=$(($GPU_NUM * $TRAINER_REPLICA))
export CUDA_VISIBLE_DEVICES=$NODE_RANK

export PERSIA_NATS_IP="nats://data_compose:4222"

/opt/conda/bin/python3 -m torch.distributed.launch --nproc_per_node=$GPU_NUM \
    --nnodes=$TRAINER_REPLICA --node_rank=$NODE_RANK $1