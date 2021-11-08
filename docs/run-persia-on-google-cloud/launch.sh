#!/bin/bash
set -x
export REPLICA_INDEX=${HOSTNAME##*-}

export PERSIA_MODEL_CONFIG=/workspace/persia-exp/criteo_config.yml
export PERSIA_EMBEDDING_CONFIG=/workspace/persia-exp/criteo_embedding_config.yml
export PERSIA_GLOBAL_CONFIG=/workspace/persia-exp/global_config.yml
export LOG_DIR=/workspace/logs/

export NCCL_DEBUG=INFO

export PERSIA_PORT=23333

mkdir -p $LOG_DIR

export LOG_LEVEL=info
export RUST_BACKTRACE-full

export PERSIA_NATS_IP=nats://persia-nats-service:4222
export PERSIA_METRICS_GATEWAY_ADDR=metrics-gateway:9091

if [ $1 == "trainer" ];then

    export NODE_RANK=${REPLICA_INDEX}
    export WORLD_SIZE=$(($NPROC_PER_NODE * $REPLICA_SIZE))

    /opt/conda/bin/python3 -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
        --nnodes=$REPLICA_SIZE --node_rank=$NODE_RANK /workspace/persia-exp/train.py

fi

if [ $1 == "datacompose" ];then
    /opt/conda/bin/python3 /workspace/persia-exp/data_compose.py
fi

if [ $1 == "middleware" ];then
    /workspace/persia-exp/runtime/$CPU_TYPE/persia-middleware-server ---port $PERSIA_PORT --global-config $PERSIA_GLOBAL_CONFIG \
        --embedding-config $PERSIA_EMBEDDING_CONFIG --replica-index $REPLICA_INDEX --replica-size $REPLICA_SIZE
fi

if [ $1 == "embserver" ];then
    /workspace/persia-exp/runtime/$CPU_TYPE/persia-embedding-server ---port $PERSIA_PORT --global-config $PERSIA_GLOBAL_CONFIG \
        --embedding-config $PERSIA_EMBEDDING_CONFIG --replica-index $REPLICA_INDEX --replica-size $REPLICA_SIZE
fi


exit 0
