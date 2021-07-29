#!/bin/bash
set -x

shard_idx=$(($JOB_ID - 1))

persia-embedding-sharded-middleware --port $MIDDLEWARE_PORT --global-config /workspace/config/middleware_config.yml \
    --embedding-config /workspace/config/embedding_config.yml \
    --replica-index $shard_idx --replica-size $MIDDLEWARE_REPLICA