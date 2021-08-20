#!/bin/bash

set -x

shard_idx=$(($JOB_ID - 1))

export PERSIA_NATS_IP="nats://data_compose:4222"

persia-embedding-sharded-server --port $SERVER_PORT --shard-idx $shard_idx \
    --num-shards $SERVER_REPLICA --global-config /workspace/config/middleware_config.yml

