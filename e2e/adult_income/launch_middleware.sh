#!/bin/bash
set -x

shard_idx=$(($JOB_ID - 1))


if [ $DEBUG ];then
excute_path=/workspace/third_party/persia-embedding-sharded-middleware
else
excute_path=persia-embedding-sharded-middleware
echo "add server into python module at release time"
exit
fi

$excute_path --port $MIDDLEWARE_PORT --global-config /workspace/config/middleware_config.yml \
    --embedding-config /workspace/config/embedding_config.yml \
    --replica-index $shard_idx --replica-size $MIDDLEWARE_REPLICA