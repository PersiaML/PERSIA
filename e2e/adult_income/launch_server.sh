#!/bin/bash

set -x

shard_idx=$(($JOB_ID - 1))

if [ $DEBUG ];then
excute_path=/workspace/third_party/persia-embedding-sharded-server
else
excute_path=persia-embedding-sharded-server
echo "add server into python module at release time"
exit
fi

$excute_path --port $SERVER_PORT --shard-idx $shard_idx \
    --num-shards $SERVER_REPLICA --global-config /workspace/config/middleware_config.yml

