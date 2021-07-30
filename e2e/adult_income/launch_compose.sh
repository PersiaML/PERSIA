#!/bin/bash
set -x

export REPLICA_INDEX=$(($JOB_ID - 1))

nats-server &

export PERSIA_NATS_IP="nats://0.0.0.0:4222"

/opt/conda/bin/python3 $1