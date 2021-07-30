#!/bin/bash
set -x

export REPLICA_INDEX=$(($JOB_ID - 1))

export PERSIA_NATS_IP="nats://trainer:4222"

/opt/conda/bin/python3 $1