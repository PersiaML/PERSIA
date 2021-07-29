#!/bin/bash
set -x

export REPLICA_INDEX=$(($JOB_ID - 1))

/opt/conda/bin/python3 $1