#!/bin/bash
set -x

export TARGET_DIR=/workspace
export CUR_DIR=`pwd`
export COMPOSE_REPLICA=1
export TRAINER_REPLICA=1
export MIDDLEWARE_REPLICA=1
export SERVER_REPLICA=1

echo "current launch mode is " $DEBUG

if [ -z $DATA_DIR ];then
    echo -e "\033[0;31mset DATA_DIR for absolute dataset path"
    exit -1
else:
echo "current mount data dir is:" $DATA_DIR
fi

if [ $DEBUG ];then
    if [ -z $PERSIA_DIR ];then
        echo -e "\\033[0;31mset PERSIA_DIR for PersiaML absolute path for DEBUG mode"
        exit -1
    fi
docker stack deploy -c docker-compose-debug.yml $1
else
docker stack deploy -c docker-compose.yml $1
fi