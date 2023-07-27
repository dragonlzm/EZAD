#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
WORK_DIR=$3
DATA_DIR=$4

PORT=${PORT:-29501}

# rm -rf ./data
# ln -sf ${DATA_DIR} ./data

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --work-dir=${WORK_DIR} ${@:5}
