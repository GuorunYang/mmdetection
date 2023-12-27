#!/bin/bash

#############################################################################
# Copyright 2023 Autra.Tech. All Rights Reserved.
#############################################################################

CONFIG_PATH=$1
MODEL_DIR=$2
EPOCH_NUM=$3
LOG_DIR=$4

for((i=1; i<=$EPOCH_NUM; i++));  
do
python tools/test.py \
    CONFIG_PATH \
    $MODEL_DIR/epoch_$i.pth \
    2>&1 | tee work_dirs/eval/epoch_$i.log
pid=$!
wait $pid
done
