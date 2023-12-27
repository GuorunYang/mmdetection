#!/bin/bash

#############################################################################
# Copyright 2023 Autra.Tech. All Rights Reserved.
#############################################################################

CONFIG_PATH=$1
MODEL_DIR=$2
EPOCH_NUM=$3
LOG_DIR=$4

mkdir -p $LOG_DIR

for i in $(seq 1 $EPOCH_NUM)
do   
python tools/test.py \
    $CONFIG_PATH \
    $MODEL_DIR/epoch_$i.pth \
    2>&1 | tee $LOG_DIR/epoch_$i.log
pid=$!
wait $pid
done
