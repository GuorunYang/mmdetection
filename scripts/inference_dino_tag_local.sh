#!/bin/bash

#############################################################################
# Copyright 2023 Autra.Tech. All Rights Reserved.
#############################################################################

RECORD_PATH=$1
OUTPUT_DIR=$2

python3 scripts/grounding_dino_mine.py \
    $RECORD_PATH \
    weights/swinb_autra_trainval_finetune_epoch_19.pth \
    --tag-score-thr 0.8 \
    --batch-size 20 \
    --save-local \
    --no-save-vis \
    --out-dir $2 \
    # --no-save-pred
    # --is_dev
