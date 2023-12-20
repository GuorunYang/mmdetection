#!/bin/bash

#############################################################################
# Copyright 2023 Autra.Tech. All Rights Reserved.
#############################################################################

RECORD_PATH=$1

python3 scripts/grounding_dino_mine.py \
    $RECORD_PATH \
    weights/swinb_autra_trainval_finetune_epoch_19.pth \
    --tag-score-thr 0.8 \
    --batch-size 20 \
    --no-save-vis \
    --no-save-pred
    # --is_dev
