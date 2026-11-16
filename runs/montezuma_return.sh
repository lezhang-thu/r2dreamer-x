#!/bin/bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
RUN_DIR=${RUN_DIR:-logdir/$(date +%Y%m%d_%H%M%S)_r2dreamer_return_montezuma_${SEED}}

CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
    env=atari_return \
    model=size100M \
    model.rep_loss=r2dreamer \
    model.compile=True \
    device=cuda:0 \
    buffer.storage_device=cpu \
    buffer.max_size=1000000 \
    return_exploration.enabled=True \
    seed="$SEED" \
    logdir="$RUN_DIR" \
    "$@"
