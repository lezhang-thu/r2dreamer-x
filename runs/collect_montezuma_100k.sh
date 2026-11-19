#!/bin/bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
RUN_DIR=${RUN_DIR:-logdir/montezuma_100k_${SEED}}

# The trainer counts repeated frames. Four per decision gives 100k interactions.
CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
    env=atari100k env.task=atari_montezuma_revenge \
    env.env_num=1 env.action_repeat=4 env.steps=400000 \
    model=size100M model.rep_loss=r2dreamer \
    return_exploration.enabled=False save_episodes=True \
    buffer.max_size=100000 buffer.storage_device=cpu \
    seed="$SEED" logdir="$RUN_DIR" "$@"
