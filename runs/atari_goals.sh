#!/usr/bin/env bash
set -euo pipefail

# Activate the r2dreamer Conda environment before invoking this script.
# 400,000 emulator frames / action_repeat 4 = 100,000 Dreamer decisions.
# Goal-agent interactions are counted separately and never enter that budget.
python train.py \
  model=size100M env=atari100k env.task=atari_hero env.steps=400000 \
  goal_conditioning.enabled=True "$@"
