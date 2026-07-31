#!/bin/bash
# Post-train the released N0-TWAM checkpoint on your task pool.
# 1. Prepare the pool + norm stats (see docs/POST_TRAINING.md).
# 2. Edit the EDIT-ME block in n0_twam/configs/twam_posttrain_cfg.py.
# 3. NGPU=8 bash run_posttrain.sh            (extra args go to train.py)
set -eu

NGPU=${NGPU:-8}
PORT=${PORT:-29620}

export PYTHONPATH=$PWD:$PWD/n0_twam
export TOKENIZERS_PARALLELISM=false

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  torchrun --nproc_per_node="$NGPU" --master_port "$PORT" --tee 3 \
  -m n0_twam.train --config-name posttrain "$@"
