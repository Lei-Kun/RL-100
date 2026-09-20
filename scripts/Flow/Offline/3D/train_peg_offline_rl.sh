#!/usr/bin/env bash
set -euo pipefail

# Train one two-stage offline-RL checkpoint for the 1024-point peg dataset.
# Usage: bash scripts/Flow/Offline/3D/train_peg_offline_rl.sh [seed] [num_gpus]

SEED=${1:-42}
NUM_GPUS=${2:-1}

BC_EPOCHS=1200 \
BATCH_SIZE="${BATCH_SIZE:-256}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}" \
LR_VALUES="1e-6" \
ROLLOUT_VALUES="3" \
CLIP_STD_MAX_VALUES="0.1" \
CHUNK_ADV_CLIP_VALUES="null" \
CHUNK_LOSS_MODE_COMBOS="scalar:scalar_iql scalar" \
bash scripts/Flow/Offline/3D/train_policy_chunk_two_stage_flow.sh \
    rl100 peg peg_1024 "${SEED}" "${NUM_GPUS}"
