#!/bin/bash

# Script to resume training from ~30k checkpoint to 100k steps (dim=768, embed_dim=768)
# Usage: bash train_resume_dim768_30k_to_100k.sh [--resume-wandb] [--wandb-id=WANDB_ID] [--wandb-name=NAME]

# Default settings
RESUME_WANDB=false
WANDB_ID=""
WANDB_NAME="bert_dim768_768_12_12_30k_to_100k"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume-wandb)
            RESUME_WANDB=true
            shift
            ;;
        --wandb-id=*)
            WANDB_ID="${1#*=}"
            shift
            ;;
        --wandb-name=*)
            WANDB_NAME="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash train_resume_670k_to_1M.sh [--resume-wandb] [--wandb-id=WANDB_ID] [--wandb-name=NAME]"
            exit 1
            ;;
    esac
done

# Set CUDA device (adjust as needed)
export CUDA_VISIBLE_DEVICES=0

# Base command
BASE_CMD="python train_lm1b_bert.py \
  --resume_from_checkpoint=/home/hangkes2/plaid/checkpoints_bert_dim768_768_12_12_100k \
  --steps=100000 \
  --checkpoint_dir=checkpoints_bert_dim768_768_12_12_100k \
  --dim=768 \
  --embed_dim=768 \
  --n_blocks=12 \
  --n_heads=12 \
  --use_wandb=True \
  --wandb_project=plaid-lm1b \
  --wandb_name=${WANDB_NAME}"

# Add wandb resume option if specified
if [ "$RESUME_WANDB" = true ]; then
    if [ -z "$WANDB_ID" ]; then
        echo "Warning: --resume-wandb specified but no --wandb-id provided."
        echo "Will create a new wandb run. If you want to resume, please provide --wandb-id=YOUR_RUN_ID"
        echo ""
    else
        BASE_CMD="${BASE_CMD} --wandb_id=${WANDB_ID} --wandb_resume=must"
        echo "Will resume wandb run with ID: $WANDB_ID"
        echo ""
    fi
else
    echo "Will create a new wandb run with name: $WANDB_NAME"
    echo ""
fi

# Print configuration
echo "=========================================="
echo "Training Configuration:"
echo "=========================================="
echo "Resume from checkpoint: /home/hangkes2/plaid/checkpoints_bert_dim768_768_12_12_100k (currently step ~30k)"
echo "Target steps: 100,000"
echo "Checkpoint dir (will write here): /home/hangkes2/plaid/checkpoints_bert_dim768_768_12_12_100k"
echo "Model: dim=768, embed_dim=768, n_blocks=12, n_heads=12"
echo "Wandb project: plaid-lm1b"
echo "Wandb name: $WANDB_NAME"
if [ "$RESUME_WANDB" = true ] && [ -n "$WANDB_ID" ]; then
    echo "Wandb resume: Yes (ID: $WANDB_ID)"
else
    echo "Wandb resume: No (new run)"
fi
echo "=========================================="
echo ""

# Execute the command
echo "Starting training..."
echo ""
eval $BASE_CMD
