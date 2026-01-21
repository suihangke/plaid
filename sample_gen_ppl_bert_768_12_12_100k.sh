#!/bin/bash

# Script to generate samples using sample_gen_ppl.py
# Checkpoint: checkpoints_bert_dim128_768_12_12_100k

# Set CUDA device (adjust as needed)
export CUDA_VISIBLE_DEVICES=1

# Checkpoint directory
CHECKPOINT_DIR="checkpoints_bert_dim128_768_12_12_100k"

# Model parameters from config.json
DIM=768
N_BLOCKS=12
N_HEADS=12
EMBED_DIM=128
SEQ_LEN=128

# Generation parameters
N_SAMPLES=1024
BATCH_SIZE=32  # Batch size for generation to avoid OOM
SAMPLING_TIMESTEPS=4096
SCORE_TEMP=0.9
DDIM_SAMPLER=false

# Tokenizer (auto-detected from checkpoint, but can override)
TOKENIZER_NAME="google-bert/bert-base-uncased"

echo "=========================================="
echo "Sample Generation Configuration:"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Model: dim=$DIM, n_blocks=$N_BLOCKS, n_heads=$N_HEADS, embed_dim=$EMBED_DIM"
echo "Sequence length: $SEQ_LEN"
echo "Number of samples: $N_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Sampling timesteps: $SAMPLING_TIMESTEPS"
echo "Score temperature: $SCORE_TEMP"
echo "DDIM sampler: $DDIM_SAMPLER"
echo "Tokenizer: $TOKENIZER_NAME"
echo "=========================================="
echo ""

# Run sample generation
python sample_gen_ppl.py \
  --checkpoint_dir=$CHECKPOINT_DIR \
  --weights_path=$CHECKPOINT_DIR \
  --dim=$DIM \
  --n_blocks=$N_BLOCKS \
  --n_heads=$N_HEADS \
  --embed_dim=$EMBED_DIM \
  --seq_len=$SEQ_LEN \
  --n_samples=$N_SAMPLES \
  --batch_size=$BATCH_SIZE \
  --sampling_timesteps=$SAMPLING_TIMESTEPS \
  --tokenizer_name=$TOKENIZER_NAME

echo ""
echo "Sample generation completed!"
echo "Samples saved to: $CHECKPOINT_DIR/samples_token_ids.pt and $CHECKPOINT_DIR/samples.txt"
