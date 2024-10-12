#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

VOCAB_FILE='/data/gpt2-openwebtext-data/gpt2-vocab.json'
MERGE_FILE='/data/gpt2-openwebtext-data/gpt2-merges.txt'
DATA_PATH='/data/gpt2-openwebtext-data/my-gpt2_text_document'

GPT_ARGS="
    --num-layers 48 \
    --hidden-size 1600 \
    --num-attention-heads 25 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 2 \
    --global-batch-size 2 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH
