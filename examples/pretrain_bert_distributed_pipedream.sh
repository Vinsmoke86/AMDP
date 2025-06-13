#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# CHECKPOINT_PATH=<Specify path>
VOCAB_FILE='/data/enwiki/bert-large-cased-vocab.txt'
DATA_PATH='/data/enwiki/my-bert_text_sentence'

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

BERT_ARGS="
    --pipeline-model-parallel-size 4 \
    --enable-asynchronous-pipeline \
    --weight-stash \
    --num-layers 32 \
    --hidden-size 1600 \
    --num-attention-heads 32 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 4 \
    --lr 0.0001 \
    --train-iters 10000 \
    --lr-decay-iters 990000 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_bert.py \
    $BERT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl 
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH
