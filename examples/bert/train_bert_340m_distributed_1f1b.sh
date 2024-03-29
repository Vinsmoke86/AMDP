#!/bin/bash

# Runs the "340M" parameter model (Bert - Large)

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

TENSORBOARD_LOGS_PATH='~/Megatron-LM-main/logs'
VOCAB_FILE='/data/bert-large-uncased-vocab.txt'
DATA_PATH='/data/wikipedia.segmented.nltk.txt'

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

BERT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 1024 
    --num-attention-heads 16 
    --seq-length 512 
    --max-position-embeddings 512 
)

TRAINING_ARGS=(
    --micro-batch-size 4 
    --global-batch-size 16 
    --train-iters 1000000 
    --weight-decay 1e-2 
    --clip-grad 1.0 
    --fp16
    --lr 0.0001
    --lr-decay-iters 990000 
    --lr-decay-style linear 
    --min-lr 1.0e-5 
    --weight-decay 1e-2 
    --lr-warmup-fraction .01 
    --clip-grad 1.0 
    --use-mcore-models
)

MODEL_PARALLEL_ARGS=( 
	--pipeline-model-parallel-size 4
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_bert.py \
    ${BERT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
