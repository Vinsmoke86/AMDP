#!/bin/bash
./examples/pretrain_gpt_distributed_inter.sh > logs/1011/8gpu_interpp_gpt2.log 2>&1
./examples/pretrain_gpt_distributed_fd.sh > logs/1011/8gpu_fd_gpt2.log 2>&1
./examples/pretrain_gpt_distributed_fd_async.sh > logs/1011/8gpu_fd_async_gpt2.log 2>&1
./examples/pretrain_gpt_distributed_async.sh > logs/1011/8gpu_pipedream_gpt2.log 2>&1
./examples/pretrain_gpt_distributed_1f1b.sh > logs/1011/8gpu_1f1b_gpt2.log 2>&1
./examples/pretrain_gpt_distributed.sh > logs/1011/8gpu_dp_gpt2.log 2>&1
# ./examples/pretrain_gpt_distributed_bd_async.sh > logs/async_bdpp_gpt2_0724.log 2>&1
# ./examples/pretrain_gpt_distributed_async.sh > logs/pipedream_gpt2_0724.log 2>&1
