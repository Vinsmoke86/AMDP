#!/bin/bash
date='1029'
# ./examples/pretrain_gpt_distributed_inter.sh > logs/$date/8gpu_interpp_gpt2.log 2>&1
./examples/pretrain_gpt_distributed_fd.sh > logs/$date/8gpu_fd_gpt2_1.log 2>&1
# ./examples/pretrain_gpt_distributed_fd_async.sh > logs/$date/8gpu_fd_async_gpt2_1.log 2>&1
# ./examples/pretrain_gpt_distributed_async.sh > logs/$date/8gpu_pipedream_gpt2.log 2>&1
# ./examples/pretrain_gpt_distributed_1f1b.sh > logs/$date/8gpu_1f1b_gpt2.log 2>&1
# ./examples/pretrain_gpt_distributed.sh > logs/$date/8gpu_dp_gpt2.log 2>&1
# ./examples/pretrain_gpt_distributed_bd_async.sh > logs/$date/4gpu_bd_async_gpt2.log 2>&1
# ./examples/pretrain_gpt_distributed_bd.sh > logs/$date/4gpu_bd_gpt2.log 2>&1
