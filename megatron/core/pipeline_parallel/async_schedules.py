# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Callable, Iterator, List, Optional, Union

import time

import torch
from torch.autograd.variable import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import get_attr_wrapped_model, get_model_config, get_model_type
from megatron.training.global_vars import (
    get_args,
    get_timers,
    get_num_microbatches)
from megatron.training.utils import calc_params_l2_norm, training_log

# Types
Shape = Union[List[int], torch.Size]
iters = 0

def get_async_traing_func():
    args = get_args()
    assert args.enable_asynchronous_pipeline is True, "Can not use asynchronous method in synchrounous pipeline training"
    if args.enable_bidirectional_pipeline:
        asyn_train_func = async_train_bidirectional_pipeline
    elif args.enable_fourdirectional_pipeline:
        asyn_train_func = async_train_fourdirectional_pipeline
    else:
        asyn_train_func = async_train_pipedream_pipeline
    return asyn_train_func

def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype,)


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format,)

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
):

    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    if parallel_state.is_bidirectional_pipeline_last_stage():
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            output_tensor = loss / num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.tensor(1.0, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.tensor(1.0)
        )
        # Set the loss scale
        MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if (
        parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    if config.deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def check_first_val_step(first_val_step, forward_only, cond):
    if (first_val_step is not None) and forward_only:
        return first_val_step and cond
    else:
        return cond

def async_train_bidirectional_pipeline(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    optimizer,
    opt_param_scheduler,
    iter_num,
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run asynchronous bidirectional schedule (pipeline run in a mirrored stage), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), "bidirectional pipeline parallelism expected mirrored stages"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid mirrord stage"
    assert isinstance(
        data_iterator, list
    ), "bidirectional pipeline parallelism expected each mirrored stage to have a data iterator"

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != seq_length:
        raise RuntimeError(
            "Interleaving is not supported with a different decoder sequence length."
        )

    args = get_args()
    timers = get_timers()

    # Tracking loss.
    total_loss_dict = {}

    report_memory_flag = True

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    # 镜像stage的上一stage的激活与下一stage的梯度，下标0是从上往下，下标1是从下往上
    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    allreduce_works = []
    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)


    model_type = get_model_type(model[0])
    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    itertions = 0
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()
    
    # TODO 计算warmup阶段的mb数量，当mb数量大于stage数的时候需要考虑。现在的做法是全部都直接往流水线中注入，之后可能有优化空间
    num_warmup_microbatches = num_microbatches

    # TODO ckpt相关，目前还不知道是什么意思，先找interleaved抄过来
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1
    
    
    def is_first_microbatch_for_model_chunk(microbatch_id):
        return microbatch_id == get_model_chunk_id(microbatch_id)

    #TODO 先简单的以奇偶区分该mb是从上往下还是从下往上的
    def get_model_chunk_id(microbatch_id):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id = microbatch_id % 2
        return model_chunk_id
    
    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        # return True
        model_id = get_model_chunk_id(microbatch_id)
        return microbatch_id == 2 + model_id
    
    def allreduce_grad(model, async_op=False):
        grads = []
        for param in model.module.parameters():
            grad = param.main_grad
            grads.append(grad.data)
        if grads:
            coalesced = _flatten_dense_tensors(grads)
            work = torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_bidirectional_pipeline_mirror_group(), async_op=async_op
            )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)
            return work
    
    def reduce_grad(model, model_id, async_op=False):
        grads = []
        for param in model.module.parameters():
            grad = param.main_grad
            grads.append(grad.data)
        if grads:
            coalesced = _flatten_dense_tensors(grads)
            if model_id == 0:
                torch.distributed.reduce(
                    coalesced, 
                    dst=torch.distributed.get_process_group_ranks(parallel_state.get_bidirectional_pipeline_mirror_group())[pipeline_parallel_rank // 2], 
                    group=parallel_state.get_bidirectional_pipeline_mirror_group(), 
                    async_op=async_op
                )
            else:
                torch.distributed.reduce(
                    coalesced, 
                    dst=torch.distributed.get_process_group_ranks(parallel_state.get_bidirectional_pipeline_mirror_group())[1-pipeline_parallel_rank // 2], 
                    group=parallel_state.get_bidirectional_pipeline_mirror_group(), 
                    async_op=async_op
                )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

    def allreduce_param(model, async_op=False):
        params = []
        for param in model.module.parameters():
            params.append(param.data)
        if params:
            coalesced = _flatten_dense_tensors(params)
            work = torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_bidirectional_pipeline_mirror_group(), async_op=async_op
            )
            for buf, synced in zip(params, _unflatten_dense_tensors(coalesced, params)):
                synced = synced / pipeline_parallel_size
                buf.copy_(synced)
        return work


    def forward_step_helper(microbatch_id, checkpoint_activations_microbatch=None):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id)
        if model_chunk_id == 0:
            parallel_state.set_bidirectional_pipeline_current_rank(pipeline_parallel_rank)
        else:
            parallel_state.set_bidirectional_pipeline_current_rank(pipeline_parallel_size - 1 - pipeline_parallel_rank)

        # forward step
        if parallel_state.is_bidirectional_pipeline_first_stage():
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]

        output_tensor = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, is_first_microbatch_for_model_chunk(microbatch_id),
            ),
        )
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id, sync_grad = False):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id)
        if model_chunk_id == 0:
            parallel_state.set_bidirectional_pipeline_current_rank(pipeline_parallel_rank)
        else:
            parallel_state.set_bidirectional_pipeline_current_rank(pipeline_parallel_size - 1 - pipeline_parallel_rank)

        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_bidirectional_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        # 梯度同步
        if sync_grad is True:
            if get_args().zero1_bidirectional_pipeline is False:
                allreduce_grad(model[model_chunk_id], False)
            else:
                reduce_grad(model[model_chunk_id],model_chunk_id,False)
            # allreduce_param(model[model_chunk_id], False)
        return input_tensor_grad
    
    
    def update_and_post_process(iteration):
        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        for work in allreduce_works:
            work.wait()
            allreduce_works.pop(0)
        
        if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
            config.finalize_model_grads_func(model, zero=get_args().zero1_bidirectional_pipeline)

        # Update parameters.
        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        timers('optimizer').stop()
        
        group = parallel_state.get_bidirectional_pipeline_mirror_group()
        # 先broadcast前半段，后broadcast后半段
        if get_args().zero1_bidirectional_pipeline is True:
            if pipeline_parallel_size // 2 > pipeline_parallel_rank:
                params = []
                for param in model[0].module.parameters():
                    params.append(param.data)
                if params:
                    coalesced = _flatten_dense_tensors(params)
                    torch.distributed.broadcast(
                        coalesced, 
                        src=torch.distributed.get_process_group_ranks(group)[0], 
                        group=group
                    )
                    for buf, synced in zip(params, _unflatten_dense_tensors(coalesced, params)):
                        buf.copy_(synced)
                params = []
                for param in model[1].module.parameters():
                    params.append(param.data)
                if params:
                    coalesced = _flatten_dense_tensors(params)
                    torch.distributed.broadcast(
                        coalesced, 
                        src=torch.distributed.get_process_group_ranks(group)[1], 
                        group=group
                    )
                    for buf, synced in zip(params, _unflatten_dense_tensors(coalesced, params)):
                        buf.copy_(synced)
            else:
                params = []
                for param in model[1].module.parameters():
                    params.append(param.data)
                if params:
                    coalesced = _flatten_dense_tensors(params)
                    torch.distributed.broadcast(
                        coalesced, 
                        src=torch.distributed.get_process_group_ranks(group)[0], 
                        group=group
                    )
                    for buf, synced in zip(params, _unflatten_dense_tensors(coalesced, params)):
                        buf.copy_(synced)
                params = []
                for param in model[0].module.parameters():
                    params.append(param.data)
                if params:
                    coalesced = _flatten_dense_tensors(params)
                    torch.distributed.broadcast(
                        coalesced, 
                        src=torch.distributed.get_process_group_ranks(group)[1], 
                        group=group
                    )
                    for buf, synced in zip(params, _unflatten_dense_tensors(coalesced, params)):
                        buf.copy_(synced)

        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        loss_reduced = {}
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            
            for key in forward_data_store[0]:
                losses_reduced_for_key = [x[key] for x in forward_data_store]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        # if iteration % args.log_interval == 0:
        #     track_e2e_metrics()
        nonlocal report_memory_flag
        batch_size = parallel_state.get_data_parallel_world_size() * \
                        args.micro_batch_size * \
                        num_microbatches
        args.consumed_train_samples += batch_size
        report_memory_flag = training_log(loss_reduced, total_loss_dict,
                                            optimizer.param_groups[0]['lr'],
                                            iteration + 1, loss_scale,
                                            report_memory_flag, skipped_iter,
                                            grad_norm, params_norm, num_zeros_in_grad)
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
            optimizer.zero_grad()
        forward_data_store.clear()

    if pipeline_parallel_rank == 0:
        fwd_order = [0, 2, 1, 3]
        bwd_order = [1, 3, 0, 2]
    elif pipeline_parallel_rank == 1:
        fwd_order = [0, 1, 2, 3]
        bwd_order = [1, 0, 3, 2]
    elif pipeline_parallel_rank == 2:
        fwd_order = [1, 0, 3, 2]
        bwd_order = [0, 1, 2, 3]
    elif pipeline_parallel_rank == 3:
        fwd_order = [1, 3, 0, 2]
        bwd_order = [0, 2, 1, 3]
    def fwd_ptr_inc(fwd_ptr):
        fwd_ptr += 1
        fwd_ptr = fwd_ptr % 4
        return fwd_ptr

    def bwd_ptr_inc(bwd_ptr):
        bwd_ptr += 1
        bwd_ptr = bwd_ptr % 4
        return bwd_ptr
    
    fwd_ptr = 0
    bwd_ptr = 0

    for iter in range(iter_num):
        for i in range(num_microbatches // 4):
            if iter==0:
                if pipeline_parallel_rank == 0:
                    output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    p2p_communication.send_forward(output_tensor, config)
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                    output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_forward_recv_backward(
                            output_tensor,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                elif pipeline_parallel_rank == 3:
                    output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    p2p_communication.send_backward(output_tensor, config)
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                    output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_backward_recv_forward(
                            output_tensor,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                elif pipeline_parallel_rank == 1:
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(p2p_communication.recv_forward(tensor_shape,config))
                    output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_forward_recv_backward(
                            output_tensor,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                    output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_backward_recv_forward(
                            output_tensor,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                elif pipeline_parallel_rank == 2:
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(p2p_communication.recv_backward(tensor_shape,config))
                    output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_backward_recv_forward(
                            output_tensor,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                    output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_forward_recv_backward(
                            output_tensor,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            else:
                if pipeline_parallel_rank == 0:
                    output_tensor1=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    output_tensor2=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                        p2p_communication.send_forward_recv_backward(
                            output_tensor1,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor1, config.deallocate_pipeline_outputs)
                    input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr])
                    bwd_ptr=bwd_ptr_inc(bwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_forward_recv_backward(
                            output_tensor2,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor2, config.deallocate_pipeline_outputs)
                elif pipeline_parallel_rank == 3:
                    output_tensor1=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    output_tensor2=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                        p2p_communication.send_backward_recv_forward(
                            output_tensor1,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor1, config.deallocate_pipeline_outputs)
                    input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr])
                    bwd_ptr=bwd_ptr_inc(bwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_backward_recv_forward(
                            output_tensor2,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor2, config.deallocate_pipeline_outputs)
                elif pipeline_parallel_rank==1:
                    input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr])
                    bwd_ptr=bwd_ptr_inc(bwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_backward_recv_forward(
                            input_tensor_grad,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_forward_recv_backward(
                            output_tensor,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                    output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_backward_recv_forward(
                            output_tensor,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                elif pipeline_parallel_rank==2:
                    input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr])
                    bwd_ptr=bwd_ptr_inc(bwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_forward_recv_backward(
                            input_tensor_grad,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_backward_recv_forward(
                            output_tensor,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                    output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr=fwd_ptr_inc(fwd_ptr)
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                        p2p_communication.send_forward_recv_backward(
                            output_tensor,
                            True,
                            tensor_shape,
                            config
                        )
                    )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            if pipeline_parallel_rank == 0:
                output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                fwd_ptr=fwd_ptr_inc(fwd_ptr)
                input_tensor_grad = backward_step_helper(bwd_order[bwd_ptr])
                bwd_ptr=bwd_ptr_inc(bwd_ptr)
                input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                    p2p_communication.send_forward_recv_backward(
                        input_tensor_grad,
                        True,
                        tensor_shape,
                        config
                    )
                )
                output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                fwd_ptr=fwd_ptr_inc(fwd_ptr)
                if iter != 0 and i==0:
                    update_and_post_process(iter)
                input_tensor_grad = backward_step_helper(bwd_order[bwd_ptr], i==(num_microbatches//4-1))
                bwd_ptr=bwd_ptr_inc(bwd_ptr)
                output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                    p2p_communication.send_forward_recv_backward(
                        input_tensor_grad,
                        True,
                        tensor_shape,
                        config
                    )
                )
                input_tensor_grad = backward_step_helper(bwd_order[bwd_ptr])
                bwd_ptr=bwd_ptr_inc(bwd_ptr)
            elif pipeline_parallel_rank == 3:
                output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                fwd_ptr=fwd_ptr_inc(fwd_ptr)
                input_tensor_grad = backward_step_helper(bwd_order[bwd_ptr])
                bwd_ptr=bwd_ptr_inc(bwd_ptr)
                input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                    p2p_communication.send_backward_recv_forward(
                        input_tensor_grad,
                        True,
                        tensor_shape,
                        config
                    )
                )
                output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                fwd_ptr=fwd_ptr_inc(fwd_ptr)
                if iter != 0 and i==0:
                    update_and_post_process(iter)
                input_tensor_grad = backward_step_helper(bwd_order[bwd_ptr], i==(num_microbatches//4-1))
                bwd_ptr=bwd_ptr_inc(bwd_ptr)
                output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                    p2p_communication.send_backward_recv_forward(
                        input_tensor_grad,
                        True,
                        tensor_shape,
                        config
                    )
                )
                input_tensor_grad = backward_step_helper(bwd_order[bwd_ptr])
                bwd_ptr=bwd_ptr_inc(bwd_ptr)
            elif pipeline_parallel_rank==1:
                output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                fwd_ptr=fwd_ptr_inc(fwd_ptr)
                input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                    p2p_communication.send_forward_recv_backward(
                        output_tensor,
                        True,
                        tensor_shape,
                        config
                    )
                )
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                fwd_ptr=fwd_ptr_inc(fwd_ptr)
                if iter != 0 and i==0:
                    update_and_post_process(iter)
                output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                    p2p_communication.send_backward_recv_forward(
                        output_tensor,
                        True,
                        tensor_shape,
                        config
                    )
                )
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr])
                bwd_ptr=bwd_ptr_inc(bwd_ptr)
                output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                    p2p_communication.send_forward_recv_backward(
                        input_tensor_grad,
                        True,
                        tensor_shape,
                        config
                    )
                )
                input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr])
                bwd_ptr=bwd_ptr_inc(bwd_ptr)
                output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                    p2p_communication.send_backward_recv_forward(
                        input_tensor_grad,
                        True,
                        tensor_shape,
                        config
                    )
                )
                input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr],i==(num_microbatches//4 - 1))
                bwd_ptr=bwd_ptr_inc(bwd_ptr)
                output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                    p2p_communication.send_forward_recv_backward(
                        input_tensor_grad,
                        True,
                        tensor_shape,
                        config
                    )
                )
            elif pipeline_parallel_rank==2:
                output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                fwd_ptr=fwd_ptr_inc(fwd_ptr)
                input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                    p2p_communication.send_backward_recv_forward(
                        output_tensor,
                        True,
                        tensor_shape,
                        config
                    )
                )
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                output_tensor=forward_step_helper(fwd_order[fwd_ptr])
                fwd_ptr=fwd_ptr_inc(fwd_ptr)
                if iter != 0 and i==0:
                    update_and_post_process(iter)
                output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                    p2p_communication.send_forward_recv_backward(
                        output_tensor,
                        True,
                        tensor_shape,
                        config
                    )
                )
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr])
                bwd_ptr=bwd_ptr_inc(bwd_ptr)
                output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                    p2p_communication.send_backward_recv_forward(
                        input_tensor_grad,
                        True,
                        tensor_shape,
                        config
                    )
                )
                input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr])
                bwd_ptr=bwd_ptr_inc(bwd_ptr)
                output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                    p2p_communication.send_forward_recv_backward(
                        input_tensor_grad,
                        True,
                        tensor_shape,
                        config
                    )
                )
                input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr],i==(num_microbatches//4 - 1))
                bwd_ptr=bwd_ptr_inc(bwd_ptr)
                output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                    p2p_communication.send_backward_recv_forward(
                        input_tensor_grad,
                        True,
                        tensor_shape,
                        config
                    )
                )
    if pipeline_parallel_rank==0:
        output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(p2p_communication.recv_backward(tensor_shape, config))
        input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr], True)
        bwd_ptr=bwd_ptr_inc(bwd_ptr)
    elif pipeline_parallel_rank==3:
        output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(p2p_communication.recv_forward(tensor_shape, config))
        input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr], True)
        bwd_ptr=bwd_ptr_inc(bwd_ptr)
    elif pipeline_parallel_rank==1:
        input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr], True)
        bwd_ptr=bwd_ptr_inc(bwd_ptr)
        p2p_communication.send_backward(input_tensor_grad, config)
    elif pipeline_parallel_rank==2:
        input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr], True)
        bwd_ptr=bwd_ptr_inc(bwd_ptr)
        p2p_communication.send_forward(input_tensor_grad, config)
    update_and_post_process(iter)

def async_train_fourdirectional_pipeline(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    optimizer,
    opt_param_scheduler,
    iter_num,
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run asynchronous bidirectional schedule (pipeline run in a mirrored stage), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), "bidirectional pipeline parallelism expected mirrored stages"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid mirrord stage"
    assert isinstance(
        data_iterator, list
    ), "bidirectional pipeline parallelism expected each mirrored stage to have a data iterator"

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != seq_length:
        raise RuntimeError(
            "Interleaving is not supported with a different decoder sequence length."
        )

    args = get_args()
    timers = get_timers()

    # Tracking loss.
    total_loss_dict = {}

    report_memory_flag = True

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    # 镜像stage的上一stage的激活与下一stage的梯度，下标0是从上往下，下标1是从下往上
    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    allreduce_works = []
    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)


    model_type = get_model_type(model[0])
    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    itertions = 0
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()
    
    # TODO 计算warmup阶段的mb数量，当mb数量大于stage数的时候需要考虑。现在的做法是全部都直接往流水线中注入，之后可能有优化空间
    num_warmup_microbatches = num_microbatches

    # TODO ckpt相关，目前还不知道是什么意思，先找interleaved抄过来
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1
    
    
    def is_first_microbatch_for_model_chunk(microbatch_id):
        return microbatch_id == get_model_chunk_id(microbatch_id)

    # TODO 以余4的结果计算是哪个model
    def get_model_chunk_id(microbatch_id):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id = microbatch_id % 4
        return model_chunk_id
    
    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        # return True
        model_id = get_model_chunk_id(microbatch_id)
        return microbatch_id == 4 + model_id
    
    def allreduce_grad(model, async_op=False):
        grads = []
        for param in model.module.parameters():
            grad = param.main_grad
            grads.append(grad.data)
        if grads:
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_fourdirectional_pipeline_mirror_group(), async_op=async_op
            )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)
    
    def reduce_grad(model, model_id, async_op=False):
        grads = []
        if pipeline_parallel_rank <= 1: 
            model_order = [0, 1, 2, 3]
        elif pipeline_parallel_rank <=3:
            model_order = [1, 0, 3, 2]
        elif pipeline_parallel_rank <= 5:
            model_order = [2, 3, 0, 1]
        else:
            model_order = [3, 2, 1, 0]
        for param in model.module.parameters():
            grad = param.main_grad
            grads.append(grad.data)
        if grads:
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.reduce(
                coalesced, 
                dst=torch.distributed.get_process_group_ranks(parallel_state.get_fourdirectional_pipeline_mirror_group())[model_order[model_id]], 
                group=parallel_state.get_fourdirectional_pipeline_mirror_group(), 
                async_op=async_op
            )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

    def forward_step_helper(microbatch_id, checkpoint_activations_microbatch=None):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id)
        if model_chunk_id == 0:
            parallel_state.set_fourdirectional_pipeline_current_rank(pipeline_parallel_rank)
        elif model_chunk_id == 1:
            parallel_state.set_fourdirectional_pipeline_current_rank((11-pipeline_parallel_rank)%8)
        elif model_chunk_id == 2:
            parallel_state.set_fourdirectional_pipeline_current_rank((4+pipeline_parallel_rank)%8)
        else:
            parallel_state.set_fourdirectional_pipeline_current_rank(pipeline_parallel_size - 1 - pipeline_parallel_rank)

        # forward step
        if parallel_state.is_fourdirectional_pipeline_first_stage():
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]

        output_tensor = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, is_first_microbatch_for_model_chunk(microbatch_id),
            ),
        )
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id, sync_grad = False):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id)
        if model_chunk_id == 0:
            parallel_state.set_fourdirectional_pipeline_current_rank(pipeline_parallel_rank)
        elif model_chunk_id == 1:
            parallel_state.set_fourdirectional_pipeline_current_rank((11-pipeline_parallel_rank)%8)
        elif model_chunk_id == 2:
            parallel_state.set_fourdirectional_pipeline_current_rank((4+pipeline_parallel_rank)%8)
        else:
            parallel_state.set_fourdirectional_pipeline_current_rank(pipeline_parallel_size - 1 - pipeline_parallel_rank)

        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_fourdirectional_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        # 梯度同步
        if sync_grad is True:
            if get_args().zero1_bidirectional_pipeline is False:
                allreduce_grad(model[model_chunk_id], False)
            else:
                reduce_grad(model[model_chunk_id],model_chunk_id,False)
            # allreduce_param(model[model_chunk_id], False)
        return input_tensor_grad
    
    
    def update_and_post_process(iteration):
        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        for work in allreduce_works:
            work.wait()
            allreduce_works.pop(0)
        
        if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
            config.finalize_model_grads_func(model, zero=get_args().zero1_bidirectional_pipeline)

        # Update parameters.
        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        timers('optimizer').stop()
        
        # 先broadcast前半段，后broadcast后半段
        if get_args().zero1_bidirectional_pipeline is True:
            rank = pipeline_parallel_rank
            if rank <= 1: 
                model_order = [0, 1, 2, 3]
            elif rank <=3:
                model_order = [1, 0, 3, 2]
            elif rank <= 5:
                model_order = [2, 3, 0, 1]
            else:
                model_order = [3, 2, 1, 0]
            for i in model_order:
                params = []
                for param in model[i].module.parameters():
                    params.append(param.data)
                if params:
                    coalesced = _flatten_dense_tensors(params)
                    torch.distributed.broadcast(
                        coalesced, src=torch.distributed.get_process_group_ranks(parallel_state.get_fourdirectional_pipeline_mirror_group())[0], group=parallel_state.get_fourdirectional_pipeline_mirror_group()
                    )
                    for buf, synced in zip(params, _unflatten_dense_tensors(coalesced, params)):
                        buf.copy_(synced)

        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        loss_reduced = {}
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            
            for key in forward_data_store[0]:
                losses_reduced_for_key = [x[key] for x in forward_data_store]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        # if iteration % args.log_interval == 0:
        #     track_e2e_metrics()
        nonlocal report_memory_flag
        batch_size = parallel_state.get_data_parallel_world_size() * \
                        args.micro_batch_size * \
                        num_microbatches
        args.consumed_train_samples += batch_size
        report_memory_flag = training_log(loss_reduced, total_loss_dict,
                                            optimizer.param_groups[0]['lr'],
                                            iteration + 1, loss_scale,
                                            report_memory_flag, skipped_iter,
                                            grad_norm, params_norm, num_zeros_in_grad)
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
            optimizer.zero_grad()
        forward_data_store.clear()

    is_embbeding_rank = pipeline_parallel_rank in [0, 3, 4, 7]

    if max_outstanding_backprops is not None:
        checkpoint_activations_microbatch = (
            0 % max_outstanding_backprops
            >= config.num_microbatches_with_partial_activation_checkpoints
        )
    else:
        checkpoint_activations_microbatch = None
    if pipeline_parallel_rank == 0:
        fwd_order = [0, 4, 1, 2, 5, 6, 3, 7]
        bwd_order = [3, 7, 2, 1, 6, 5, 0, 4]
    elif pipeline_parallel_rank == 1:
        fwd_order = [0, 1, 4, 5, 2, 3, 6, 7]
        bwd_order = [3, 2, 7, 6, 1, 0, 5, 4]
    elif pipeline_parallel_rank == 2:
        fwd_order = [1, 0, 5, 4, 3, 2, 7, 6]
        bwd_order = [2, 3 ,6, 7, 0, 1, 4, 5]
    elif pipeline_parallel_rank == 3:
        fwd_order = [1, 5, 0, 3, 4, 7, 2, 6]
        bwd_order = [2, 6, 3, 0, 7, 4, 1, 5]
    elif pipeline_parallel_rank == 4:
        fwd_order = [2, 6, 3, 0, 7, 4, 1, 5]
        bwd_order = [1, 5, 0, 3, 4, 7, 2, 6]
    elif pipeline_parallel_rank == 5:
        fwd_order = [2, 3, 6, 7, 0, 1, 4, 5]
        bwd_order = [1, 0, 5, 4, 3, 2, 7, 6]
    elif pipeline_parallel_rank == 6:
        fwd_order = [3, 2, 7, 6, 1, 0, 5, 4]
        bwd_order = [0, 1, 4, 5, 2, 3, 6, 7]
    elif pipeline_parallel_rank == 7:
        fwd_order = [3, 7, 2, 1, 6, 5, 0, 4]
        bwd_order = [0, 4, 1, 2, 5, 6, 3, 7]

    def fwd_ptr_inc(fwd_ptr):
        fwd_ptr += 1
        fwd_ptr = fwd_ptr % 8
        return fwd_ptr

    def bwd_ptr_inc(bwd_ptr):
        bwd_ptr += 1
        bwd_ptr = bwd_ptr % 8
        return bwd_ptr
    fwd_ptr = 0
    bwd_ptr = 0
    for iter in range(iter_num):
        for i in range(num_microbatches // 8):
            if iter == 0:
                if is_embbeding_rank:
                    output_tensor = forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr = fwd_ptr_inc(fwd_ptr)
                    if pipeline_parallel_rank % 2 == 0:
                        p2p_communication.send_forward(output_tensor, config)
                    else:
                        p2p_communication.send_backward(output_tensor, config)
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                    output_tensor = forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr = fwd_ptr_inc(fwd_ptr)
                    if pipeline_parallel_rank % 2 == 0:
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_forward_recv_backward(
                                    output_tensor,
                                    True,
                                    tensor_shape,
                                    config
                            )
                        )
                    else:
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_backward_recv_forward(
                                output_tensor,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                else:
                    if pipeline_parallel_rank % 2 == 0:
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(p2p_communication.recv_backward(tensor_shape, config))
                        output_tensor = forward_step_helper(fwd_order[fwd_ptr])
                        fwd_ptr = fwd_ptr_inc(fwd_ptr)
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_backward_recv_forward(
                                output_tensor,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                    else:
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(p2p_communication.recv_forward(tensor_shape, config))
                        output_tensor = forward_step_helper(fwd_order[fwd_ptr])
                        fwd_ptr = fwd_ptr_inc(fwd_ptr)
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_forward_recv_backward(
                                output_tensor,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                    output_tensor = forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr = fwd_ptr_inc(fwd_ptr)
                    if pipeline_parallel_rank % 2 == 0:
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_forward_recv_backward(
                                output_tensor,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                    else:
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_backward_recv_forward(
                                output_tensor,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            else:
                if is_embbeding_rank:
                    output_tensor1 = forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr = fwd_ptr_inc(fwd_ptr)
                    output_tensor2 = forward_step_helper(fwd_order[fwd_ptr])
                    fwd_ptr = fwd_ptr_inc(fwd_ptr)
                    if pipeline_parallel_rank % 2 == 0:
                        output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                            p2p_communication.send_forward_recv_backward(
                                output_tensor1,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                        deallocate_output_tensor(output_tensor1, config.deallocate_pipeline_outputs)
                        input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr])
                        bwd_ptr=bwd_ptr_inc(bwd_ptr)
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_forward_recv_backward(
                                output_tensor2,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                        deallocate_output_tensor(output_tensor2, config.deallocate_pipeline_outputs)
                    else:
                        output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                            p2p_communication.send_backward_recv_forward(
                                output_tensor1,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                        deallocate_output_tensor(output_tensor1, config.deallocate_pipeline_outputs)
                        input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr])
                        bwd_ptr=bwd_ptr_inc(bwd_ptr)
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_backward_recv_forward(
                                output_tensor2,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                        deallocate_output_tensor(output_tensor2, config.deallocate_pipeline_outputs)
                else:
                    input_tensor_grad=backward_step_helper(bwd_order[bwd_ptr])
                    bwd_ptr=bwd_ptr_inc(bwd_ptr)
                    if pipeline_parallel_rank % 2==1:
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_backward_recv_forward(
                                input_tensor_grad,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                        output_tensor=forward_step_helper(get_model_chunk_id(fwd_order[fwd_ptr]))
                        fwd_ptr=fwd_ptr_inc(fwd_ptr)
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_forward_recv_backward(
                                output_tensor,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                        output_tensor=forward_step_helper(get_model_chunk_id(fwd_order[fwd_ptr]))
                        fwd_ptr=fwd_ptr_inc(fwd_ptr)
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_backward_recv_forward(
                                output_tensor,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                    else:
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_forward_recv_backward(
                                input_tensor_grad,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                        output_tensor=forward_step_helper(get_model_chunk_id(fwd_order[fwd_ptr]))
                        fwd_ptr=fwd_ptr_inc(fwd_ptr)
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_backward_recv_forward(
                                output_tensor,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                        output_tensor=forward_step_helper(get_model_chunk_id(fwd_order[fwd_ptr]))
                        fwd_ptr=fwd_ptr_inc(fwd_ptr)
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_forward_recv_backward(
                                output_tensor,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                    
                        
            for j in range(5):
                output_tensor = forward_step_helper(fwd_order[fwd_ptr])
                fwd_ptr = fwd_ptr_inc(fwd_ptr)
                if j < 4 or not is_embbeding_rank:
                    if (j % 2) == (pipeline_parallel_rank % 2):
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_backward_recv_forward(
                                output_tensor,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                    else:
                        input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_forward_recv_backward(
                                output_tensor,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            torch.distributed.barrier()
            if is_embbeding_rank:
                input_tensor_grad = backward_step_helper(bwd_order[bwd_ptr])
                bwd_ptr = bwd_ptr_inc(bwd_ptr)
                if pipeline_parallel_rank % 2 == 0:
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_forward_recv_backward(
                                input_tensor_grad,
                                True,
                                tensor_shape,
                                config
                            )
                        )
                else:
                    input_tensors[get_model_chunk_id(fwd_order[fwd_ptr])].append(
                            p2p_communication.send_backward_recv_forward(
                                input_tensor_grad,
                                True,
                                tensor_shape,
                                config
                            )
                        )
            output_tensor = forward_step_helper(fwd_order[fwd_ptr])
            fwd_ptr=fwd_ptr_inc(fwd_ptr)
            if iter!=0 and i==0:
                update_and_post_process(iter)
            if not is_embbeding_rank:
                if pipeline_parallel_rank % 2 == 0:
                    output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                        p2p_communication.send_forward_recv_backward(
                                output_tensor,
                                True,
                                tensor_shape,
                                config
                        )
                    )
                else:
                    output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                        p2p_communication.send_backward_recv_forward(
                                output_tensor,
                                True,
                                tensor_shape,
                                config
                        )
                    )
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                input_tensor_grad = backward_step_helper(bwd_order[bwd_ptr])
                bwd_ptr = bwd_ptr_inc(bwd_ptr)
                if pipeline_parallel_rank % 2 == 0:
                    output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                        p2p_communication.send_backward_recv_forward(
                                input_tensor_grad,
                                True,
                                tensor_shape,
                                config
                        )
                    )
                else:
                    output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                        p2p_communication.send_forward_recv_backward(
                                input_tensor_grad,
                                True,
                                tensor_shape,
                                config
                        )
                    )
            last_iter = (i == num_microbatches // 8 - 1)
            for k in range(6):
                sync_grad = last_iter and is_last_microbatch_for_model_chunk(bwd_order[bwd_ptr])
                # if pipeline_parallel_rank == 7:
                #     print(k,'  ',iter,' ')
                input_tensor_grad = backward_step_helper(bwd_order[bwd_ptr], sync_grad)
                bwd_ptr = bwd_ptr_inc(bwd_ptr)
                if k < 5 or not is_embbeding_rank:
                    if (k % 2) == (pipeline_parallel_rank % 2):
                        output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                            p2p_communication.send_forward_recv_backward(
                                    input_tensor_grad,
                                    True,
                                    tensor_shape,
                                    config
                            )
                        )
                    else:
                        output_tensor_grads[get_model_chunk_id(bwd_order[bwd_ptr])].append(
                            p2p_communication.send_backward_recv_forward(
                                    input_tensor_grad,
                                    True,
                                    tensor_shape,
                                    config
                            )
                        )
    if is_embbeding_rank:
        if pipeline_parallel_rank % 2 == 0:
            output_tensor_grads[get_model_chunk_id(bwd_order[-1])].append(p2p_communication.recv_backward(tensor_shape,config))
        else:
            output_tensor_grads[get_model_chunk_id(bwd_order[-1])].append(p2p_communication.recv_forward(tensor_shape,config))
        input_tensor_grad = backward_step_helper(bwd_order[-1], True)
    if not is_embbeding_rank:
        if pipeline_parallel_rank % 2 == 0:
            p2p_communication.send_forward(input_tensor_grad, config)
        else:
            p2p_communication.send_backward(input_tensor_grad, config)
    update_and_post_process(iter_num)


def async_train_pipedream_pipeline(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    optimizer,
    opt_param_scheduler,
    iter_num,
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run asynchronous pipedream schedule, with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "pipedream pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "pipedream schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "pipedream pipeline parallelism does not support overlapping p2p communication"
        )
    
    model_type = get_model_type(model)
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != seq_length:
        raise RuntimeError(
            "Interleaving is not supported with a different decoder sequence length."
        )
    
    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()


    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_microbatches_remaining = iter_num - num_warmup_microbatches

    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    args = get_args()
    timers = get_timers()

    # Tracking loss.
    total_loss_dict = {}

    report_memory_flag = True

    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)
    
    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    def update_and_post_process(iteration):
        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Update parameters.
        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        timers('optimizer').stop()

        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        loss_reduced = {}
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            
            for key in forward_data_store[0]:
                losses_reduced_for_key = [x[key] for x in forward_data_store]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        # if iteration % args.log_interval == 0:
        #     track_e2e_metrics()
        nonlocal report_memory_flag
        batch_size = parallel_state.get_data_parallel_world_size() * \
                        args.micro_batch_size
        args.consumed_train_samples += batch_size
        report_memory_flag = training_log(loss_reduced, total_loss_dict,
                                            optimizer.param_groups[0]['lr'],
                                            iteration, loss_scale,
                                            report_memory_flag, skipped_iter,
                                            grad_norm, params_norm, num_zeros_in_grad)
        
        model.zero_grad_buffer()
        optimizer.zero_grad()
        forward_data_store.clear()
        
    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
    # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = p2p_communication.recv_forward(tensor_shape, config)
        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
        )
        p2p_communication.send_forward(output_tensor, config)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communication.recv_forward(tensor_shape, config)

    # Run 1F1B in steady state.
    for i in range(1, num_microbatches_remaining + 1):
        last_iteration = i == (num_microbatches_remaining)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 1) and (num_warmup_microbatches == 0)
            ),
        )
        output_tensor_grad = None
        if forward_only:
            p2p_communication.send_forward(output_tensor, config)

            if not last_iteration:
                input_tensor = p2p_communication.recv_forward(tensor_shape, config)
        else:
            if parallel_state.is_pipeline_last_stage() is False:
                output_tensor_grad = p2p_communication.send_forward_recv_backward(
                    output_tensor=output_tensor,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                    config=config
                )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or pipeline_parallel_rank == 0:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
            # if i % num_microbatches == 0:
            #     update_and_post_process(i // num_microbatches)
            update_and_post_process(i)

            if last_iteration:
                input_tensor = None
                p2p_communication.send_backward(input_tensor_grad, config)
            else:
                if parallel_state.is_pipeline_first_stage() is False:
                    input_tensor = p2p_communication.send_backward_recv_forward(
                        input_tensor_grad=input_tensor_grad,
                        recv_prev=True,
                        tensor_shape=tensor_shape,
                        config=config
                    )
    
    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or pipeline_parallel_rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communication.recv_backward(tensor_shape, config)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
            update_and_post_process(i + num_microbatches_remaining + 1)

            p2p_communication.send_backward(input_tensor_grad, config)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

