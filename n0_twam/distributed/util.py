# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist
from datetime import timedelta


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    """Prepare a model for distributed use: optionally freeze it for eval, then
    apply the sharding function and move it to the target device/dtype."""
    if eval_mode:
        model.eval().requires_grad_(False)
    if dist.is_initialized():
        dist.barrier()

    if dist.is_initialized():
        model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)

    return model


def init_distributed(world_size, local_rank, rank):
    # if world_size > 1:
    torch.cuda.set_device(local_rank)
    # Bind the NCCL communicator to THIS rank's local GPU explicitly. Without
    # device_id, NCCL guesses the device from the *global* rank, which is wrong
    # on multi-node (global rank 8 -> cuda:8, but node 1 only has cuda:0-7) and
    # deadlocks the first collective -> watchdog SIGABRT. Single-node happened to
    # work because global rank == local rank there.
    dist.init_process_group(backend="nccl",
                            timeout=timedelta(seconds=1800),
                            init_method="env://",
                            rank=rank,
                            world_size=world_size,
                            device_id=torch.device("cuda", local_rank))

def dist_mean(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
    return local_tensor

def dist_max(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return local_tensor
