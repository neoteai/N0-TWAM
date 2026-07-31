# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

def _mot_expert_stacks(model):
    """For a MoT model, yield each per-expert block ModuleList; else None."""
    if hasattr(model, "mot"):
        return [model.mot.experts[name].blocks for name in model.mot.expert_names]
    return None


def apply_ac(model):
    """Apply activation checkpointing to the model's transformer blocks."""
    stacks = _mot_expert_stacks(model)
    if stacks is None:
        for layer_id, transformer_block in enumerate(model.blocks):
            model.blocks[layer_id] = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        return
    # MoT: do NOT per-block checkpoint. The block is split into pre/post around the
    # shared attention, so a per-block wrapper leaves q/k/v/attn (~10GB at S~15k)
    # resident -> 80GB OOM. Instead MoTBackbone.forward checkpoints the WHOLE layer
    # body (legacy granularity), recomputing q/k/v/attn/ffn in backward.
    if hasattr(model, "mot"):
        model.mot.gradient_checkpointing = True


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}

    stacks = _mot_expert_stacks(model)
    if stacks is None:
        for block in model.blocks:
            fully_shard(block.attn1, **fsdp_config)
            fully_shard(block.attn2, **fsdp_config)
            fully_shard(block.ffn, **fsdp_config)
            fully_shard(block, **fsdp_config)
    else:
        # MoT: shard each expert block as ONE FSDP unit (it IS run via forward, so
        # the all-gather hook fires and the block reshards after each pre/post call).
        # We do NOT separately shard attn1/attn2/ffn: those are invoked via
        # project_qkv/merge_out (not their own forward), so a nested fully_shard
        # would bypass their hook. Block-level sharding gathers them together.
        for stack in stacks:
            for block in stack:
                fully_shard(block, **fsdp_config)

    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
