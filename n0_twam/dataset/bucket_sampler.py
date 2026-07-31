# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Shape-bucketed distributed batch sampler for per-GPU batch_size > 1.

The multi-robot pretraining pool is heterogeneous: camera count (1 vs 3) and
tactile-stream count (0 / 2 / 4) vary across repos, so torch's default_collate
cannot stack arbitrary samples into a batch (shapes differ). This sampler groups
samples that share a *shape signature* — (n_cameras, n_tactile_streams) — into
batches, so every batch is collatable, then shards the batches equally across
distributed ranks (each rank gets the SAME number of batches => no FSDP/DDP
step desync).

Latent frame count (33) and the action tensor (20-dim, fixed horizon) are
uniform across the whole pool, so the signature only needs (n_cam, n_tac).

batch_size == 1 does not need this (every singleton is trivially collatable);
train.py only switches to it when config.batch_size > 1.
"""
from collections import defaultdict

import torch


class BucketedDistributedBatchSampler:
    """Yield lists of dataset indices (batches) whose samples share a shape
    signature, sharded equally across ``num_replicas`` ranks.

    Args:
        signatures: list mapping global sample index -> hashable shape signature
            (e.g. ``(n_cam, n_tac)``). ``len(signatures) == len(dataset)``.
        batch_size: per-rank batch size (>= 2 in practice).
        num_replicas / rank: distributed world size / this rank.
        shuffle: shuffle within buckets and across batches each epoch.
        seed: base RNG seed (combined with epoch so all ranks agree).
        drop_last: drop each bucket's ragged tail so every batch is exactly
            ``batch_size`` (keeps tensor shapes rigid). Also drops the
            ``total_batches % num_replicas`` remainder so ranks stay balanced.
    """

    def __init__(self, signatures, batch_size, num_replicas=1, rank=0,
                 shuffle=True, seed=42, drop_last=True):
        self.signatures = list(signatures)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        buckets = defaultdict(list)
        for idx, sig in enumerate(self.signatures):
            buckets[sig].append(idx)
        self._buckets = dict(buckets)

        # Deterministic batch count: with drop_last the per-bucket batch count
        # is independent of the shuffle, so __len__ is stable across epochs.
        if self.drop_last:
            total = sum(len(v) // self.batch_size for v in self._buckets.values())
        else:
            total = sum((len(v) + self.batch_size - 1) // self.batch_size
                        for v in self._buckets.values())
        self._num_batches = total // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _all_batches(self):
        """Build the full (rank-independent) list of batches for this epoch.

        Deterministic given (seed, epoch): every rank computes the identical
        list and then slices its own disjoint, equal-sized share.
        """
        gen = torch.Generator()
        gen.manual_seed(self.seed + self.epoch)
        batches = []
        for _sig, idxs in sorted(self._buckets.items(), key=lambda kv: str(kv[0])):
            n = len(idxs)
            order = (torch.randperm(n, generator=gen).tolist()
                     if self.shuffle else list(range(n)))
            limit = (n // self.batch_size) * self.batch_size if self.drop_last else n
            for i in range(0, limit, self.batch_size):
                batches.append([idxs[order[j]]
                                for j in range(i, min(i + self.batch_size, n))])
        if self.shuffle and batches:
            perm = torch.randperm(len(batches), generator=gen).tolist()
            batches = [batches[i] for i in perm]
        return batches

    def __iter__(self):
        batches = self._all_batches()
        per_rank = self._num_batches
        start = self.rank * per_rank
        return iter(batches[start:start + per_rank])

    def __len__(self):
        return self._num_batches


__all__ = ["BucketedDistributedBatchSampler"]
