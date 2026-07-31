# Copyright 2025-2026 NeoteAI Team. All rights reserved.
from .lerobot_latent_dataset import MultiLatentLeRobotDataset as _DefaultMultiLatentLeRobotDataset
from .lerobot_latent_dataset_pi05_delta import MultiLatentLeRobotPi05DeltaDataset
from .bucket_sampler import BucketedDistributedBatchSampler


class MultiLatentLeRobotDataset:
    def __new__(cls, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        action_delta_mode = str(getattr(config, "action_delta_mode", "")).lower()
        if action_delta_mode in {"pi05_delta", "openpi_delta", "pi0.5_delta"}:
            return MultiLatentLeRobotPi05DeltaDataset(*args, **kwargs)
        return _DefaultMultiLatentLeRobotDataset(*args, **kwargs)

__all__ = [
    'MultiLatentLeRobotDataset',
    'MultiLatentLeRobotPi05DeltaDataset',
    'BucketedDistributedBatchSampler',
]
