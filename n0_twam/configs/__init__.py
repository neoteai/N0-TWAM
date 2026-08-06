# Copyright 2025-2026 NeoteAI Team. All rights reserved.
from .twam_base_cfg import twam_base_cfg
from .twam_i2va import twam_i2va_cfg
from .twam_server_cfg import twam_server_cfg
from .twam_posttrain_cfg import twam_posttrain_cfg
from .twam_posttrain_server_cfg import twam_posttrain_server_cfg
from .twam_multitask_server_cfg import twam_multitask_server_cfg

TWAM_CONFIGS = {
    "base": twam_base_cfg,
    "twam_i2va": twam_i2va_cfg,
    "twam_server": twam_server_cfg,
    "posttrain": twam_posttrain_cfg,
    "posttrain_server": twam_posttrain_server_cfg,
    "multitask_server": twam_multitask_server_cfg,
}
