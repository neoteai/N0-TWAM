# Copyright 2025-2026 NeoteAI Team. All rights reserved.
import json
from pathlib import Path

from easydict import EasyDict
from .twam_base_cfg import twam_base_cfg

# Image-to-video-action demo: roll the model forward from the example frames in
# `example_client/cam_test/` (named after the pretrain camera keys) without a robot.
twam_i2va_cfg = EasyDict()
twam_i2va_cfg.update(twam_base_cfg)
twam_i2va_cfg.__name__ = 'Config: N0-TWAM i2va'

# Point this at the released bundle (transformer/ vae/ tokenizer/ text_encoder/).
twam_i2va_cfg.wan22_pretrained_model_name_or_path = "/path/to/n0-twam-bundle"
twam_i2va_cfg.input_img_path = 'example_client/cam_test'
twam_i2va_cfg.num_chunks_to_infer = 30
twam_i2va_cfg.prompt = 'pick up the object'
twam_i2va_cfg.infer_mode = 'i2va'

# Pretrain norm stats ship at the bundle root (same auto-load as twam_server).
_norm_path = Path(twam_i2va_cfg.wan22_pretrained_model_name_or_path) / 'norm_stat_pretrain.json'
if _norm_path.is_file():
    twam_i2va_cfg.norm_stat = json.loads(_norm_path.read_text())
    twam_i2va_cfg.norm_stat_path = str(_norm_path)
    twam_i2va_cfg.per_repo_norm_stat = {}
