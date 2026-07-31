# Copyright 2025-2026 NeoteAI Team. All rights reserved.
import json
from pathlib import Path

from easydict import EasyDict
from .twam_base_cfg import twam_base_cfg

# Inference-server config: serves the pretrained N0-TWAM model over a websocket.
# It inherits the model / action / tactile settings from the pretrain config so
# the action space (20-dim pi0.5 delta) and tactile channels match the released
# checkpoint; only serving-specific fields are overridden below.
twam_server_cfg = EasyDict()
twam_server_cfg.update(twam_base_cfg)
twam_server_cfg.__name__ = 'Config: N0-TWAM server'

twam_server_cfg.infer_mode = 'server'
twam_server_cfg.host = '0.0.0.0'
twam_server_cfg.port = 29536

# Point this at a *serve bundle*: a directory that contains the trained
# `transformer/` (your checkpoint) plus `vae/`, `tokenizer/` and `text_encoder/`
# copied or symlinked from the base model. See docs/DEPLOY.md.
twam_server_cfg.wan22_pretrained_model_name_or_path = '/path/to/serve-bundle'
twam_server_cfg.save_root = '/path/to/serve-output'

# Action norm stats the released checkpoint was TRAINED with. The HF bundle
# ships them as `norm_stat_pretrain.json` at its root — picked up automatically
# once the bundle path above is set. Without real stats the server refuses to
# start (it must de-normalize actions with the training-time quantiles).
_norm_path = Path(twam_server_cfg.wan22_pretrained_model_name_or_path) / 'norm_stat_pretrain.json'
if _norm_path.is_file():
    twam_server_cfg.norm_stat = json.loads(_norm_path.read_text())
    twam_server_cfg.norm_stat_path = str(_norm_path)
    twam_server_cfg.per_repo_norm_stat = {}

# Debug-only: load the VAE / text-encoder on CPU (and keep them there) to save
# VRAM. Per-frame VAE encodes then run on CPU — minutes per chunk — so leave
# this False for real serving (see docs/DEPLOY.md, "GPU memory").
twam_server_cfg.enable_offload = False

# Denoising steps at inference: fewer = faster, more = higher fidelity.
twam_server_cfg.num_inference_steps = 10
twam_server_cfg.action_num_inference_steps = 10

# Classifier-free guidance scales (1 = disabled = one forward pass per step).
twam_server_cfg.guidance_scale = 1
twam_server_cfg.action_guidance_scale = 1

# Co-denoise the GlobalTactile stream with the video and condition the action
# pass on the IMAGINED tactile — consistent with training, where video /
# tactile / action are jointly generated.
twam_server_cfg.server_tactile_denoise = True

# Seed the diffusion RNG with the client's per-episode seed on reset
# (reproducible eval); cold_seed_mode only applies to absolute-EE checkpoints.
twam_server_cfg.deterministic_episode_seed = True
twam_server_cfg.cold_seed_mode = 'free'
