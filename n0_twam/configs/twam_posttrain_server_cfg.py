# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Serve config for a post-trained checkpoint (registered "posttrain_server").

Inherits twam_posttrain_cfg wholesale so every serve-critical field (norm,
keys, action space, ablation switches) has a single source of truth; only
server-mode fields are overridden. Point wan22_pretrained_model_name_or_path
at a bundle from script/make_serve_bundle.py, then:

    python -m n0_twam.n0_twam_server --config-name posttrain_server --port 29601
"""
from easydict import EasyDict

from .twam_posttrain_cfg import twam_posttrain_cfg

s = EasyDict()
s.update(twam_posttrain_cfg)
s.__name__ = "Config: N0-TWAM post-train SERVER"

s.infer_mode = "server"
s.host = "0.0.0.0"
s.port = 29601                     # --port on the command line overrides this

# serve bundle: transformer/ (your post-trained ckpt) + vae/ tokenizer/
# text_encoder/ symlinked from the base model — see script/make_serve_bundle.py.
s.wan22_pretrained_model_name_or_path = "/path/to/serve-bundle"
s.resume_from = None
s.save_root = "/path/to/serve-output"      # per-infer dumps land here; must be writable

# prompt fallback for bare resets — use the training text VERBATIM.
s.prompt = twam_posttrain_cfg.eval_prompt

# Denoising steps (pure inference knob; video steps dominate latency).
# 3/4 = the baseline all internal closed-loop scores used; raise the cheap
# action steps first, keep fixed within an eval batch.
s.num_inference_steps = 3
s.action_num_inference_steps = 4
s.video_exec_step = -1
s.guidance_scale = 1
s.action_guidance_scale = 1
s.enable_offload = False

# deterministic_episode_seed: per-episode RNG from the client seed.
# cold_seed_mode (absEE only): "free" (default) | "current_state" | "zeros".
s.deterministic_episode_seed = True
s.cold_seed_mode = "free"

# optional chunk-boundary ramp (delta mode only); measured neutral on success.
s.delta_smooth = False
s.delta_smooth_k = 3

# de-normalized ABSOLUTE poses (delta mode reconstructs from current_state);
# all 20 channels returned, single-arm clients read the first 10.
s.server_action_output_format = "absolute"
s.server_return_action_channel_ids = list(range(20))

# co-denoise GlobalTactile with the video (training-consistent); ablation
# switches inherit from the training config — never override them here alone.
s.server_tactile_denoise = True
s.use_local_tactile = bool(twam_posttrain_cfg.use_local_tactile)
s.tactile_global_zero = bool(twam_posttrain_cfg.tactile_global_zero)

s.enable_wandb = False

assert s.local_tactile_mode == "current"
assert s.action_delta_mode in ("none", "pi05_delta")
assert s.server_action_output_format == "absolute"

twam_posttrain_server_cfg = s
