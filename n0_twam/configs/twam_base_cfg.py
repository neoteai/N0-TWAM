# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Base configuration for the released N0-TWAM model.

Defines the model / action / tactile settings of the released checkpoint
(20-dim dual-arm EE action; MoT video / tactile / action experts). The inference
server (twam_server_cfg), the post-training config (twam_posttrain_cfg), and the
image-to-video-action demo (twam_i2va) all inherit from this and override only
what they need. Every path below is a placeholder — point it at your own data,
checkpoint, and base model.
"""
import torch
from easydict import EasyDict

from .shared_config import twam_shared_cfg

twam_base_cfg = EasyDict(__name__="Config: N0-TWAM base")
twam_base_cfg.update(twam_shared_cfg)
cfg = twam_base_cfg

# ───────── model / video ─────────
cfg.wan22_pretrained_model_name_or_path = "/path/to/base-model"
cfg.attn_window = 72
cfg.frame_chunk_size = 2
cfg.height = 256          # latent is already 256x256
cfg.width = 256
cfg.snr_shift = 5.0       # video / tactile resolution shift
cfg.action_snr_shift = 1.0
cfg.param_dtype = torch.bfloat16

# ───────── inference defaults (server / demo override as needed) ─────────
cfg.guidance_scale = 1
cfg.action_guidance_scale = 1
cfg.num_inference_steps = 25
cfg.action_num_inference_steps = 50
cfg.video_exec_step = -1

# ───────── observations ─────────
cfg.obs_cam_keys = ["observation.images.third_view",
                    "observation.images.left_wrist_view",
                    "observation.images.right_wrist_view"]
cfg.tactile_keys = ["observation.images.tactile_a",
                    "observation.images.tactile_b"]
cfg.tactile_optional = True
cfg.tactile_cfg_prob = 0.1
cfg.tactile_sensor_id_map = {"observation.images.tactile_a": 0,
                             "observation.images.tactile_b": 1}
cfg.max_tactile_streams = 4
# LocalTactile cross-attn branch into the action head. OFF in the released ckpt
# (GlobalTactile alone is enough for the general prior); post-training flips it ON
# and warm-starts the branch fresh + zero-init over the local-off checkpoint.
cfg.use_local_tactile = False
# LocalTactile stream form: 'current' (absolute current frame, streaming-robust)
# or 'residual' (frame[t]-frame[t-1]). Only matters when use_local_tactile is on.
cfg.local_tactile_mode = "current"
cfg.tactile_in_channels = 3
cfg.tactile_num_tokens = 4
cfg.tactile_encoder_dim = 256
cfg.tactile_latent_height = 8
cfg.tactile_latent_width = 8
cfg.tactile_resize = 128

# ───────── action: 20-dim pi0.5 horizon delta ─────────
cfg.action_dim = 20
cfg.action_delta_mode = "pi05_delta"
cfg.pi05_action_horizon = 16
cfg.action_per_frame = 16
cfg.pi05_source_action_is_delta = False   # absolute EE poses in the source
cfg.pi05_state_column = "observation.state"
cfg.pi05_delta_channel_ids = list(range(0, 9)) + list(range(10, 19))
cfg.pi05_condition_first_frame_zero = True
cfg.pi05_condition_first_frame_loss = False
cfg.pi05_invalid_horizon_mask = True
cfg.pi05_rot6d_relative_delta = False
cfg.used_action_channel_ids = list(range(20))   # dual-arm default
# Single-arm robots mask the right-arm half; the dataset loader applies this
# automatically by matching the first token of the repo name against this set.
cfg.single_arm_robot_prefixes = {"ur", "flexiv", "franka", "piper", "univtac", "neosim"}
_inverse = [len(cfg.used_action_channel_ids)] * cfg.action_dim
for _i, _j in enumerate(cfg.used_action_channel_ids):
    _inverse[_j] = _i
cfg.inverse_used_action_channel_ids = _inverse

# Action normalization: q01/q99 quantiles. Placeholder until you compute stats
# (script/build_task_pool.py); train.py and the server refuse to run on it.
cfg.action_norm_method = "q01q99"
cfg.norm_stat = {"q01": [-1.0] * 20, "q99": [1.0] * 20}

cfg.max_latent_frames = 0
cfg.max_tactile_frames = 0

# ───────── predictive-contact gate (opt-in add-on, OFF in the released ckpt) ─────────
# When enabled, the (predicted) GlobalTactile hidden FiLM-modulates the action
# stream. Zero-init FiLM => identity at start, so flipping it on over an existing
# checkpoint is unperturbed at step 0.
cfg.use_contact_gate = False
cfg.contact_gate_layers = 2
cfg.contact_gate_heads = 8
cfg.contact_gate_stop_grad = False

# ───────── MoT backbone (= released checkpoint) ─────────
# Narrow action / tactile experts (hidden 1024 / FFN 4096; q/k/v project to the
# shared 3072). On resume the shape follows the checkpoint's config.json.
cfg.use_mot = True
cfg.mot_cross_attn_experts = ("video", "action")
cfg.mot_warmstart_experts = ("video", "action", "tactile")
cfg.mot_expert_hidden_dim = {"action": 1024, "tactile": 1024}
cfg.mot_expert_ffn_dim = {"action": 4096, "tactile": 4096}

# ───────── data / training defaults (post-training overrides these) ─────────
cfg.dataset_path = "/path/to/your_dataset/train"
cfg.val_dataset_path = "/path/to/your_dataset/val"
cfg.val_interval = 100
cfg.empty_emb_path = "/path/to/base-model/empty_emb.pt"
cfg.resume_from = None
cfg.save_root = "./train_out"
cfg.enable_wandb = False
cfg.load_worker = 4
cfg.num_init_worker = 1
cfg.batch_size = 1
cfg.gradient_accumulation_steps = 4
cfg.num_steps = 2000
cfg.save_interval = 500
cfg.gc_interval = 50
cfg.learning_rate = 1e-4
cfg.beta1, cfg.beta2 = 0.9, 0.95
cfg.weight_decay = 0.1
cfg.warmup_steps = 20
cfg.cfg_prob = 0.1
cfg.noisy_cond_prob_tactile = 0.5
cfg.lr_schedule = "cosine"
cfg.lr_min_ratio = 0.1
