# Copyright 2025-2026 NeoteAI Team. All rights reserved.
import torch
from easydict import EasyDict

twam_shared_cfg = EasyDict()

twam_shared_cfg.host = '0.0.0.0'
twam_shared_cfg.port = 29536

twam_shared_cfg.param_dtype = torch.bfloat16
twam_shared_cfg.save_root = './train_out'

twam_shared_cfg.patch_size = (1, 2, 2)

twam_shared_cfg.enable_offload = True

twam_shared_cfg.tactile_keys = []
twam_shared_cfg.max_tactile_streams = 4
twam_shared_cfg.tactile_height = 64
twam_shared_cfg.tactile_width = 64
twam_shared_cfg.synthetic_tactile_data = False

# ───── New (latent-tactile pipeline) ─────
# Shared by the global and local tactile pathways
twam_shared_cfg.tactile_latent_root_name = 'latents_tactile'   # dir under dataset root
twam_shared_cfg.tactile_sensor_id_map = {                       # tactile_key → sensor_id
    # single-arm default — override per-dataset cfg if more sensors
    'observation.images.tactile_a': 0,
    'observation.images.tactile_b': 1,
    # Reserved IDs for future bimanual setups:
    # 'observation.images.tactile_ll': 0, 'observation.images.tactile_lr': 1,
    # 'observation.images.tactile_rl': 2, 'observation.images.tactile_rr': 3,
}
twam_shared_cfg.tactile_latent_height = 8      # 128 / 16 (Wan VAE 16x spatial compress)
twam_shared_cfg.tactile_latent_width = 8
twam_shared_cfg.tactile_cfg_prob = 0.1          # prob to drop tactile (CFG dropout)
