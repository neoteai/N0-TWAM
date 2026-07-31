# Copyright 2025-2026 NeoteAI Team. All rights reserved.
import argparse
import os
import sys
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import TWAM_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from models.utils import (
    load_transformer,
    load_mot_transformer,
    load_mot_checkpoint,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset, BucketedDistributedBatchSampler
import gc


class Trainer:
    def __init__(self, config, inference_only=False):
        if config.enable_wandb and config.rank == 0 and not inference_only:
            # self-hosted wandb via WANDB_BASE_URL/WANDB_API_KEY; else standard wandb.ai
            if os.getenv('WANDB_BASE_URL') and os.getenv('WANDB_API_KEY'):
                wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                entity=os.getenv("WANDB_TEAM_NAME") or None,
                project=os.getenv("WANDB_PROJECT", "twam-pretrain"),
                config=config,
                mode="online",
                name=os.getenv("WANDB_RUN_NAME", "twam-train")
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        # FlowMatch schedulers — needed by _add_noise / _prepare_input_dict and by
        # inference sampling, so set them up before the (training-only) heavy setup.
        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)
        # symdiff: GlobalTactile becomes a diffusion target. Re-use video snr_shift.
        self.train_scheduler_tactile = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_tactile.set_timesteps(1000, training=True)
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)

        if inference_only:
            # Lean setup for rendering / denoising-consistency checks: the caller
            # assigns self.transformer (e.g. via load_mot_checkpoint) and builds
            # any dataset it needs. No FSDP / optimizer / dataloaders / wandb here,
            # so all the _prepare_input_dict / _add_noise / compute_loss machinery
            # is reused EXACTLY (the inference must match training, not re-derive it).
            return

        # Refuse placeholder norm stats (mirror of the server-side guard).
        _ns = getattr(config, 'norm_stat', None) or {}
        _q01 = [float(v) for v in _ns.get('q01', [])]
        _q99 = [float(v) for v in _ns.get('q99', [])]
        if _q01 and _q01 == [-1.0] * len(_q01) and _q99 == [1.0] * len(_q99):
            raise RuntimeError(
                'norm_stat is the [-1, 1] placeholder — the stats file was '
                f'missing when the config was imported '
                f'(norm_stat_path={getattr(config, "norm_stat_path", None)!r}). '
                'Compute the norm stats (script/build_task_pool.py) before training.')

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        _loader_kwargs = dict(
            torch_dtype=torch.float32,
            torch_device='cpu',
            attn_mode="flex",
            target_action_dim=int(getattr(config, 'action_dim', 30)),
            max_tactile_streams=config.max_tactile_streams,
            # LocalTactile cross-attn branch. The released pretrain checkpoint has
            # it OFF; the post-train config flips it True (built zero-init on
            # resume). Default False = match the released checkpoint.
            use_local_tactile=bool(getattr(config, 'use_local_tactile', False)),
            # opt-in predictive-contact gate (defaults OFF -> base ckpts unchanged;
            # passed as from_pretrained kwarg so it overrides the ckpt config when
            # resuming a checkpoint whose config predates the gate).
            use_contact_gate=bool(getattr(config, 'use_contact_gate', False)),
            contact_gate_layers=int(getattr(config, 'contact_gate_layers', 2)),
            contact_gate_heads=int(getattr(config, 'contact_gate_heads', 8)),
            contact_gate_stop_grad=bool(getattr(config, 'contact_gate_stop_grad', True)),
        )
        if bool(getattr(config, 'use_mot', False)):
            # Mixture-of-Transformers: 3 per-modality experts warm-started from the
            # shared backbone. Cross-attn / warm-start experts are config-overridable.
            # Built in bf16: an fp32 3-expert build on every rank OOMs the container
            # memory cgroup. (Proper fp32-master via meta-init is a later optimization.)
            _mot_kwargs = dict(_loader_kwargs)
            _mot_kwargs['torch_dtype'] = torch.bfloat16
            if hasattr(config, 'resume_from') and config.resume_from:
                # Resuming: transformer_path is a SAVED MoT checkpoint (is_mot in its
                # config.json). Load it directly (rebuild saved expert structure +
                # strict weights) — do NOT warm-start from a legacy backbone.
                if config.rank == 0:
                    logger.info(f"Resuming MoT via load_mot_checkpoint: {transformer_path}")
                # Post-training may flip on modules the pretrain ckpt config lacks
                # (use_local_tactile / use_contact_gate): pass them as overrides so
                # the branch is built (zero-init) and its absent-in-ckpt weights are
                # tolerated, instead of rebuilding purely from the local-off ckpt
                # config. Matches the load_mot_transformer (non-resume) path, which
                # already receives these via **_mot_kwargs.
                _mot_overrides = {
                    k: _mot_kwargs[k]
                    for k in ('use_local_tactile', 'use_contact_gate',
                              'contact_gate_layers', 'contact_gate_heads',
                              'contact_gate_stop_grad')
                    if k in _mot_kwargs
                }
                self.transformer = load_mot_checkpoint(
                    transformer_path, torch_dtype=torch.bfloat16,
                    torch_device='cpu', attn_mode='flex',
                    config_overrides=_mot_overrides)
            else:
                if config.rank == 0:
                    logger.info("Building Mixture-of-Transformers (MoT) model (bf16).")
                self.transformer = load_mot_transformer(
                    transformer_path,
                    mot_expert_ffn_dim=getattr(config, 'mot_expert_ffn_dim', None),
                    mot_expert_hidden_dim=getattr(config, 'mot_expert_hidden_dim', None),
                    mot_cross_attn_experts=tuple(getattr(
                        config, 'mot_cross_attn_experts', ("video", "action"))),
                    mot_warmstart_experts=tuple(getattr(
                        config, 'mot_warmstart_experts', ("video", "action", "tactile"))),
                    **_mot_kwargs,
                )
        else:
            self.transformer = load_transformer(transformer_path, **_loader_kwargs)

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        if str(getattr(config, 'lr_schedule', 'constant')) == 'cosine':
            from utils import warmup_cosine_lambda
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: warmup_cosine_lambda(
                    step, warmup_steps=config.warmup_steps,
                    total_steps=config.num_steps,
                    min_ratio=float(getattr(config, 'lr_min_ratio', 0.1))))
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(config=config)
        _bs = int(getattr(config, 'batch_size', 1))
        if _bs > 1:
            # Per-GPU batch>1: the pool is shape-heterogeneous (cameras 1/3,
            # tactile streams 0/2/4) so default_collate cannot stack arbitrary
            # samples. Bucket same-signature samples into batches and shard them
            # equally across ranks (each rank gets the same #batches => no FSDP
            # step desync). Requires the dataset to expose per-sample signatures.
            if not hasattr(train_dataset, 'sample_signatures'):
                raise RuntimeError(
                    "batch_size>1 needs a dataset exposing .sample_signatures "
                    "for shape bucketing (only the pi05-delta dataset does)."
                )
            train_batch_sampler = BucketedDistributedBatchSampler(
                train_dataset.sample_signatures,
                batch_size=_bs,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                seed=42,
                drop_last=True,
            )
            self.train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                num_workers=config.load_worker,
            )
            if config.rank == 0:
                logger.info(
                    f"Bucketed batch_size={_bs}: {len(train_batch_sampler)} "
                    f"batches/rank across {config.world_size} ranks."
                )
        else:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                seed=42
            ) if config.world_size > 1 else None
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=(train_sampler is None),
                num_workers=config.load_worker,
                sampler=train_sampler,
            )

        # Optional validation set: episode-level held-out repos. Loss-only —
        # never touches the optimizer. Sharded across ranks like training.
        self.val_loader = None
        val_path = getattr(config, 'val_dataset_path', None)
        if val_path:
            import copy as _copy
            val_config = _copy.copy(config)
            val_config.dataset_path = val_path
            val_dataset = MultiLatentLeRobotDataset(config=val_config)
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=False,
            ) if config.world_size > 1 else None
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=2,
                sampler=val_sampler,
            )
            self.val_interval = int(getattr(config, 'val_interval', 100))
            if config.rank == 0:
                logger.info(f"Validation set: {len(val_dataset)} segments, "
                            f"every {self.val_interval} steps")

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader_iter = None
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler/batch_sampler and iterator when epoch finishes.
            # batch_size>1 uses a bucketed batch_sampler; batch_size==1 uses a
            # plain (Distributed)Sampler. Both expose set_epoch for reshuffle.
            for _samp in (getattr(self.train_loader, 'batch_sampler', None),
                          self.train_loader.sampler):
                if _samp is not None and hasattr(_samp, 'set_epoch'):
                    _samp.set_epoch(getattr(_samp, 'epoch', 0) + 1)
                    break
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from n0_twam_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']
        # Tactile inputs (shared CFG-drop mechanism):
        #   - tactile_cond_drop is explicit CFG dropout; when set, no tactile
        #     tensors are passed to the model so tactile modules get no grad
        #     (the model's tactile_zero_anchor keeps their params in the graph).
        #   - LocalTactile + sensor_ids are a clean cross-attention condition.
        # The CFG-drop decision is made ONCE in convert_input_format (which runs
        # immediately before this in both _train_step and _validate) and stashed
        # on self, so device movement there and the target/forward path here use
        # the SAME flag — no independent re-draw that could disagree (the old
        # double-decision built a target from a CPU latent -> device crash).
        tactile_cond_drop = getattr(self, '_tactile_cond_drop', False)
        action_dict['tactile_cond_drop'] = torch.tensor(
            tactile_cond_drop, dtype=torch.bool, device=self.device)
        if not tactile_cond_drop:
            for key in ('tactile_local_latent', 'tactile_sensor_ids'):
                if key in batch_dict:
                    action_dict[key] = batch_dict[key]
            # symdiff: noise the GlobalTactile latent so the dedicated
            # tactile_proj_out head can predict the velocity. Treat each sensor as
            # an independent diffusion sample by folding S into the batch dim.
            # GlobalTactile is a diffusion target here, not a clean condition, so
            # the raw tactile_global_latent is consumed via the noised tensors below.
            if 'tactile_global_latent' in batch_dict:
                g = batch_dict['tactile_global_latent']                # (B, S, C, F, H, W)
                if getattr(self.config, 'tactile_global_zero', False):
                    # global-ablation: zero the GlobalTactile latent so the noisy
                    # segment (= pure noise) and clean condition carry no contact
                    # info; sequence/loss structure stays identical to full model.
                    if not getattr(self, '_tgz_logged', False):
                        self._tgz_logged = True
                        logger.info(
                            "tactile_global_zero ACTIVE: zeroing global latent "
                            "(incoming max=%.4f)", g.abs().max().item())
                    g = torch.zeros_like(g)
                B, S, C, F, H, W = g.shape
                g_flat = g.reshape(B * S, C, F, H, W).contiguous()
                tdict = self._add_noise(
                    latent=g_flat,
                    train_scheduler=self.train_scheduler_tactile,
                    action_mask=None,
                    action_mode=False,
                    noisy_cond_prob=getattr(self.config, 'noisy_cond_prob_tactile', 0.5),
                )
                action_dict['tactile_global_noisy_latent'] = (
                    tdict['noisy_latents'].reshape(B, S, C, F, H, W))
                action_dict['tactile_global_clean_latent'] = (
                    tdict['latent'].reshape(B, S, C, F, H, W))
                action_dict['tactile_global_targets'] = (
                    tdict['targets'].reshape(B, S, C, F, H, W))
                # timesteps from _add_noise are (B*S, F); collapse the S dim by
                # taking the first sensor's schedule (per-frame timestep is shared).
                action_dict['tactile_global_timesteps'] = (
                    tdict['timesteps'].reshape(B, S, F)[:, 0])         # (B, F)
                action_dict['tactile_global_cond_timesteps'] = (
                    tdict['cond_timesteps'].reshape(B, S, F)[:, 0])    # (B, F)

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        # Decide the tactile CFG-drop ONCE here, BEFORE moving tensors, and stash
        # it on self so _prepare_input_dict reuses the SAME flag. Otherwise the
        # drop used HERE (for device movement) and one re-drawn in _prepare can
        # disagree: a "drop" here skips moving tactile_global_latent to GPU, then
        # a "keep" in _prepare builds the tactile target from that CPU tensor ->
        # mse_loss device-mismatch crash. Batch-level (one flag/batch); shape
        # bucketing keeps tactile presence uniform within a batch, so one draw is
        # correct and avoids per-sample OR-ing that over-drops at batch>1.
        _has_tactile = (('tactile_global_latent' in input_dict)
                        or ('tactile_local_latent' in input_dict))
        if not _has_tactile:
            tactile_cond_drop = True
        else:
            _cfg_p = float(getattr(self.config, 'tactile_cfg_prob', 0.1))
            tactile_cond_drop = bool(torch.rand(1).item() < _cfg_p)
        self._tactile_cond_drop = tactile_cond_drop
        for key, value in input_dict.items():
            if tactile_cond_drop and key in (
                'tactile_global_latent',
                'tactile_local_latent',
            ):
                continue
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def _compute_latent_loss(self, input_dict, latent_pred):
        latent_target = input_dict['latent_dict']['targets']
        # Transformer video output is a patch sequence:
        #   latent_pred_seq: [B, N_video_tokens, C_patch]
        # Convert it back to dense FlowMatch target layout:
        #   latent_pred/latent_target: [B, C_latent, F_video, H_latent, W_latent]
        latent_pred = data_seq_to_patch(
            self.patch_size, latent_pred,
            latent_target.shape[-3],
            latent_target.shape[-2],
            latent_target.shape[-1],
            batch_size=latent_pred.shape[0])
        # timesteps: [B, F_video], one sampled diffusion/flow timestep per frame.
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        # latent_loss_weight: [B, F_video], broadcast over C/H/W below.
        latent_loss_weight = self.train_scheduler_latent.training_weight(
            input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Per-element MSE:
        #   latent_loss: [B, C_latent, F_video, H_latent, W_latent]
        latent_loss = F.mse_loss(
            latent_pred.float(),
            latent_target.float().detach(),
            reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Move frame next to batch and flatten all non-frame dimensions:
        #   [B, C, F, H, W] -> [B, F, H, W, C] -> [B*F, H*W*C]
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)
        # Normalize each frame by its element count, then average over B*F.
        latent_loss_per_frame = latent_loss.sum(dim=1)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)
        return (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

    def _compute_action_loss(self, input_dict, action_pred):
        action_target = input_dict['action_dict']['targets']
        action_mask = input_dict['action_dict']['actions_mask'].float()
        # Transformer action output is a token sequence:
        #   action_pred_seq: [B, F_action * N_action, C_action]
        # Dense action target/mask layout is:
        #   action_target/action_mask: [B, C_action, F_action, N_action, 1]
        action_pred = rearrange(
            action_pred,
            'b (f n) c -> b c f n 1',
            f=action_target.shape[-3])
        # timesteps: [B, F_action], one sampled diffusion/flow timestep per frame.
        Bn, Fn = input_dict['action_dict']['timesteps'].shape
        # action_loss_weight: [B, F_action], broadcast over C/N below.
        action_loss_weight = self.train_scheduler_action.training_weight(
            input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Per-element MSE:
        #   action_loss: [B, C_action, F_action, N_action, 1]
        action_loss = F.mse_loss(
            action_pred.float(),
            action_target.float().detach(),
            reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * action_mask
        # Move frame next to batch and flatten action channels/horizon:
        #   [B, C, F, N, 1] -> [B, F, N, 1, C] -> [B*F, N*C]
        action_loss = action_loss.permute(0, 2, 3, 4, 1)
        action_mask = action_mask.permute(0, 2, 3, 4, 1)
        action_loss = action_loss.flatten(0, 1).flatten(1)
        action_mask = action_mask.flatten(0, 1).flatten(1)
        # Normalize by valid action elements per frame. Frames with no valid
        # action tokens, such as pi05's first condition frame, do not contribute.
        action_loss_per_frame = action_loss.sum(dim=1)
        action_mask_per_frame = action_mask.sum(dim=1)
        valid_frame = action_mask_per_frame > 0
        if not valid_frame.any():
            return action_loss_per_frame.sum() * 0.0
        return (action_loss_per_frame[valid_frame] / action_mask_per_frame[valid_frame]).mean()

    def compute_loss(self,
        input_dict,
        pred
    ):
        # Symdiff: pred is 3-tuple (video_pred, action_pred, tactile_pred).
        # Legacy (video+action): pred is 2-tuple (video_pred, action_pred). Handle both.
        #   latent_pred: [B, N_video_tokens, C_patch]
        #   action_pred: [B, F_action * N_action, C_action]
        if isinstance(pred, tuple) and len(pred) == 3:
            latent_pred, action_pred, tactile_pred = pred
        else:
            latent_pred, action_pred = pred
            tactile_pred = None

        latent_loss = self._compute_latent_loss(input_dict, latent_pred)
        action_loss = self._compute_action_loss(input_dict, action_pred)

        # Symdiff tactile diffusion MSE: target = ε - x_0 (flow matching),
        # stored under action_dict['tactile_global_targets']. Zero when target
        # absent (CFG drop) — but we MUST still touch tactile_pred so the
        # tactile_proj_out parameters get a grad and FSDP doesn't see mixed
        # dtypes (None grad treated as fp32 vs bf16 grad on the rest).
        device = latent_pred.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        tactile_loss = zero
        if tactile_pred is not None:
            if 'tactile_global_targets' in input_dict['action_dict']:
                target = input_dict['action_dict']['tactile_global_targets']
                # target: dense (B, S, C, F, H, W). tactile_pred is a patch
                # sequence (B, S*F*H*W, C) in (sensor, [f hp wp], [pf ph pw]) patch
                # order — the SAME convention as the video latent prediction. So,
                # exactly like _compute_latent_loss, un-patchify the prediction
                # back to dense (folding the sensor axis into the batch) and
                # compare dense-to-dense. The previous code instead row-major-
                # flattened the *target*, which silently mismatched the patch-block
                # ordering of the prediction and supervised the wrong spatial cells.
                B, S, C, F_lat, H_lat, W_lat = target.shape
                pred_dense = data_seq_to_patch(
                    self.patch_size,
                    tactile_pred.reshape(B * S, F_lat * H_lat * W_lat, C),
                    F_lat, H_lat, W_lat,
                    batch_size=B * S,
                ).reshape(B, S, C, F_lat, H_lat, W_lat)
                # Keep MSE in pred.dtype (bf16) so tactile_proj_out's grad dtype
                # stays uniform under FSDP, then promote the scalar to fp32.
                tactile_loss = torch.nn.functional.mse_loss(
                    pred_dense, target.to(pred_dense.dtype)).float()
            else:
                # CFG-drop / no-target step: keep tactile_proj_out in autograd
                # graph with zero contribution so FSDP grad dtype stays uniform.
                tactile_loss = (tactile_pred.sum() * 0.0).float()

        weight = float(getattr(self.config, 'tactile_diffusion_loss_weight', 1.0))
        total_loss = latent_loss + action_loss + weight * tactile_loss
        # Divide before backward so accumulated micro-batches produce the same
        # gradient scale as a single large batch.
        scale = self.gradient_accumulation_steps
        return {
            'latent_loss': latent_loss / scale,
            'action_loss': action_loss / scale,
            'tactile_loss': tactile_loss / scale,
            'total_loss': total_loss / scale,
        }

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        # batch_idx is the micro-step index inside the accumulation window.
        # FSDP gradient sync is skipped until the last micro-step.
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0

        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        output = self.transformer(input_dict, train_mode=True)
        loss_dict = self.compute_loss(input_dict, output)
        # total_loss is already divided by gradient_accumulation_steps.
        loss_dict['total_loss'].backward()

        losses = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in loss_dict.items()
        }
        
        # Only update weights after accumulating gradients
        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    @torch.no_grad()
    def _validate(self):
        """Loss-only pass over the held-out set.

        Randomness (timesteps, noisy-cond, chunk/window sizes, CFG drops) is
        re-seeded identically every call via fork_rng, so successive val losses
        are comparable rather than dominated by diffusion-timestep noise.
        """
        metrics = {'latent_loss': [], 'action_loss': [], 'tactile_loss': [], 'total_loss': []}
        with torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(1234 + self.config.rank)
            torch.cuda.manual_seed_all(1234 + self.config.rank)
            for batch in self.val_loader:
                batch = self.convert_input_format(batch)
                input_dict = self._prepare_input_dict(batch)
                output = self.transformer(input_dict, train_mode=True)
                loss_dict = self.compute_loss(input_dict, output)
                for name in metrics:
                    if name in loss_dict:
                        # undo the grad-accum division for honest per-sample loss
                        metrics[name].append(
                            loss_dict[name].detach() * self.gradient_accumulation_steps)
        out = {}
        for name, vals in metrics.items():
            local = torch.stack(vals).mean() if vals else torch.zeros((), device=self.device)
            out[name] = dist_mean(local).item()
        return out

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            # optim_state = get_optimizer_state_dict(
            #         self.transformer, self.optimizer,
            #         options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            #     )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # train/serve consistency snapshot (cross-checked at serve startup)
                _c = self.config
                meta = {
                    'step': self.step,
                    'norm_stat': {k: list(v) for k, v in dict(_c.norm_stat).items()
                                  if isinstance(v, (list, tuple))},
                    'norm_stat_path': getattr(_c, 'norm_stat_path', None),
                    'action_norm_method': getattr(_c, 'action_norm_method', None),
                    'action_delta_mode': getattr(_c, 'action_delta_mode', None),
                    'action_dim': int(getattr(_c, 'action_dim', 0)),
                    'pi05_action_horizon': int(getattr(_c, 'pi05_action_horizon', 0)),
                    'action_per_frame': int(getattr(_c, 'action_per_frame', 0)),
                    'used_action_channel_ids': [int(v) for v in getattr(_c, 'used_action_channel_ids', [])],
                    'use_local_tactile': bool(getattr(_c, 'use_local_tactile', False)),
                    'local_tactile_mode': getattr(_c, 'local_tactile_mode', None),
                    'tactile_global_zero': bool(getattr(_c, 'tactile_global_zero', False)),
                    'obs_cam_keys': list(getattr(_c, 'obs_cam_keys', [])),
                    'tactile_keys': list(getattr(_c, 'tactile_keys', [])),
                    'eval_prompt': getattr(_c, 'eval_prompt', None),
                }
                with open(checkpoint_dir / "train_meta.json", 'w') as f:
                    json.dump(meta, f, indent=2)

                # # Save optimizer state and training metadata in PyTorch format
                # training_state_path = checkpoint_dir / "training_state.pt"
                # logger.info(f"Saving training state to {training_state_path}")
                # torch.save({
                #     'step': self.step,
                #     'optimizer_state_dict': optim_state,
                #     'config': vars(self.config),
                # }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()
            save_error = None

        except Exception as e:
            save_error = e
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            # Ensure all processes stay synchronized even on error
            if dist.is_initialized():
                dist.barrier()

        # A failed save must stop training on ALL ranks in sync — continuing
        # produces a "successful" run with no usable weights.
        _failed = torch.tensor(
            [0 if save_error is None else 1],
            device=self.device if torch.cuda.is_available() else 'cpu')
        if dist.is_initialized():
            dist.all_reduce(_failed, op=dist.ReduceOp.MAX)
        if int(_failed.item()):
            raise RuntimeError(
                f'checkpoint save failed at step {self.step} — refusing to '
                'continue training with unsaved weights') from save_error

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        # optimizer-steps per epoch (len() covers both the bucketed batch_sampler
        # and the plain DistributedSampler paths) -> show epoch progress alongside
        # the step-based loop.
        try:
            self.steps_per_epoch = max(
                1, len(self.train_loader) // max(1, self.gradient_accumulation_steps))
        except Exception:
            self.steps_per_epoch = 0
        _ep = (f"{self.config.num_steps / self.steps_per_epoch:.2f}"
               if self.steps_per_epoch else "?")
        logger.info(f"Starting training for {self.config.num_steps} steps "
                    f"(~{self.steps_per_epoch} optim-steps/epoch -> ~{_ep} epochs)...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        metric_names = [
            'latent_loss',
            'action_loss',
            'total_loss',
            'tactile_loss',     # symdiff; always tracked (0 if unused)
        ]
        accumulated_metrics = {name: [] for name in metric_names}
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()

            try:
                losses = self._train_step(batch, step_in_accumulation)
            except Exception as e:
                import traceback
                rank = self.config.rank
                err_msg = f"\n{'='*60}\n[RANK {rank}] CRASH at step {self.step}, iter {step_in_accumulation}\n{traceback.format_exc()}{'='*60}\n"
                logger.error(err_msg)
                # Also write to a file so we can find it
                with open(f"crash_rank{rank}.log", "w") as f:
                    f.write(err_msg)
                raise
            
            # Accumulate losses for logging
            for name in metric_names:
                if name in losses:
                    accumulated_metrics[name].append(losses[name])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                metric_shows = {}
                max_metric_shows = {}
                for name, values in accumulated_metrics.items():
                    if not values:
                        continue
                    metric_tensor = torch.stack(values).sum()
                    metric_shows[name] = dist_mean(metric_tensor).detach().cpu().item()
                    max_metric_shows[name] = dist_max(metric_tensor).detach().cpu().item()

                accumulated_metrics = {name: [] for name in metric_names}
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_postfix = {
                        'latent_loss':  f'{metric_shows.get("latent_loss", 0.0):.4f}',
                        'action_loss':  f'{metric_shows.get("action_loss", 0.0):.4f}',
                        'tactile_loss': f'{metric_shows.get("tactile_loss", 0.0):.4f}',
                        'step': self.step,
                        'epoch': (f'{self.step / self.steps_per_epoch:.2f}'
                                  if self.steps_per_epoch else '?'),
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    }
                    progress_bar.set_postfix(progress_postfix)
                    if self.config.enable_wandb:
                        wandb_log = {
                            'loss_metrics/global_avg_video_loss':   metric_shows.get('latent_loss', 0.0),
                            'loss_metrics/global_avg_action_loss':  metric_shows.get('action_loss', 0.0),
                            'loss_metrics/global_avg_tactile_loss': metric_shows.get('tactile_loss', 0.0),
                            'loss_metrics/global_max_video_loss':   max_metric_shows.get('latent_loss', 0.0),
                            'loss_metrics/global_max_action_loss':  max_metric_shows.get('action_loss', 0.0),
                            'loss_metrics/global_max_tactile_loss': max_metric_shows.get('tactile_loss', 0.0),
                            'loss_metrics/global_avg_total_loss':   metric_shows.get('total_loss', 0.0),
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                            'epoch': (self.step / self.steps_per_epoch
                                      if self.steps_per_epoch else 0),
                        }
                        self.wandb.log(wandb_log, step=self.step)
                
                self.step += 1

                if self.val_loader is not None and self.step % self.val_interval == 0:
                    val_metrics = self._validate()
                    if self.config.rank == 0:
                        logger.info(
                            f"[val @ step {self.step}] " + " ".join(
                                f"{k}={v:.4f}" for k, v in val_metrics.items()))
                        if self.config.enable_wandb:
                            self.wandb.log({f'val_metrics/{k}': v
                                            for k, v in val_metrics.items()},
                                           step=self.step)

                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = TWAM_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='posttrain',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
