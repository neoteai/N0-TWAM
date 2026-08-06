# Copyright 2025-2026 NeoteAI Team. All rights reserved.
import argparse
import os
import sys
import time
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import TWAM_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from models.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class TWAM_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.frame_st_id = 0  # defensive init: avoid AttributeError if _infer before _reset (WS reconnect bug)
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.tactile_global_vae = WanVAEStreamingWrapper(self.vae)
        self.tactile_local_vae = WanVAEStreamingWrapper(self.vae)
        self.tactile_first_frames = None
        self.tactile_prev_frames = None
        # [tactile-pred-eval] last chunk's GENERATED GlobalTactile latent (+ the
        # frame_st_id it predicted for), so the NEXT compute_kv_cache (which carries
        # the REAL tactile observed AFTER executing this chunk's actions) can report
        # |predicted_future_tactile - real_observed_tactile| — the actual measure of
        # how good the tactile prediction is. None until the first generated chunk.
        self._last_gen_tactile = None
        self._last_gen_tactile_fsid = None

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        _tpath = os.path.join(job_config.wan22_pretrained_model_name_or_path, 'transformer')
        _is_mot = False
        try:
            import json as _json
            _is_mot = _json.load(open(os.path.join(_tpath, 'config.json'))).get('is_mot', False)
        except Exception:
            _is_mot = False
        if _is_mot:
            from models.utils import load_mot_checkpoint
            self.transformer = load_mot_checkpoint(
                _tpath, torch_dtype=self.dtype, torch_device=self.device, attn_mode='torch')
        else:
            self.transformer = load_transformer(
                _tpath,
                torch_dtype=self.dtype,
                torch_device=self.device,
                max_tactile_streams=job_config.max_tactile_streams,
                target_action_dim=int(getattr(job_config, 'action_dim', 30)),
                attn_mode=getattr(job_config, 'attn_mode', 'flashattn'),
            )
        logger.info('loaded transformer: %s (is_mot=%s) from %s',
                    type(self.transformer).__name__, _is_mot, _tpath)
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        self._check_train_serve_consistency()

    def _check_train_serve_consistency(self):
        """Refuse placeholder norm stats, and cross-check the checkpoint's
        train_meta.json (when present) against the live serve config.

        Multi-task checkpoints (``serve_task`` set, see multitask_server): the
        snapshot's norm_stat is the pool-level fallback envelope no task
        actually trained with, so the live norm is checked against the pool's
        per-task table instead, and ``used_action_channel_ids`` — the training
        union vs the served task's subset — is checked for containment."""
        import json
        from pathlib import Path

        serve_task = getattr(self.job_config, 'serve_task', None)

        ns = getattr(self.job_config, 'norm_stat', None) or {}
        q01 = [float(v) for v in ns.get('q01', [])]
        q99 = [float(v) for v in ns.get('q99', [])]
        if q01 and q01 == [-1.0] * len(q01) and q99 == [1.0] * len(q99):
            raise RuntimeError(
                'norm_stat is the [-1, 1] placeholder — the stats file was '
                f'missing when the config was imported '
                f'(norm_stat_path={getattr(self.job_config, "norm_stat_path", None)!r}). '
                'Compute the norm stats and point the config at them before serving.')

        problems = []
        if serve_task:
            # per-task norm source of truth: the pool table, not the snapshot.
            per_path = Path(getattr(self.job_config, 'multitask_norm_path', ''))
            if not per_path.is_file():
                raise RuntimeError(
                    f'[consistency] serve_task={serve_task!r} but the per-task '
                    f'norm table is missing: {per_path}')
            per = json.loads(per_path.read_text()).get(serve_task)
            if per is None:
                raise RuntimeError(
                    f'[consistency] task {serve_task!r} not in {per_path}')
            for key in ('q01', 'q99'):
                want = np.asarray(per.get(key, []), dtype=np.float64)
                live = np.asarray(ns.get(key, []), dtype=np.float64)
                if want.shape != live.shape or not np.allclose(want, live, atol=1e-6):
                    problems.append(
                        f'norm_stat.{key} differs from the per-task table '
                        f'({per_path.name}[{serve_task}])')

        meta_path = (Path(self.job_config.wan22_pretrained_model_name_or_path)
                     / 'transformer').resolve().parent / 'train_meta.json'
        if not meta_path.is_file():
            if problems:
                raise RuntimeError(
                    '[consistency] serve config does not match the per-task '
                    'norm table:\n  ' + '\n  '.join(problems))
            logger.info('[consistency] no train_meta.json next to the checkpoint '
                        '(%s) — skipping the training-snapshot cross-check', meta_path)
            if serve_task:
                logger.info('[consistency] multi-task per-task norm check passed '
                            '(task=%s)', serve_task)
            return
        meta = json.loads(meta_path.read_text())
        if not serve_task:
            meta_ns = meta.get('norm_stat') or {}
            for key in ('q01', 'q99'):
                trained = np.asarray(meta_ns.get(key, []), dtype=np.float64)
                live = np.asarray(ns.get(key, []), dtype=np.float64)
                if trained.shape != live.shape or not np.allclose(trained, live, atol=1e-6):
                    problems.append(f'norm_stat.{key} differs from training')
        for key in ('action_norm_method', 'action_delta_mode', 'action_dim',
                    'action_per_frame', 'pi05_action_horizon',
                    'used_action_channel_ids', 'use_local_tactile',
                    'local_tactile_mode', 'tactile_global_zero'):
            if key not in meta:
                continue
            live = getattr(self.job_config, key, None)
            if isinstance(meta[key], bool):
                live = bool(live)
            elif isinstance(meta[key], list):
                live = [int(v) for v in (live or [])]
                meta[key] = [int(v) for v in meta[key]]
            if serve_task and key == 'used_action_channel_ids':
                # training records the union over all tasks; a served task uses
                # its own subset (e.g. 10 of 20 for a single-arm task).
                if not set(live) <= set(meta[key]):
                    problems.append(
                        f'{key}: serve {live!r} is not a subset of train {meta[key]!r}')
                continue
            if meta[key] != live:
                problems.append(f'{key}: train={meta[key]!r} serve={live!r}')
        if problems:
            raise RuntimeError(
                '[consistency] serve config does not match the training '
                f'snapshot {meta_path}:\n  ' + '\n  '.join(problems))
        logger.info('[consistency] train_meta.json cross-check passed (%s%s)',
                    meta_path,
                    f'; multi-task norm checked per-task ({serve_task})'
                    if serve_task else '')

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def _uses_pi05_delta_actions(self):
        action_delta_mode = str(getattr(self.job_config, 'action_delta_mode', '')).lower()
        return action_delta_mode in {'pi05_delta', 'openpi_delta', 'pi0.5_delta'}

    def _pi05_delta_channel_ids(self):
        action_dim = int(self.job_config.action_dim)
        delta_ids = getattr(self.job_config, 'pi05_delta_channel_ids',
                            list(range(0, 9)) + list(range(10, 19)))
        return [int(v) for v in delta_ids if 0 <= int(v) < action_dim]

    def _pi05_required_state_dim(self):
        delta_ids = self._pi05_delta_channel_ids()
        return max(delta_ids) + 1 if delta_ids else 1

    def _pad_state_vector(self, state_vec, field_name):
        state_vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)
        required_dim = self._pi05_required_state_dim()
        action_dim = int(self.job_config.action_dim)
        if state_vec.shape[0] < required_dim:
            raise ValueError(
                f"{field_name} must contain at least {required_dim} dims for "
                f"pi05_delta channels {self._pi05_delta_channel_ids()}, got "
                f"shape {state_vec.shape}."
            )
        if state_vec.shape[0] < action_dim:
            state_vec = np.pad(state_vec, (0, action_dim - state_vec.shape[0]))
        return state_vec[:action_dim].astype(np.float32, copy=False)

    def _extract_current_state_vector(self, state, field_name='current_state'):
        if state is None:
            raise ValueError(
                f"pi05_delta requires obs['{field_name}'] to convert between "
                "model deltas and absolute action targets."
            )
        state_arr = np.asarray(state, dtype=np.float32)
        action_dim = int(self.job_config.action_dim)
        required_dim = self._pi05_required_state_dim()
        if state_arr.ndim == 1:
            state_vec = state_arr
        elif state_arr.ndim == 2:
            if required_dim <= state_arr.shape[0] <= action_dim:
                state_vec = state_arr[:, -1]
            elif required_dim <= state_arr.shape[1] <= action_dim:
                state_vec = state_arr[-1, :]
            elif state_arr.shape[0] >= action_dim:
                state_vec = state_arr[:action_dim, -1]
            elif state_arr.shape[1] >= action_dim:
                state_vec = state_arr[-1, :action_dim]
            else:
                raise ValueError(
                    f"Unsupported {field_name} shape for pi05_delta: "
                    f"{state_arr.shape}"
                )
        elif state_arr.ndim >= 3:
            if required_dim <= state_arr.shape[0] <= action_dim:
                # Channel-first [C, F, H]. Use newest frame and first horizon slot.
                state_vec = state_arr[:, -1, 0]
            elif state_arr.shape[0] >= action_dim:
                state_vec = state_arr[:action_dim, -1, 0]
            elif state_arr.shape[-1] >= required_dim:
                state_vec = state_arr.reshape(-1, state_arr.shape[-1])[-1]
            else:
                raise ValueError(
                    f"Unsupported {field_name} shape for pi05_delta: "
                    f"{state_arr.shape}"
                )
        else:
            raise ValueError(
                f"Unsupported {field_name} shape for pi05_delta: {state_arr.shape}"
            )
        return self._pad_state_vector(state_vec, field_name)

    def _canonical_action_chunk(self, action, field_name='state'):
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim != 3:
            raise ValueError(
                f"{field_name} must have shape [C, F, H], got {action_arr.shape}"
            )
        action_dim = int(self.job_config.action_dim)
        if action_arr.shape[0] == action_dim:
            return action_arr.astype(np.float32, copy=True)
        if action_arr.shape[0] > action_dim:
            raise ValueError(
                f"{field_name} has {action_arr.shape[0]} channels, expected "
                f"<= action_dim={action_dim}."
            )

        padded = np.zeros(
            (action_dim, action_arr.shape[1], action_arr.shape[2]),
            dtype=np.float32,
        )
        used_ids = list(getattr(self.job_config, 'used_action_channel_ids', []))
        if len(used_ids) == action_arr.shape[0]:
            for src_i, dst_i in enumerate(used_ids):
                dst_i = int(dst_i)
                if 0 <= dst_i < action_dim:
                    padded[dst_i] = action_arr[src_i]
        else:
            padded[:action_arr.shape[0]] = action_arr
        return padded

    def preprocess_action(self, action, action_anchor_state=None, action_format=None,
                          cold_first_frame=False):
        action_model_input_np = self._canonical_action_chunk(action)
        if self._uses_pi05_delta_actions():
            action_format = str(action_format or 'absolute').lower()
            if action_format in {'absolute', 'absolute_target', 'target'}:
                anchor_state = self._extract_current_state_vector(
                    action_anchor_state,
                    field_name='action_anchor_state',
                )
                # Symmetric to postprocess_action's per-frame anchoring: recover the
                # per-frame raw deltas the model was trained on (and that the KV cache
                # expects) by subtracting each frame's own anchor (frame 0 -> supplied
                # anchor_state, frame f>0 -> frame f-1's last absolute target). Capture
                # the anchors from the ORIGINAL absolute values before subtracting, so
                # the grounding round-trip still cancels and the KV cache stays correct.
                for dim in self._pi05_delta_channel_ids():
                    frame_anchors = [float(anchor_state[dim])]
                    for f in range(1, action_model_input_np.shape[1]):
                        frame_anchors.append(float(action_model_input_np[dim, f - 1, -1]))
                    for f in range(action_model_input_np.shape[1]):
                        action_model_input_np[dim, f] -= frame_anchors[f]
            elif action_format in {'pi05_delta', 'openpi_delta', 'pi0.5_delta', 'delta'}:
                pass
            else:
                raise ValueError(
                    f"Unsupported state_action_format for pi05_delta: "
                    f"{action_format!r}"
                )

        action_model_input = torch.from_numpy(action_model_input_np)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if str(self.action_norm_method).lower() in {'quantiles', 'q01q99'}:
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError(
                f"Unsupported action_norm_method: {self.action_norm_method!r}")
        # cold-chunk grounding: training masks the whole
        # frame0 action token to zeros in NORMALIZED space
        # (pi05_condition_first_frame_zero). With the postprocess fix, frame0 now
        # round-trips as raw-zero deltas, which normalize to (0-q01)/(q99-q01)*2-1
        # != 0 for asymmetric quantiles — zero it here or the KV-cache history
        # token drifts off the training distribution.
        if (cold_first_frame and self._uses_pi05_delta_actions()
                and action_model_input.shape[1] > 0):
            action_model_input[:, 0, :] = 0.
            logger.info('[cold-anchor-fix] preprocess: zeroed normalized frame0 '
                        'token for cold-chunk KV grounding')
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def _tactile_image_size(self):
        tactile_resize = int(getattr(self.job_config, 'tactile_resize', 0) or 0)
        if tactile_resize > 0:
            return tactile_resize, tactile_resize
        return (
            int(getattr(self.job_config, 'tactile_height', 64)),
            int(getattr(self.job_config, 'tactile_width', 64)),
        )

    def _preprocess_single_tactile_frame(self, frame):
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = frame[..., None]
        if frame.shape[-1] not in (1, 3):
            raise ValueError(f"Unsupported tactile frame shape: {frame.shape}")
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1)
        if frame_tensor.shape[0] == 1:
            frame_tensor = frame_tensor.repeat(3, 1, 1)
        tactile_height, tactile_width = self._tactile_image_size()
        frame_tensor = F.interpolate(
            frame_tensor.unsqueeze(0),
            size=(tactile_height, tactile_width),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        return frame_tensor

    def _build_tactile_tensor(self, obs):
        tactile = obs.get('tactile')
        if tactile is None:
            if getattr(self.job_config, 'synthetic_tactile_data', False):
                tactile_height, tactile_width = self._tactile_image_size()
                n_streams = len(self.job_config.tactile_keys)
                # match the number of frames the client sent for VIDEO (obs['obs']),
                # so the synthetic black tactile is frame-aligned and the streaming VAE
                # gets the same temporal length as video (a single frame on a >=3-frame
                # compute_kv_cache grounding would crash WAN's avg_shortcut conv).
                _vid = obs.get('obs')
                n_frames = len(_vid) if isinstance(_vid, list) and len(_vid) >= 1 else 1
                logger.info("Using synthetic zero-valued tactile (synthetic_tactile_data=True, F=%d)", n_frames)
                return torch.zeros(
                    n_streams,
                    3,
                    n_frames,
                    tactile_height,
                    tactile_width,
                    dtype=torch.float32,
                )
            return None

        tactile_history = tactile if isinstance(tactile, list) else [tactile]
        tactile_streams = []
        for key in self.job_config.tactile_keys:
            frames = []
            for tactile_frame_dict in tactile_history:
                if key not in tactile_frame_dict:
                    logger.warning("Missing tactile key %s; dropping tactile condition", key)
                    return None
                frames.append(self._preprocess_single_tactile_frame(tactile_frame_dict[key]))
            tactile_streams.append(torch.stack(frames, dim=1))

        if not tactile_streams:
            return None
        return torch.stack(tactile_streams, dim=0).contiguous()

    def _encode_tactile_residual_latent(self, residual, streaming_vae):
        vae_device = next(streaming_vae.vae.parameters()).device
        residual = residual.to(device=vae_device, dtype=self.dtype)
        enc_out = streaming_vae.encode_chunk(residual)
        mu, _ = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        return mu_norm.unsqueeze(0).to(self.device, dtype=self.dtype)

    def _encode_tactile_obs(self, obs):
        if not self.job_config.tactile_keys:
            raise ValueError(
                "tactile cond server requires non-empty job_config.tactile_keys."
            )

        # Tactile mirrors video: the persistent streaming VAE is advanced ONLY by
        # compute_kv_cache (grounding) and by the frame_st_id==0 cold seed in _infer —
        # never by a mid-episode plain-infer. So no per-call clear: the cache is cold
        # only right after reset (the 1-frame seed) and warm for every kv_cache chunk,
        # exactly like streaming_vae for video.
        tactile_tensor = self._build_tactile_tensor(obs)
        if tactile_tensor is None:
            raise ValueError(
                "tactile cond server requires obs['tactile'] with all configured "
                f"tactile keys: {self.job_config.tactile_keys}"
            )

        scale = 255.0 if float(tactile_tensor.max()) > 1.5 else 1.0
        first_frames = self.tactile_first_frames
        prev_frames = self.tactile_prev_frames
        if first_frames is None:
            first_frames = tactile_tensor[:, :, 0].clone()
        if prev_frames is None:
            prev_frames = tactile_tensor[:, :, 0].clone()

        global_residual = (tactile_tensor - first_frames[:, :, None]) / scale

        # LocalTactile: 'residual' (legacy) = frame[t]-frame[t-1]; 'current' = the
        # current frame mapped to [-1,1] (no delta), matching encode_tactile_latent's
        # --local-mode current. Must match how the loaded ckpt was trained.
        local_mode = str(getattr(self.job_config, 'local_tactile_mode', 'current'))
        if local_mode == 'current':
            # (t/255)*2 - 1  ==  encode script's frames_u8/127.5 - 1  (tactile_tensor is [0,255])
            local_residual = (tactile_tensor / scale) * 2.0 - 1.0
        else:
            local_residual = torch.empty_like(tactile_tensor)
            local_residual[:, :, 0] = (tactile_tensor[:, :, 0] - prev_frames) / scale
            if tactile_tensor.shape[2] > 1:
                local_residual[:, :, 1:] = (
                    tactile_tensor[:, :, 1:] - tactile_tensor[:, :, :-1]
                ) / scale

        global_latent = self._encode_tactile_residual_latent(
            global_residual, self.tactile_global_vae)
        local_latent = self._encode_tactile_residual_latent(
            local_residual, self.tactile_local_vae)

        # global-ablation serve mode: zero ONLY GlobalTactile everywhere (clean
        # cond, denoise target, kv grounding); LocalTactile stays real.
        if bool(getattr(self.job_config, 'tactile_global_zero', False)):
            global_latent = torch.zeros_like(global_latent)

        sensor_id_map = getattr(self.job_config, 'tactile_sensor_id_map', {}) or {}
        sensor_ids = torch.tensor(
            [int(sensor_id_map.get(key, idx)) for idx, key in enumerate(self.job_config.tactile_keys)],
            dtype=torch.long,
            device=self.device,
        )[None]

        self.tactile_first_frames = first_frames.detach()
        self.tactile_prev_frames = tactile_tensor[:, :, -1].detach()
        logger.info(
            "encoded tactile latents: global=%s local=%s sensor_ids=%s",
            tuple(global_latent.shape),
            tuple(local_latent.shape),
            sensor_ids.detach().cpu().tolist(),
        )
        return {
            'tactile_global_latent': global_latent,
            'tactile_local_latent': local_latent,
            'tactile_sensor_ids': sensor_ids,
        }

    def _reset_tactile_state(self):
        self.tactile_global_vae.clear_cache()
        self.tactile_local_vae.clear_cache()
        self.tactile_first_frames = None
        self.tactile_prev_frames = None
        self._last_gen_tactile = None          # [tactile-pred-eval] don't cross episodes
        self._last_gen_tactile_fsid = None

    def postprocess_action(self, action, current_state=None, output_format=None,
                           cold_first_frame=False):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  # C, F, H
        if str(self.action_norm_method).lower() in {'quantiles', 'q01q99'}:
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError(
                f"Unsupported action_norm_method: {self.action_norm_method!r}")
        action = action.squeeze(0).detach().cpu().numpy()
        active_ids = [int(v) for v in getattr(self.job_config, 'used_action_channel_ids', [])]
        inactive_ids = [i for i in range(action.shape[0]) if i not in active_ids]
        if inactive_ids:
            action[inactive_ids] = 0.0

        if self._uses_pi05_delta_actions():
            # cold chunk (frame_st_id==0): frame0 is the
            # zero-clamped condition slot, not a prediction. De-normalized it lands
            # on the q01/q99 midpoint (a small constant offset), and the
            # sequential anchoring below carries that constant bias into EVERY
            # frame of the chunk. Zero the frame0 deltas so frame0 == current_state
            # exactly and frame1 re-anchors to the real current state.
            if cold_first_frame and action.shape[1] > 0:
                for dim in self._pi05_delta_channel_ids():
                    action[dim, 0, :] = 0.0
                logger.info('[cold-anchor-fix] postprocess: zeroed frame0 deltas '
                            '(cold chunk, frame1 re-anchors to current_state)')
            # [delta-smooth] opt-in: ramp the first delta_smooth_k actions of a
            # WARM chunk into the previous chunk's last delta (cold chunks reset).
            if bool(getattr(self.job_config, 'delta_smooth', False)):
                dims = list(self._pi05_delta_channel_ids())
                if cold_first_frame:
                    self._delta_smooth_prev = None
                prev = getattr(self, '_delta_smooth_prev', None)
                _, n_frames, n_slots = action.shape
                seq = action[dims].reshape(len(dims), -1).copy()
                if prev is not None and seq.shape[1] > 0:
                    ramp_k = max(1, int(getattr(self.job_config, 'delta_smooth_k', 3)))
                    for i in range(min(ramp_k, seq.shape[1])):
                        w = (i + 1.0) / (ramp_k + 1.0)
                        seq[:, i] = w * seq[:, i] + (1.0 - w) * prev
                    action[dims] = seq.reshape(len(dims), n_frames, n_slots)
                    logger.info('[delta-smooth] ramped first %d actions into prev motion',
                                min(ramp_k, seq.shape[1]))
                self._delta_smooth_prev = seq[:, -1].copy() if seq.shape[1] > 0 else prev
            output_format = str(
                output_format
                or getattr(self.job_config, 'server_action_output_format', 'absolute')
            ).lower()
            if output_format in {'absolute', 'absolute_target', 'target'}:
                state_vec = self._extract_current_state_vector(
                    current_state,
                    field_name='current_state',
                )
                # Per-frame anchoring (fix intra-chunk retreat seam): each predicted
                # frame's deltas were trained relative to that frame's OWN anchor
                # state, and consecutive frame anchors are action_per_frame steps
                # apart (see _build_pi05_delta_actions in
                # lerobot_latent_dataset_pi05_delta.py). Reconstructing every frame against one shared current_state
                # short-changed every frame after the first by ~one frame of motion,
                # producing a backward step at each frame0->frame1 seam. Instead
                # reconstruct sequentially: frame f anchors to frame f-1's last
                # absolute target so the executed trajectory stays continuous.
                for dim in self._pi05_delta_channel_ids():
                    anchor = float(state_vec[dim])
                    for f in range(action.shape[1]):
                        action[dim, f] += anchor
                        anchor = float(action[dim, f, -1])
            elif output_format in {'pi05_delta', 'openpi_delta', 'pi0.5_delta', 'delta'}:
                pass
            else:
                raise ValueError(
                    f"Unsupported server_action_output_format for pi05_delta: "
                    f"{output_format!r}"
                )
        return_ids = getattr(
            self.job_config,
            'server_return_action_channel_ids',
            self.job_config.used_action_channel_ids,
        )
        return action[[int(v) for v in return_ids]]
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
            if 'tactile_global_latent' in input_dict:
                input_dict['tactile_global_latent'] = input_dict['tactile_global_latent'].repeat(2, 1, 1, 1, 1, 1)
            if 'tactile_local_latent' in input_dict:
                input_dict['tactile_local_latent'] = input_dict['tactile_local_latent'].repeat(2, 1, 1, 1, 1, 1)
            if 'tactile_sensor_ids' in input_dict:
                input_dict['tactile_sensor_ids'] = input_dict['tactile_sensor_ids'].repeat(2, 1)
            # tactile-denoise (video loop) injects these AFTER the other keys; must
            # also be CFG-doubled or the cond/uncond batch sizes mismatch -> crash.
            if 'tactile_noisy_latent' in input_dict:
                input_dict['tactile_noisy_latent'] = input_dict['tactile_noisy_latent'].repeat(2, 1, 1, 1, 1, 1)
            if 'tactile_timesteps' in input_dict:
                reps = [2] + [1] * (input_dict['tactile_timesteps'].dim() - 1)
                input_dict['tactile_timesteps'] = input_dict['tactile_timesteps'].repeat(*reps)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2),
                              tactile_latents=None):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0
            if tactile_latents is not None:
                input_dict['latent_res_lst']['tactile_global_latent'] = (
                    tactile_latents['tactile_global_latent']
                )
                input_dict['latent_res_lst']['tactile_sensor_ids'] = (
                    tactile_latents['tactile_sensor_ids']
                )

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if tactile_latents is not None:
                input_dict['action_res_lst'].update(tactile_latents)

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        videos = []
        for k in self.job_config.obs_cam_keys:
            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(self.height, self.width),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
        vae_device = next(self.streaming_vae.vae.parameters()).device
        videos_chunk = videos.to(vae_device).to(self.dtype)
        enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def _reset(self, prompt=None):
        logger.info('Reset.')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        self.last_tactile_latents = None   # mirror init_latent: tactile cond reuse slot
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        self._reset_tactile_state()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
            self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        if self.job_config.tactile_keys:
            tactile_latent_height = int(getattr(self.job_config, 'tactile_latent_height', 8))
            tactile_latent_width = int(getattr(self.job_config, 'tactile_latent_width', 8))
            tactile_token_per_chunk = (
                len(self.job_config.tactile_keys) *
                self.job_config.frame_chunk_size *
                tactile_latent_height *
                tactile_latent_width
            ) // (patch_size[0] * patch_size[1] * patch_size[2])
            latent_token_per_chunk += tactile_token_per_chunk
            action_token_per_chunk += tactile_token_per_chunk
            logger.info("tactile cache tokens per chunk: %d", tactile_token_per_chunk)
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt (bare reset falls back to the config prompt)
        if prompt is None:
            prompt = getattr(self.job_config, 'prompt', None)
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _tactile_pred_to_latent(self, tactile_pred, tactile_g):
        """Un-patchify the model's tactile velocity prediction (patch-sequence
        (B', S*F*spatial, 48)) back to the GlobalTactile latent layout
        (1, S, 48, F, H, W), matching tactile_g — mirrors train.py's tactile loss
        un-patchify (data_seq_to_patch). Drops the CFG duplicate if present."""
        _, S, C, F_lat, H_lat, W_lat = tactile_g.shape
        # CFG: model was fed a 2x-repeated batch -> take the conditional (first) half.
        if tactile_pred.shape[0] > 1:
            if self.job_config.guidance_scale > 1:
                cond = tactile_pred[:1]
                uncond = tactile_pred[1:2]
                tactile_pred = uncond + self.job_config.guidance_scale * (cond - uncond)
            else:
                tactile_pred = tactile_pred[:1]
        pred_dense = data_seq_to_patch(
            self.job_config.patch_size,
            tactile_pred.reshape(S, F_lat * H_lat * W_lat, C),
            F_lat, H_lat, W_lat,
            batch_size=S,
        ).reshape(1, S, C, F_lat, H_lat, W_lat)
        return pred_dense.to(tactile_g.dtype)

    def _infer(self, obs, frame_st_id=0):
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            # Cold seed — mirror video's init_latent: encode the current tactile once
            # (advances the persistent tactile streaming VAE, just like _encode_obs does
            # for streaming_vae), then commit it so later plain-infers can reuse it.
            # The persistent tactile VAE feat_cache must be FRESH for this 1-frame cold
            # seed (WAN avg_shortcut needs Rep padding or kernel(3)>input crashes); the
            # warm-cache discipline only holds for the >=3-frame kv_cache groundings after.
            self.tactile_global_vae.clear_cache()
            self.tactile_local_vae.clear_cache()
            tactile_latents = (
                self._encode_tactile_obs(obs) if self.job_config.tactile_keys else None
            )
            self.last_tactile_latents = tactile_latents
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent
        else:
            # Mid-episode plain-infer: do NOT re-encode (that double-fed the streaming
            # VAE and corrupted the temporal grid). Reuse the tactile latent committed by
            # the last compute_kv_cache — video does the same here (no obs encode; it
            # reads the transformer KV cache).
            tactile_latents = self.last_tactile_latents

        # TACTILE DENOISE: co-generate GlobalTactile in the video loop (training-
        # consistent, default on); off = condition on the raw observed tactile.
        tactile_denoise = bool(getattr(self.job_config, 'server_tactile_denoise', False))
        tactile_g = None
        if tactile_denoise and tactile_latents is not None:
            # noisy GlobalTactile latent to denoise — same shape as the encoded
            # global latent (1, S, 48, F_lat, H_lat, W_lat).
            tactile_g = torch.randn_like(tactile_latents['tactile_global_latent'])

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop (co-generates GlobalTactile when enabled)
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id,
                    tactile_latents=tactile_latents)

                # inject the current noisy GlobalTactile so the model denoises it
                # alongside the video (same timestep t, frame-aligned co-generation).
                if tactile_g is not None:
                    input_dict['latent_res_lst']['tactile_noisy_latent'] = tactile_g
                    input_dict['latent_res_lst']['tactile_timesteps'] = (
                        torch.ones([tactile_g.shape[1] * tactile_g.shape[3]],
                                   device=self.device) * t)

                out = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)
                if isinstance(out, tuple):
                    video_noise_pred, tactile_noise_pred = out
                else:
                    video_noise_pred, tactile_noise_pred = out, None

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)
                    # step the GlobalTactile latent with the SAME scheduler (same
                    # snr_shift as video in training), drop CFG duplicate if present.
                    if tactile_g is not None and tactile_noise_pred is not None:
                        _pre = float(tactile_g.float().std())
                        tac_pred = self._tactile_pred_to_latent(
                            tactile_noise_pred, tactile_g)
                        tactile_g = self.scheduler.step(
                            tac_pred, t, tactile_g, return_dict=False)
                        logger.info("[tac-step] t=%s pre_std=%.4f post_std=%.4f pred_std=%.4f",
                                    int(t) if hasattr(t, '__int__') else t, _pre,
                                    float(tactile_g.float().std()), float(tac_pred.float().std()))

                if frame_st_id == 0:
                    latents[:, :, 0:1] = latent_cond

            # video loop done: if we co-generated GlobalTactile, the action loop below
            # conditions on the GENERATED tactile (not the observed residual) — the
            # "predict tactile -> act on predicted touch" flow. NOTE: this only affects
            # the in-chunk action pass; _compute_kv_cache re-encodes the OBSERVED
            # tactile for the cross-chunk cache (so the cache stays grounded in reality).
            if tactile_g is not None and tactile_latents is not None:
                # diagnostic proof the tactile was actually denoised (not pass-through):
                # gen_std near randn(1.0) = never denoised (BUG); small = denoised.
                logger.info(
                    "[tactile-denoise] generated GlobalTactile: gen_std=%.4f gen_mean=%.4f",
                    float(tactile_g.float().std()), float(tactile_g.float().mean()))
                # [tactile-pred-eval] stash this chunk's GENERATED future tactile so the
                # NEXT compute_kv_cache (real tactile observed AFTER executing the actions
                # of this chunk) can score the prediction. This is the correct, time-
                # shifted comparison: predicted_future vs real_observed_after_execution.
                self._last_gen_tactile = tactile_g.detach().clone()
                self._last_gen_tactile_fsid = int(frame_st_id)
                tactile_latents = dict(tactile_latents)
                tactile_latents['tactile_global_latent'] = tactile_g

            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                if frame_st_id != 0:
                    action_cond = None
                else:
                    # cold-seed frame0 mode. delta: zeros == "stay" (correct, forced).
                    # absolute (cfg.cold_seed_mode): free = do NOT clamp, let the model
                    # denoise frame0 (its learned cold-start; base loader trained frame0
                    # with loss ON, so this is the training-consistent mode — DEFAULT,
                    # ablation-validated); current_state = clean-clamp normalized current
                    # pose (semantically nice but never seen in training); zeros =
                    # original (de-normalizes to q01/q99 midpoint = OOD).
                    _cs_mode = ("zeros" if self._uses_pi05_delta_actions()
                                else str(getattr(self.job_config, 'cold_seed_mode', 'free')).lower())
                    _cs_val = obs.get("current_state")
                    if _cs_mode == "free":
                        action_cond = None
                    elif _cs_mode == "current_state" and _cs_val is not None:
                        _cs = np.asarray(_cs_val, dtype=np.float32).reshape(-1)
                        _adim = int(self.job_config.action_dim)
                        if _cs.shape[0] < _adim:
                            _cs = np.pad(_cs, (0, _adim - _cs.shape[0]))
                        _cs_chunk = np.repeat(_cs[:_adim].reshape(-1, 1, 1), self.action_per_frame, axis=2)
                        action_cond = self.preprocess_action(_cs_chunk).to(device=self.device, dtype=self.dtype)
                    else:
                        action_cond = torch.zeros(
                            [1, self.job_config.action_dim, 1, self.action_per_frame, 1],
                            device=self.device, dtype=self.dtype)

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id,
                    tactile_latents=tactile_latents)
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                if action_cond is not None:
                    actions[:, :, 0:1] = action_cond

        actions[:, ~self.action_mask] *= 0

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        current_state = obs.get('current_state')
        if current_state is None and 'state' in obs:
            current_state = obs['state']
        actions = self.postprocess_action(actions, current_state=current_state,
                                          cold_first_frame=(frame_st_id == 0))
        torch.cuda.empty_cache()
        return actions, latents

    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        action_anchor_state = obs.get('action_anchor_state', obs.get('current_state'))
        action_format = obs.get('state_action_format', obs.get('action_format'))
        action_model_input = self.preprocess_action(
            obs['state'],
            action_anchor_state=action_anchor_state,
            action_format=action_format,
            cold_first_frame=(self.frame_st_id == 0),
        )
        action_model_input = action_model_input.to(latent_model_input)
        tactile_latents = self._encode_tactile_obs(obs)
        # [tactile-pred-eval] tactile_latents['tactile_global_latent'] here IS the REAL
        # tactile observed AFTER executing the previous chunk's actions (fed back by the client). Score
        # last chunk's GENERATED future tactile against it — the true "did the tactile
        # prediction match what actually happened" metric. Diagnostic only (no effect on
        # cache/inference). real-tactile only meaningful; black-tactile -> both ~0.
        if self._last_gen_tactile is not None and tactile_latents is not None:
            _gen = self._last_gen_tactile.float()
            _real = tactile_latents['tactile_global_latent'].float()
            if _gen.shape == _real.shape:
                _err = (_gen - _real).abs().mean().item()
                _rstd = _real.std().item()
                logger.info(
                    "[tactile-pred-eval] predicted(fsid=%s) vs real-observed-after-exec: "
                    "|pred-real|mean=%.4f  pred_std=%.4f  real_std=%.4f  rel=%.3f",
                    self._last_gen_tactile_fsid, _err, _gen.std().item(), _rstd,
                    _err / (_rstd + 1e-6))
            else:
                logger.info("[tactile-pred-eval] shape mismatch pred=%s real=%s (skip)",
                            tuple(_gen.shape), tuple(_real.shape))
        # Commit so the next plain-infer (frame_st_id!=0) reuses this grounded latent
        # instead of re-encoding the current obs into the streaming VAE.
        self.last_tactile_latents = tactile_latents
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id,
                                                tactile_latents=tactile_latents)

        with (
                torch.no_grad(),
        ):
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)

        if reset:
            logger.info(f"******************* Reset server ******************")
            # deterministic sampling for reproducible eval: seed torch/np RNG per episode
            # with the client-supplied eval seed, so the same seed reproduces the same
            # diffusion noise every run (fair A/B of cold-seed modes).
            # cfg.deterministic_episode_seed=False disables.
            _seed = obs.get("seed", None)
            if _seed is not None and bool(getattr(self.job_config,
                                                  'deterministic_episode_seed', True)):
                _s = int(_seed) & 0x7fffffff
                torch.manual_seed(_s)
                torch.cuda.manual_seed_all(_s)
                np.random.seed(_s)
                logger.info(f"deterministic manual_seed({_s}) on reset")
            self._reset(prompt=prompt)
            return dict()
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        else:
            logger.info(f"################# Infer One Chunk #################")
            # Keep the streaming VAE temporal cache after frame 0 so later
            # compute_kv_cache calls can encode raw sub-keyframes as a
            # continuation of the previous chunk.
            if self.frame_st_id == 0:
                self.streaming_vae.clear_cache()
                self._reset_tactile_state()
            action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            # fsid discipline: imagination does NOT advance the time axis; only
            # grounding (_compute_kv_cache) does. Advancing on both double-counts
            # time and drifts KV RoPE off the training geometry.
            return dict(action=action)
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        del self.transformer
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):    
    
    config = TWAM_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = TWAM_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE i2va mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """Parse CLI args and launch the inference server."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='twam_server',
        help="config name (registered in TWAM_CONFIGS).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
