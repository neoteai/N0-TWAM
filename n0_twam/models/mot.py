# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Mixture-of-Transformers (MoT) backbone for the TWAM model.

Replaces the single shared `WanTransformerBlock` stack with one *expert* stack
per modality (video / action / tactile). At every layer each expert computes its
own q/k/v from its own weights (norm1 + AdaLN + q/k/v proj + RoPE), the q/k/v are
CONCATENATED across experts, a SINGLE shared self-attention runs over the union
(the "Multimodal Shared Attention" of FastWAM, with the existing joint
causal+noise mask), then the output is split back and each expert applies its own
output proj + gate + (optional) text cross-attention + FFN.

Design choices (see discussion / repo docs):
  * Token order in the concatenated sequence is the SAME as the legacy model:
    [video(noisy,clean) | action(noisy,clean) | tactile(noisy,clean) | pad].
    Each modality block is contiguous, so the legacy flex self-attn mask applies
    verbatim — the cascade ordering "predict visual+tactile, then action" is
    already encoded in that mask's frame-id causality and is preserved unchanged.
  * Experts share num_layers / num_heads / head_dim / hidden dim (required to
    concatenate q/k/v for the shared attention); per-expert `ffn_dim` may differ.
  * Tactile expert skips text cross-attention (legacy behaviour: the cross mask
    excluded tactile queries).

Correctness: the per-expert block split reproduces the legacy block.forward, and
the MoT with weights tied to a single stack matches that shared stack run over the
whole sequence (routing / concat / split are transparent).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt

from .model import WanTransformerBlock, FlexAttnFunc, custom_sdpa, WanTransformer3DModel


class SharedSelfAttention(nn.Module):
    """The one cross-expert attention per layer.

    Two interchangeable backends:
      * flex (default, GPU/train): a sparse `BlockMask` set via `set_block_mask`,
        identical to the legacy self-attention path.
      * dense (CPU / no-GPU tests): an explicit boolean mask [S, S] via
        `set_dense_mask`, routed through `custom_sdpa` (no triton/compile needed).
    When a dense mask is set it takes precedence; otherwise the flex op is used.
    """

    def __init__(self) -> None:
        super().__init__()
        self.flex = FlexAttnFunc(is_cross=False)
        self._dense_mask: Optional[torch.Tensor] = None
        self.attn_caches = {}   # streaming KV-cache pools per cache_name

    def set_block_mask(self, block_mask) -> None:
        self.flex.set_block_mask(block_mask)

    def set_dense_mask(self, dense_mask: Optional[torch.Tensor]) -> None:
        self._dense_mask = dense_mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                update_cache: int = 0, cache_name=None) -> torch.Tensor:
        # q/k/v: [B(=1), S, heads, head_dim]
        # Streaming KV-cache path: when a cache pool exists for cache_name,
        # append this chunk's K/V to the rolling window and attend q over all valid
        # (committed past + current) K/V via plain SDPA — temporal causality is
        # encoded by what is committed in the pool (mirrors WanAttention cache path).
        cache = self.attn_caches.get(cache_name) if cache_name is not None else None
        if cache is not None:
            slots = self.update_cache(cache_name, k, v, is_pred=(update_cache == 1))
            valid = cache['mask'].nonzero(as_tuple=False).squeeze(-1)
            k_w = cache['k'][:, valid]
            v_w = cache['v'][:, valid]
            out = custom_sdpa(q, k_w, v_w)
            if update_cache == 0:
                self.restore_cache(cache_name, slots)
            return out
        if self._dense_mask is not None:
            return custom_sdpa(q, k, v, attn_mask=self._dense_mask)
        if self.flex.block_mask is None:
            return custom_sdpa(q, k, v)
        return self.flex(q, k, v)

    # ───── streaming KV-cache (copied from WanAttention, operates on q/k/v heads) ─────
    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                      device, dtype, batch_size):
        self.attn_caches[cache_name] = {
            'k': torch.empty([batch_size, total_tolen, num_head, head_dim], device=device, dtype=dtype),
            'v': torch.empty([batch_size, total_tolen, num_head, head_dim], device=device, dtype=dtype),
            'id': torch.full((total_tolen,), -1, device=device),
            'mask': torch.zeros((total_tolen,), dtype=torch.bool, device=device),
            'is_pred': torch.zeros((total_tolen,), dtype=torch.bool, device=device),
        }

    def clear_cache(self, cache_name):
        self.attn_caches[cache_name] = None

    def clear_pred_cache(self, cache_name):
        c = self.attn_caches.get(cache_name)
        if c is None:
            return
        pred = c['is_pred'] & c['mask']
        c['mask'][pred] = False
        c['id'][pred] = -1
        c['is_pred'][pred] = False

    def _next_cache_id(self, cache_name):
        ids = self.attn_caches[cache_name]['id']
        mask = self.attn_caches[cache_name]['mask']
        if mask.any():
            return ids[mask].max() + 1
        return torch.tensor(0, device=ids.device)

    def allocate_slots(self, cache_name, key_size):
        cache = self.attn_caches[cache_name]
        mask = cache['mask']; ids = cache['id']
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)
        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)
            order = torch.argsort(ids[used])
            need = key_size - free.numel()
            to_free = used[order[:need]]
            mask[to_free] = False; ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)
        assert free.numel() >= key_size
        return free[:key_size]

    def update_cache(self, cache_name, key, value, is_pred):
        cache = self.attn_caches[cache_name]
        slots = self.allocate_slots(cache_name, key.shape[1])
        new_id = self._next_cache_id(cache_name)
        cache['k'][:, slots] = key
        cache['v'][:, slots] = value
        cache['mask'][slots] = True
        cache['id'][slots] = new_id
        cache['is_pred'][slots] = is_pred
        return slots

    def restore_cache(self, cache_name, slots):
        self.attn_caches[cache_name]['mask'][slots] = False


class MoTExpert(nn.Module):
    """One modality's expert stack, optionally with a NARROW residual width.

    The residual/FFN width `hidden_dim` may be smaller than the shared
    attention/interface width `shared_dim` (= num_heads * attn_head_dim). The
    block's q/k/v still project hidden_dim -> shared_dim so they concatenate with
    the other experts in the shared attention; FFN/norm run at hidden_dim. For a
    narrow expert we add four thin projections at the expert boundary so the rest
    of the model (token embeddings, output norm, heads) stays at `shared_dim`:
        in_proj  : shared_dim -> hidden_dim       (narrow the incoming embedding)
        time_proj: shared_dim -> 6*hidden_dim     (AdaLN modulation from time vec)
        text_proj: shared_dim -> hidden_dim       (text KV for the cross-attention)
        out_proj : hidden_dim -> shared_dim       (widen the output back)
    A full expert (hidden_dim == shared_dim) keeps identities and uses the shared
    timestep_proj/text directly, so it is byte-identical to the legacy stack.
    """

    def __init__(self, shared_dim, hidden_dim, num_heads, attn_head_dim, ffn_dim,
                 num_layers, cross_attn_norm, eps, attn_mode, do_cross_attn):
        super().__init__()
        self.shared_dim = int(shared_dim)
        self.hidden_dim = int(hidden_dim)
        self.narrow = self.hidden_dim != self.shared_dim
        self.do_cross_attn = bool(do_cross_attn)
        self.blocks = nn.ModuleList([
            WanTransformerBlock(self.hidden_dim, ffn_dim, num_heads, cross_attn_norm,
                                eps, attn_mode=attn_mode, attn_head_dim=attn_head_dim)
            for _ in range(num_layers)
        ])
        if self.narrow:
            self.in_proj = nn.Linear(self.shared_dim, self.hidden_dim)
            self.out_proj = nn.Linear(self.hidden_dim, self.shared_dim)
            self.time_proj = nn.Linear(self.shared_dim, 6 * self.hidden_dim)
            self.text_proj = (nn.Linear(self.shared_dim, self.hidden_dim)
                              if self.do_cross_attn else None)

    def embed_in(self, h):
        return self.in_proj(h) if self.narrow else h

    def embed_out(self, h):
        return self.out_proj(h) if self.narrow else h

    def modulation(self, timestep_proj_slice, time_vec_slice):
        """Per-token AdaLN modulation at hidden_dim: full experts reuse the shared
        6*shared_dim `timestep_proj`; narrow experts derive 6*hidden_dim from the
        time vector (shared_dim)."""
        if not self.narrow:
            return timestep_proj_slice
        return self.time_proj(time_vec_slice).unflatten(-1, (6, self.hidden_dim))

    def text_kv(self, text):
        if not self.do_cross_attn:
            return None
        return self.text_proj(text) if self.narrow else text


class MoTBackbone(nn.Module):
    """Per-modality experts (each a MoTExpert, possibly narrow) + one shared
    attention per layer.

    Args:
        num_layers/dim/num_heads/eps/cross_attn_norm/attn_mode: block hyper-params.
            `dim` is the SHARED interface/attention width; attn head dim = dim//num_heads.
        ffn_dim: default FFN width.
        expert_names: ordered modality names = concatenation/slice order.
        expert_ffn_dim: optional {name: ffn_dim} overrides.
        expert_hidden_dim: optional {name: hidden_dim} overrides for narrow experts
            (e.g. {"action": 1024, "tactile": 1024}); default = dim (full).
        cross_attn_experts: experts that run text cross-attention.
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        attn_mode: str = "torch",
        expert_names=("video", "action", "tactile"),
        expert_ffn_dim: Optional[dict] = None,
        expert_hidden_dim: Optional[dict] = None,
        cross_attn_experts=("video", "action"),
    ) -> None:
        super().__init__()
        self.expert_names = list(expert_names)
        self.cross_attn_experts = set(cross_attn_experts)
        self.num_layers = int(num_layers)
        self.shared_dim = int(dim)
        self.attn_head_dim = dim // num_heads        # shared attention head dim
        ffn_over = dict(expert_ffn_dim or {})
        hdim_over = dict(expert_hidden_dim or {})

        self.experts = nn.ModuleDict()
        for name in self.expert_names:
            self.experts[name] = MoTExpert(
                shared_dim=dim,
                hidden_dim=int(hdim_over.get(name, dim)),
                num_heads=num_heads,
                attn_head_dim=self.attn_head_dim,
                ffn_dim=int(ffn_over.get(name, ffn_dim)),
                num_layers=self.num_layers,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
                attn_mode=attn_mode,
                do_cross_attn=(name in self.cross_attn_experts),
            )
        self.shared_attn = nn.ModuleList(
            [SharedSelfAttention() for _ in range(self.num_layers)]
        )
        self._cross_masks: dict = {}
        # Layer-level activation checkpointing. The legacy model checkpoints a WHOLE
        # block (incl. attention) so q/k/v/attn are recomputed, not stored. The MoT
        # splits each block into pre/post around the shared attention, so a per-block
        # wrapper would leave q/k/v/attn (~10GB at S~15k) resident. Checkpointing the
        # ENTIRE layer body here restores legacy-granularity AC (fixes 80GB OOM).
        self.gradient_checkpointing = True

    # ───────────────────────── mask wiring ─────────────────────────
    def set_masks(self, self_block_mask=None, dense_self_mask=None, cross_masks=None):
        """Set the shared self-attn mask (all layers) and per-expert cross masks.
        `self_block_mask` (flex, GPU) OR `dense_self_mask` (bool [S,S], CPU);
        `cross_masks` maps expert name -> dense bool (1,1,S_q,S_text)."""
        for layer in range(self.num_layers):
            if dense_self_mask is not None:
                self.shared_attn[layer].set_dense_mask(dense_self_mask)
            else:
                self.shared_attn[layer].set_dense_mask(None)
                self.shared_attn[layer].set_block_mask(self_block_mask)
        self._cross_masks = dict(cross_masks or {})

    # ───────────────────────── forward ─────────────────────────
    def forward(self, hidden_states, encoder_hidden_states, timestep_proj, temb,
                rotary_emb, slices, collect_cache=False, update_cache=0, cache_name=None):
        """Run the MoT stack.

        Args:
            hidden_states: [1, S_total, shared_dim] — modality blocks concatenated
                in `slices` order (pad folded into the last expert's slice).
            encoder_hidden_states: [1, L_text, shared_dim] text KV.
            timestep_proj: [1, S_total, 6, shared_dim] AdaLN modulation (full experts).
            temb: [1, S_total, shared_dim] time VECTOR (narrow experts' time_proj
                source); may be None when all experts are full.
            rotary_emb: [1, S_total, 1, head_dim//2] complex RoPE freqs.
            slices: ordered list of (expert_name, start, end) spanning S_total.

        Returns:
            [1, S_total, shared_dim] updated, same layout as input.
        """
        names = [s[0] for s in slices]
        if names != self.expert_names:
            raise ValueError(
                f"slice order {names} must equal expert order {self.expert_names}")

        # 1. split + per-expert narrow projection / modulation / text / rope
        streams, mod, text, rope_e, seg_len = {}, {}, {}, {}, {}
        for (name, s, e) in slices:
            ex = self.experts[name]
            streams[name] = ex.embed_in(hidden_states[:, s:e])
            mod[name] = ex.modulation(timestep_proj[:, s:e],
                                      temb[:, s:e] if temb is not None else None)
            text[name] = ex.text_kv(encoder_hidden_states)
            rope_e[name] = rotary_emb[:, s:e]
            seg_len[name] = e - s

        # 2. layers: per-expert pre -> concat q/k/v -> shared attn -> per-expert post.
        # The ENTIRE layer body is one activation-checkpoint unit (legacy granularity):
        # in backward, q/k/v/attn/ffn are recomputed rather than held resident.
        kv_cache = [] if collect_cache else None
        order = [name for (name, _s, _e) in slices]

        def _layer(layer, *streams_in):
            sl_in = {order[i]: streams_in[i] for i in range(len(order))}
            q_chunks, k_chunks, v_chunks, post = [], [], [], []
            for name in order:
                block = self.experts[name].blocks[layer]
                q, k, v, residual, gate, cs, csc, cg = block(
                    sl_in[name], temb=mod[name], rotary_emb=rope_e[name], mot_mode='pre')
                q_chunks.append(q); k_chunks.append(k); v_chunks.append(v)
                post.append((name, block, residual, gate, cs, csc, cg))
            q = torch.cat(q_chunks, dim=1)
            k = torch.cat(k_chunks, dim=1)
            v = torch.cat(v_chunks, dim=1)
            if collect_cache:
                kv_cache.append((k, v))         # full-sequence K/V at this layer
            # update_cache/cache_name default to 0/None in training (no streaming cache;
            # SharedSelfAttention falls back to normal attn) and carry the streaming
            # KV-cache args at inference. Captured from the enclosing forward scope.
            attn = self.shared_attn[layer](q, k, v, update_cache=update_cache, cache_name=cache_name)
            out = {}
            cursor = 0
            for (name, block, residual, gate, cs, csc, cg) in post:
                sl = attn[:, cursor:cursor + seg_len[name]]
                do_cross = self.experts[name].do_cross_attn and (text[name] is not None)
                out[name] = block(
                    residual, mot_mode='post',
                    mot_kwargs=dict(
                        attn_output=sl, gate_msa=gate, c_shift_msa=cs,
                        c_scale_msa=csc, c_gate_msa=cg,
                        encoder_hidden_states=text[name], do_cross_attn=do_cross,
                        cross_attn_mask=self._cross_masks.get(name)))
                cursor += seg_len[name]
            return tuple(out[name] for name in order)

        use_ckpt = (self.gradient_checkpointing and torch.is_grad_enabled()
                    and not collect_cache)
        for layer in range(self.num_layers):
            cur = tuple(streams[name] for name in order)
            new = (_ckpt(_layer, layer, *cur, use_reentrant=False)
                   if use_ckpt else _layer(layer, *cur))
            streams = {order[i]: new[i] for i in range(len(order))}

        # 3. widen each expert back to shared_dim and concatenate in slice order
        out = torch.cat([self.experts[name].embed_out(streams[name])
                         for (name, _s, _e) in slices], dim=1)
        if collect_cache:
            return out, kv_cache
        return out

    @torch.no_grad()
    def forward_action_cached(self, a_noisy_stream, a_timestep_proj, a_temb,
                              a_rope, kv_cache, a_cols, rows_mask,
                              encoder_hidden_states=None, cross_attn_mask=None):
        """KV-cache fast action denoising: only the action-noisy tokens are
        recomputed each step; the fixed upstream (video / tactile / action-clean)
        K/V come from `kv_cache` (per-layer full-sequence K/V from a prefill forward
        with collect_cache=True). For every layer we recompute the a_noisy q/k/v,
        splice the new a_noisy K/V into the cached full-sequence K/V at columns
        `a_cols`, attend (a_noisy queries × all columns via `rows_mask`), and run the
        action block's post. Equivalent to the full joint forward's a_noisy output
        (verified) but skips the heavy video/tactile expert recompute per step.

        Args:
            a_noisy_stream: [1, S_anoisy, shared_dim] the action-noisy tokens.
            a_*: modulation/time/rope SLICES for the a_noisy tokens.
            kv_cache: list per layer of (k_full, v_full) [1, S_total, heads, dh].
            a_cols: slice for the a_noisy columns within S_total.
            rows_mask: dense bool [S_anoisy, S_total] (a_noisy queries × all cols).
        Returns: [1, S_anoisy, shared_dim] widened a_noisy output.
        """
        ex = self.experts["action"]
        stream = ex.embed_in(a_noisy_stream)
        a_mod = ex.modulation(a_timestep_proj, a_temb)      # per-expert AdaLN (own dim)
        text = ex.text_kv(encoder_hidden_states)            # action's text KV (own dim)
        do_cross = ex.do_cross_attn and (text is not None)
        for layer in range(self.num_layers):
            block = ex.blocks[layer]
            q, k, v, residual, gate, cs, csc, cg = block(
                stream, temb=a_mod, rotary_emb=a_rope, mot_mode='pre')
            k_full, v_full = kv_cache[layer]
            k_full = k_full.clone(); v_full = v_full.clone()
            k_full[:, a_cols] = k                       # splice fresh a_noisy K/V
            v_full[:, a_cols] = v
            attn = custom_sdpa(q, k_full, v_full, attn_mask=rows_mask)
            stream = block(
                residual, mot_mode='post',
                mot_kwargs=dict(attn_output=attn, gate_msa=gate, c_shift_msa=cs,
                                c_scale_msa=csc, c_gate_msa=cg,
                                encoder_hidden_states=text, do_cross_attn=do_cross,
                                cross_attn_mask=cross_attn_mask))
        return ex.embed_out(stream)


# ─────────────────────────── MoT model (3-expert) ───────────────────────────
class WanMoTTransformer3DModel(WanTransformer3DModel):
    """Mixture-of-Transformers variant of WanTransformer3DModel.

    The single shared `blocks` stack is replaced by three per-modality expert
    stacks (video / action / tactile) joined by one shared attention per layer.
    EVERYTHING ELSE is inherited unchanged from the parent — embeddings, text/time
    conditioning, GlobalTactile diffusion head, LocalTactile cross-attn, contact
    gate, output norm, losses — only the backbone execution differs, via the
    overridden `_run_backbone` hook.

    Cascade / ordering is preserved exactly: the concatenated sequence keeps the
    legacy layout [video | action | tactile | pad] and the legacy joint flex mask,
    so "predict visual+tactile, then action" still holds (it lives in the mask's
    frame-id causality, not in the weight split).

    Notes:
      * Experts share num_layers/num_heads/head_dim/dim (required for the shared
        attention); per-expert `mot_expert_ffn_dim` overrides are allowed but then
        warm-starting from a shared checkpoint can only copy the matching-width
        experts (see remap_shared_to_mot_state_dict).
      * Inference uses a streaming KV-cache (implemented below).
    """

    def __init__(self, *args, mot_expert_ffn_dim=None, mot_expert_hidden_dim=None,
                 mot_cross_attn_experts=("video", "action"), **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        inner_dim = cfg.num_attention_heads * cfg.attention_head_dim
        self.mot_expert_names = ("video", "action", "tactile")
        self.mot = MoTBackbone(
            num_layers=cfg.num_layers,
            dim=inner_dim,
            ffn_dim=cfg.ffn_dim,
            num_heads=cfg.num_attention_heads,
            cross_attn_norm=cfg.cross_attn_norm,
            eps=cfg.eps,
            attn_mode="torch",
            expert_names=self.mot_expert_names,
            expert_ffn_dim=mot_expert_ffn_dim,
            expert_hidden_dim=mot_expert_hidden_dim,   # e.g. {"action":1024,"tactile":1024}
            cross_attn_experts=mot_cross_attn_experts,
        )
        # the shared stack is superseded by the per-expert stacks
        del self.blocks
        # Persist the MoT-specific args into config.json so a saved checkpoint can be
        # rebuilt with the SAME expert structure for inference (see load_mot_checkpoint).
        self.register_to_config(
            is_mot=True,
            mot_expert_ffn_dim=mot_expert_ffn_dim,
            mot_expert_hidden_dim=mot_expert_hidden_dim,
            mot_cross_attn_experts=list(mot_cross_attn_experts),
        )

    # ───────────────── backbone hook (the only forward change) ─────────────────
    def _run_backbone(self, hidden_states, encoder_hidden_states, timestep_proj,
                      rotary_emb, self_attention_mask, cross_attention_mask,
                      split_list, batch_size, temb=None):
        # split_list = [v_noisy, v_clean, a_noisy, a_clean, t_noisy, t_clean, pad]
        v = split_list[0] + split_list[1]
        a = split_list[2] + split_list[3]
        t = split_list[4] + split_list[5] + split_list[6]   # tactile absorbs the pad
        slices = [
            ("video", 0, v),
            ("action", v, v + a),
            ("tactile", v + a, v + a + t),
        ]
        if v + a + t != hidden_states.shape[1]:
            raise ValueError(
                f"MoT slices sum ({v + a + t}) != sequence length "
                f"({hidden_states.shape[1]}); split_list={split_list}")
        text_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
        cross_masks = self._mot_cross_masks(split_list, batch_size, text_len,
                                            device=hidden_states.device)
        self.mot.set_masks(self_block_mask=self_attention_mask, cross_masks=cross_masks)
        return self.mot(hidden_states, encoder_hidden_states, timestep_proj, temb,
                        rotary_emb, slices)

    def _mot_cross_masks(self, split_list, batch_size, text_len, device):
        """Dense bool cross-attn masks (1,1,S_q,S_text) enforcing the legacy
        batch-isolation: a query token only attends text of its own (packed) batch.
        For batch_size==1 there is nothing to isolate -> no mask (attend all text),
        matching the legacy single-batch behaviour."""
        if batch_size <= 1 or text_len == 0:
            return {}
        if text_len % batch_size != 0:
            raise ValueError(
                f"text_len ({text_len}) not divisible by batch_size ({batch_size})")
        text_per = text_len // batch_size
        t_batch = torch.arange(batch_size, device=device).repeat_interleave(text_per)

        def q_batch(seg_lens):
            chunks = []
            for seg in seg_lens:
                if seg == 0:
                    continue
                if seg % batch_size != 0:
                    raise ValueError(
                        f"segment length {seg} not divisible by batch_size {batch_size}")
                chunks.append(
                    torch.arange(batch_size, device=device).repeat_interleave(seg // batch_size))
            if not chunks:
                return torch.empty(0, dtype=torch.long, device=device)
            return torch.cat(chunks)

        masks = {}
        seg_map = {"video": split_list[0:2], "action": split_list[2:4]}
        for name, seg in seg_map.items():
            if name not in self.mot.cross_attn_experts:
                continue
            qb = q_batch(seg)
            masks[name] = (qb[:, None] == t_batch[None, :])[None, None]  # (1,1,S_q,S_text)
        return masks

    # ─────────── cache hooks (self.blocks is gone; inference = Phase 2) ───────────
    def _expert_blocks(self):
        for name in self.mot.expert_names:
            for blk in self.mot.experts[name].blocks:
                yield blk

    def clear_cache(self, cache_name):
        for sa in self.mot.shared_attn:
            sa.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        for sa in self.mot.shared_attn:
            sa.clear_pred_cache(cache_name)

    def create_empty_cache(self, cache_name, attn_window,
                           latent_token_per_chunk, action_token_per_chunk,
                           device, dtype, batch_size):
        # Phase-2 streaming KV-cache: pool lives on each shared cross-expert
        # attention (the actual attention site in MoT), sized like the legacy path.
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2) * action_token_per_chunk
        for sa in self.mot.shared_attn:
            sa.init_kv_cache(cache_name, total_tolen,
                             self.num_attention_heads, self.attention_head_dim,
                             device, dtype, batch_size)

    def _run_main_blocks(self, hidden_states, encoder_hidden_states, timestep_proj,
                         temb, rotary_emb, update_cache, cache_name, action_mode,
                         main_token_count, tactile_token_count):
        # MoT streaming inference: the [main, tactile] sequence -> per-modality slices.
        # One of video/action is empty per pass (video pass: action empty; action pass:
        # video empty); the apply_rotary_emb 0-length guard + empty-slice handling cope.
        # Runs the expert cascade with the shared-attn streaming KV cache.
        m, t = int(main_token_count), int(tactile_token_count)
        if action_mode:
            slices = [("video", 0, 0), ("action", 0, m), ("tactile", m, m + t)]
        else:
            slices = [("video", 0, m), ("action", m, m), ("tactile", m, m + t)]
        return self.mot(hidden_states, encoder_hidden_states, timestep_proj, temb,
                        rotary_emb, slices, update_cache=update_cache, cache_name=cache_name)


def remap_shared_to_mot_state_dict(legacy_sd, expert_names=("video", "action", "tactile")):
    """Warm-start: turn a legacy (shared-backbone) state_dict into a MoT one by
    copying each `blocks.{i}.*` tensor into every expert `mot.experts.{name}.blocks.{i}.*`.
    All non-block keys (embeddings, heads, contact gate, output norm, ...) are kept
    verbatim. NOTE: this assumes the listed experts are FULL-width (== shared dim);
    narrow experts cannot receive the wide WAN blocks and must be left random (use
    the in-place warm-start in utils.load_mot_transformer, which skips them). Load
    with strict=False (shared_attn / narrow-expert projections have no source)."""
    new_sd = {}
    prefix = "blocks."
    for key, value in legacy_sd.items():
        if key.startswith(prefix):
            rest = key[len(prefix):]                 # "{layer}.{...}"
            for name in expert_names:
                new_sd[f"mot.experts.{name}.blocks.{rest}"] = value.clone()
        else:
            new_sd[key] = value
    return new_sd
