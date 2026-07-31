# Copyright 2025-2026 NeoteAI Team. All rights reserved.
import torch
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel


def _reset_module_parameters(module):
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def _materialize_tactile_modules(model, device):
    tactile_module_names = [
        # GlobalTactile latent modules (predicted in the self-attention sequence)
        "tactile_patch_embed",
        "sensor_id_embed",
        "tactile_norm",
        # LocalTactile cross-attn (action stream)
        "local_tactile_patch_embed",
        "local_tactile_sensor_embed",
        "local_tactile_frame_embed",
        "local_tactile_h_embed",
        "local_tactile_w_embed",
        "local_tactile_norm",
        "local_tactile_pre_norm",
        "local_tactile_cross_attn",
        # symdiff tactile: tactile diffusion output head
        "tactile_proj_out",
        # predictive-contact gate (opt-in predictive-contact-gate add-on; not in base ckpts)
        "contact_gate",
    ]
    target_device = torch.device(device)
    for module_name in tactile_module_names:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        has_meta_params = any(getattr(param, "is_meta", False) for param in module.parameters(recurse=True))
        has_meta_buffers = any(getattr(buf, "is_meta", False) for buf in module.buffers(recurse=True))
        if not has_meta_params and not has_meta_buffers:
            continue
        module.to_empty(device=target_device)
        _reset_module_parameters(module)
        # contact_gate was just materialised fresh (absent from the checkpoint):
        # re-zero its FiLM so it starts as an exact identity / no-op. A checkpoint
        # that already contains a trained contact_gate is NOT meta -> skipped above
        # -> keeps its trained weights.
        if module_name == "contact_gate" and hasattr(module, "reset_film"):
            module.reset_film()


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def _resize_action_head(model, target_action_dim, device):
    """Replace action_embedder (Linear in→inner) and action_proj_out
    (Linear inner→out) when the new schema size differs from the pretrained
    one. Both layers are re-initialised from scratch — the rest of the
    transformer keeps the loaded weights.

    Used when a downstream task needs a different action super-schema (e.g.
    20-dim left/right canonical, or any other reshuffle) without retraining
    the entire VLA.
    """
    import torch.nn as nn
    old_in_dim = model.action_embedder.in_features
    inner_dim = model.action_embedder.out_features
    if target_action_dim == old_in_dim:
        return
    model.action_embedder = nn.Linear(target_action_dim, inner_dim).to(
        device=device, dtype=model.action_embedder.weight.dtype)
    model.action_proj_out = nn.Linear(inner_dim, target_action_dim).to(
        device=device, dtype=model.action_proj_out.weight.dtype)
    # Keep config in sync — downstream code (loss, eval) AND from_pretrained on
    # resume read model.config.action_dim. Assigning model.config.action_dim
    # directly is a NO-OP (diffusers FrozenDict ignores it), which makes the saved
    # config.json disagree with the resized weights and re-randomises the action
    # head on every resume. register_to_config actually persists the new value so
    # the saved config matches the 20-dim head and resume skips a second resize.
    if hasattr(model, "register_to_config"):
        model.register_to_config(action_dim=target_action_dim)
    elif hasattr(model, "config"):
        model.config.action_dim = target_action_dim


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    target_action_dim=None,
    **kwargs,
):
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        **kwargs,
    )
    _materialize_tactile_modules(model, torch_device)
    if target_action_dim is not None:
        _resize_action_head(model, target_action_dim, torch_device)
    return model.to(torch_device)


@torch.no_grad()
def _warmstart_mot_from_legacy(mot, legacy, warmstart_experts):
    """In-place copy a legacy (shared-backbone) model's weights into a MoT model:
    every `blocks.{i}.*` tensor is copied into each chosen FULL-width expert
    `mot.experts.{e}.blocks.{i}.*`; all other tensors (token embeddings, text/time
    conditioning, tactile modules, heads, output norm) are copied verbatim. In-place
    `copy_` avoids building a 3x-cloned state_dict.

    NARROW experts (hidden_dim != shared dim) cannot receive the wide WAN blocks, so
    they are skipped and keep their random init (+ their thin in/out/time/text
    projections). Returns (n_copied, full_warmstarted, skipped_narrow)."""
    legacy_sd = legacy.state_dict()
    tgt = dict(mot.named_parameters())
    tgt.update(dict(mot.named_buffers()))
    # only FULL-width experts can take the wide WAN blocks
    full_ws = [e for e in warmstart_experts if not mot.mot.experts[e].narrow]
    skipped = [e for e in warmstart_experts if mot.mot.experts[e].narrow]
    copied, missed = 0, []

    def _copy(dst_name, src):
        nonlocal copied
        dst = tgt.get(dst_name)
        if dst is None:
            missed.append(dst_name)
        elif tuple(dst.shape) != tuple(src.shape):
            missed.append(f"{dst_name}[{tuple(dst.shape)} vs {tuple(src.shape)}]")
        else:
            dst.copy_(src)
            copied += 1

    for name, src in legacy_sd.items():
        if name.startswith("blocks."):
            rest = name[len("blocks."):]
            for e in full_ws:
                _copy(f"mot.experts.{e}.blocks.{rest}", src)
        else:
            _copy(name, src)   # model-level embeddings/heads/conditioning, verbatim

    if missed:
        raise RuntimeError(
            f"MoT warm-start: {len(missed)} legacy keys could not be mapped, "
            f"e.g. {missed[:10]}")
    return copied, full_ws, skipped


def load_mot_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    target_action_dim=None,
    mot_expert_ffn_dim=None,
    mot_expert_hidden_dim=None,
    mot_cross_attn_experts=("video", "action"),
    mot_warmstart_experts=("video", "action", "tactile"),
    **kwargs,
):
    """Build a Mixture-of-Transformers TWAM model, warm-started from a shared-backbone
    WAN checkpoint. The legacy model is loaded normally (WAN transformer weights +
    materialised tactile modules + resized action head); its weights are then copied
    into the MoT model — `blocks.{i}.*` into each `mot_warmstart_experts` (default
    all three), everything else verbatim. Experts not in `mot_warmstart_experts`
    keep their random init."""
    from .mot import WanMoTTransformer3DModel

    legacy = load_transformer(
        transformer_path,
        torch_dtype=torch_dtype,
        torch_device='cpu',
        target_action_dim=target_action_dim,
        **kwargs,
    )
    cfg = {k: v for k, v in dict(legacy.config).items() if not k.startswith("_")}
    # Build the MoT skeleton in the SAME (low-precision) dtype: 3 experts built in
    # fp32 on every rank would OOM the container memory cgroup (a single rank peaks
    # at ~3x the base model). set_default_dtype makes nn.Linear/LayerNorm/Parameter
    # allocate directly in torch_dtype, avoiding a transient fp32 copy. FSDP master
    # params therefore start in torch_dtype (pass bf16 for the memory-lean path).
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        mot = WanMoTTransformer3DModel(
            **cfg,
            mot_expert_ffn_dim=mot_expert_ffn_dim,
            mot_expert_hidden_dim=mot_expert_hidden_dim,
            mot_cross_attn_experts=tuple(mot_cross_attn_experts),
        )
    finally:
        torch.set_default_dtype(old_dtype)
    n, full_ws, skipped = _warmstart_mot_from_legacy(
        mot, legacy, tuple(mot_warmstart_experts))
    print(f"[load_mot_transformer] warm-started {n} tensors; full experts from WAN="
          f"{full_ws}; narrow experts left random={skipped}; "
          f"cross-attn experts={tuple(mot_cross_attn_experts)}; "
          f"hidden_dim={mot_expert_hidden_dim}")
    del legacy
    import gc
    gc.collect()
    return mot.to(torch_device)


def load_mot_checkpoint(checkpoint_dir, torch_dtype=torch.bfloat16,
                        torch_device="cpu", attn_mode="torch",
                        config_overrides=None):
    """Load a trained MoT checkpoint (a `.../transformer` dir with config.json +
    diffusion_pytorch_model.safetensors), rebuilding the WanMoTTransformer3DModel
    with the saved expert structure (hidden_dim / ffn_dim / cross-attn experts
    recorded in config.json by register_to_config).

    config_overrides (dict|None): optional model-init overrides applied on top of
    the checkpoint's config.json BEFORE building. Used for post-training "flips"
    (e.g. use_local_tactile False->True) where the local-off pretrain ckpt lacks a
    module the post-train wants: the module is built (zero-init) and its
    absent-in-ckpt weights are tolerated as missing. None (default, e.g. inference/
    serve) => exact rebuild from the ckpt config, strict load as before."""
    import json
    import os
    from safetensors.torch import load_file
    from .mot import WanMoTTransformer3DModel

    tdir = os.path.join(checkpoint_dir, "transformer")
    if not os.path.isdir(tdir):
        tdir = checkpoint_dir
    with open(os.path.join(tdir, "config.json")) as f:
        cfg = json.load(f)
    if not cfg.get("is_mot", False):
        raise ValueError(f"{tdir}/config.json is not a MoT checkpoint (is_mot missing).")
    mot_ffn = cfg.get("mot_expert_ffn_dim")
    mot_hdim = cfg.get("mot_expert_hidden_dim")
    mot_cross = cfg.get("mot_cross_attn_experts", ("video", "action"))

    # Capture the ckpt's ORIGINAL modality flags before applying overrides, so we
    # can tell which modules a config_overrides "flip" newly enables (their weights
    # are legitimately absent from the ckpt -> keep zero-init, tolerate as missing).
    ckpt_use_local = bool(cfg.get("use_local_tactile", False))
    ckpt_use_gate = bool(cfg.get("use_contact_gate", False))
    if config_overrides:
        for k, v in config_overrides.items():
            cfg[k] = v

    # keep only the kwargs the parent WanTransformer3DModel actually accepts: drop
    # diffusers private keys, the MoT-specific ones we pass explicitly, and any
    # config keys this code version's model no longer accepts -- e.g. a newer-branch
    # pretrain ckpt whose config.json carries use_proprio_state that this tree lacks
    # (forwarding it would raise TypeError in WanTransformer3DModel.__init__).
    import inspect
    from .model import WanTransformer3DModel
    _sig = inspect.signature(WanTransformer3DModel.__init__).parameters
    _accepted = None if any(p.kind == p.VAR_KEYWORD for p in _sig.values()) \
        else (set(_sig) - {"self"})
    drop = {"is_mot", "mot_expert_ffn_dim", "mot_expert_hidden_dim",
            "mot_cross_attn_experts"}
    init_cfg = {k: v for k, v in cfg.items()
                if not k.startswith("_") and k not in drop
                and (_accepted is None or k in _accepted)}
    init_cfg["attn_mode"] = attn_mode  # inference: SDPA, no flex compile

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        model = WanMoTTransformer3DModel(
            **init_cfg,
            mot_expert_ffn_dim=mot_ffn,
            mot_expert_hidden_dim=mot_hdim,
            mot_cross_attn_experts=tuple(mot_cross),
        )
    finally:
        torch.set_default_dtype(old)

    sd = load_file(os.path.join(tdir, "diffusion_pytorch_model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Tolerated missing keys: the shared-attn stub is always allowed; and a module
    # NEWLY enabled via config_overrides (built here but off in the ckpt) is
    # legitimately absent -> keep its init. LocalTactile's out-proj is zero-init
    # (identity at step 0), so warm-starting a local-off ckpt is a no-op until
    # training moves it. Normal loads (no override) keep the strict check unchanged.
    tolerated = ["mot.shared_attn"]
    if bool(cfg.get("use_local_tactile", False)) and not ckpt_use_local:
        tolerated.append("local_tactile")
    if bool(cfg.get("use_contact_gate", False)) and not ckpt_use_gate:
        tolerated.append("contact_gate")
    missing = [k for k in missing if not any(k.startswith(p) for p in tolerated)]
    if missing or unexpected:
        raise RuntimeError(
            f"load_mot_checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    model.eval()
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
