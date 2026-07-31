"""MoT inference — denoising-consistency check (correctness first).

This does NOT re-derive an inference path (the legacy one drifted from the model).
It reuses the EXACT training machinery — Trainer(inference_only=True) + the same
_prepare_input_dict / _add_noise / forward_train / compute_loss — and verifies that
a trained (overfit) MoT checkpoint, run in eval, reconstructs the training data:

  * the three flow-matching losses should be ~the trained-step losses (model+forward
    are self-consistent: inference computes the same function as training);
  * the predicted velocity v=ε−x0 should recover x0 from the noised latent
    (x0_hat = x_noisy − σ·v), measured as a per-frame latent reconstruction MSE.

Run on a GPU (flex) box:
  cd <code> && python n0_twam/render_mot.py --ckpt <.../checkpoint_step_400/transformer> --n-batches 5
"""
import argparse
import os
import sys

# The package mixes import styles: train.py uses `from models...` (needs code/n0_twam
# on path) while model.py uses `from n0_twam.utils...` (needs code/ on path). Add BOTH.
_HERE = os.path.dirname(os.path.abspath(__file__))   # .../code/n0_twam
sys.path.insert(0, _HERE)                            # configs / models / train / utils / dataset
sys.path.insert(0, os.path.dirname(_HERE))           # code/  -> `import n0_twam`

import torch
from torch.utils.data import DataLoader

from configs import TWAM_CONFIGS
from train import Trainer
from models.utils import load_mot_checkpoint
from dataset import MultiLatentLeRobotDataset


def _sigma_of(scheduler, timesteps_flat):
    """Map per-frame timesteps -> sigma using the scheduler's sigmas table."""
    ts = timesteps_flat.detach().float().cpu()
    tid = torch.argmin((scheduler.timesteps[:, None] - ts[None, :]).abs(), dim=0)
    return scheduler.sigmas[tid]


@torch.no_grad()
def _decode_one(vae, lat_bcfhw):
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(lat_bcfhw.device, vae.dtype)
    std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(lat_bcfhw.device, vae.dtype)
    lat = lat_bcfhw.to(vae.dtype) * std + mean
    vid = vae.decode(lat, return_dict=False)[0]                 # [B,3,F',H',W'] in [-1,1]
    return ((vid.float().clamp(-1, 1) + 1) / 2 * 255).round().byte().cpu()


@torch.no_grad()
def decode_video_latent(vae, latent_bcfhw, tile_w=16):
    """Decode a normalized WAN video latent [B, z_dim, F, H, W] -> RGB uint8 frames.

    IMPORTANT: training tiles multiple CAMERAS along the latent W axis
    (_cat_video_latents: torch.cat(per-cam, dim=W)). A single camera is tile_w wide
    (=16 here). The VAE must decode EACH camera separately (it has no notion of the
    horizontal cam-stack); decoding the concatenated W in one shot scrambles the
    spatial conv -> snow. So split W into tile_w-wide cameras, decode each, then
    re-tile the decoded RGB along pixel-W. tile_w<=0 (or W==tile_w) = single video."""
    B, C, F, H, W = latent_bcfhw.shape
    if tile_w and tile_w > 0 and W > tile_w and W % tile_w == 0:
        n = W // tile_w
        cams = [_decode_one(vae, latent_bcfhw[:, :, :, :, i * tile_w:(i + 1) * tile_w])
                for i in range(n)]
        return torch.cat(cams, dim=-1)                          # tile decoded RGB along W
    return _decode_one(vae, latent_bcfhw)


def _timestep_for_sigma(sched, sigma):
    """Nearest training timestep for a target sigma."""
    tid = torch.argmin((sched.sigmas - float(sigma)).abs())
    return sched.timesteps[tid].item()


@torch.no_grad()
def sample_video(trainer, batch, n_steps=30, teacher_force=False):
    """i2v rectified-flow sampling that DRIVES forward_train (the exact trained
    computation) — frame 0 is the conditioning image, frames 1..F-1 are generated
    from noise via Euler steps x <- x + v*(dsigma), v=ε−x0 predicted by the model.

    teacher_force=True  : clean-stream = true x0 (plumbing test; must reconstruct).
    teacher_force=False : free i2v — clean-stream = self-conditioned x0_hat estimate
                          (frame 0 = true), so the model only ever sees frame 0 as
                          ground truth and generates the rest.
    Returns (x_final, x0_true) latents [B,48,F,H,W]."""
    model = trainer.transformer
    sched = trainer.train_scheduler_latent
    input_dict = trainer._prepare_input_dict(batch)
    # Fixed chunk/window for sampling: chunk_size=1 -> each frame its own chunk, so
    # frame f attends clean frames < f (clean autoregressive i2v causality).
    input_dict['chunk_size'] = 1
    input_dict['window_size'] = 64

    ld = input_dict['latent_dict']
    x0_true = ld['latent'].clone()                      # [B,48,F,H,W]
    B, C, F, H, W = x0_true.shape
    dev = x0_true.device

    # condition the action/tactile streams as CLEAN (sigma~0): for video i2v we feed
    # them as context, not as generation targets (their noisy halves = their clean).
    t0 = _timestep_for_sigma(sched, 0.0)
    ad = input_dict['action_dict']
    ad['noisy_latents'] = ad['latent'].clone()
    ad['timesteps'] = torch.full_like(ad['timesteps'], t0)
    if 'tactile_global_noisy_latent' in ad and ad.get('tactile_global_clean_latent') is not None:
        ad['tactile_global_noisy_latent'] = ad['tactile_global_clean_latent'].clone()
        if 'tactile_global_timesteps' in ad:
            ad['tactile_global_timesteps'] = torch.full_like(ad['tactile_global_timesteps'], t0)

    # sigma schedule (subset of the training sigmas, high->low), n_steps+1 points
    full_sig = sched.sigmas
    idx = torch.linspace(0, len(full_sig) - 1, n_steps + 1).round().long()
    sigmas = full_sig[idx].tolist()                     # [n_steps+1], ~1 -> ~0

    # init: frame0 = true (condition), rest = noise at sigma[0]
    x = torch.randn_like(x0_true) * 1.0
    x[:, :, 0:1] = x0_true[:, :, 0:1]
    x0_hat = x0_true[:, :, 0:1].repeat(1, 1, F, 1, 1).clone()  # clean-stream init

    cond_ts = torch.full((B, F), t0, device=dev, dtype=ld['timesteps'].dtype)
    for i in range(n_steps):
        sig, sig_next = sigmas[i], sigmas[i + 1]
        t_sig = _timestep_for_sigma(sched, sig)
        ts = torch.full((B, F), t_sig, device=dev, dtype=ld['timesteps'].dtype)
        ts[:, 0] = t0                                   # frame 0 clean
        ld['noisy_latents'] = x
        ld['timesteps'] = ts
        ld['cond_timesteps'] = cond_ts
        ld['latent'] = x0_true if teacher_force else x0_hat
        pred = model(input_dict, train_mode=True)
        v = _v_dense(trainer, pred[0], x0_true)         # dense velocity [B,48,F,H,W]
        x = x + v * (sig_next - sig)
        x[:, :, 0:1] = x0_true[:, :, 0:1]               # keep frame 0 fixed
        x0_hat = (x - sig_next * v)
        x0_hat[:, :, 0:1] = x0_true[:, :, 0:1]
    return x, x0_true


@torch.no_grad()
def sample_video_ar(trainer, batch, n_steps=8):
    """Faithful autoregressive diffusion-forcing i2v (chunk_size=1, frame-by-frame).
    Frame 0 is the condition; frame k (k=1..F-1) is denoised from noise over n_steps
    Euler steps while frames 0..k-1 are held CLEAN (the truly-generated past) and
    future frames are noise. The chunk-causal mask makes frame k attend only clean
    frames < k, so each frame conditions on the real generated past (no self-cond
    approximation). Returns (x_clean[B,48,F,H,W], x0_true)."""
    model = trainer.transformer
    sched = trainer.train_scheduler_latent
    input_dict = trainer._prepare_input_dict(batch)
    input_dict['chunk_size'] = 1                          # each frame its own chunk
    input_dict['window_size'] = 256
    ld = input_dict['latent_dict']
    x0_true = ld['latent'].clone()                        # [B,48,F,H,W]
    B, C, F, H, W = x0_true.shape
    dev = x0_true.device
    tdtype = ld['timesteps'].dtype

    t0 = _timestep_for_sigma(sched, 0.0)
    tmax = _timestep_for_sigma(sched, sched.sigmas[0].item())
    # action/tactile as clean conditioning context
    ad = input_dict['action_dict']
    ad['noisy_latents'] = ad['latent'].clone()
    ad['timesteps'] = torch.full_like(ad['timesteps'], t0)
    if 'tactile_global_noisy_latent' in ad and ad.get('tactile_global_clean_latent') is not None:
        ad['tactile_global_noisy_latent'] = ad['tactile_global_clean_latent'].clone()
        if 'tactile_global_timesteps' in ad:
            ad['tactile_global_timesteps'] = torch.full_like(ad['tactile_global_timesteps'], t0)

    full_sig = sched.sigmas
    idx = torch.linspace(0, len(full_sig) - 1, n_steps + 1).round().long()
    sigmas = full_sig[idx].tolist()                       # ~1 -> ~0

    clean = x0_true.clone()                               # holds generated clean frames
    # only frame 0 is truly known; init others to frame0 (overwritten as generated)
    clean[:, :, 1:] = x0_true[:, :, 0:1]
    cond_ts = torch.full((B, F), t0, device=dev, dtype=tdtype)

    for k in range(1, F):
        xk = torch.randn(B, C, 1, H, W, device=dev, dtype=x0_true.dtype)
        for i in range(n_steps):
            sig, sig_next = sigmas[i], sigmas[i + 1]
            # full sequence: past(<k)=clean, current(k)=xk@sig, future(>k)=noise
            noisy = clean.clone()
            noisy[:, :, k:k + 1] = xk
            if k + 1 < F:
                noisy[:, :, k + 1:] = torch.randn_like(noisy[:, :, k + 1:])
            ts = torch.full((B, F), t0, device=dev, dtype=tdtype)
            ts[:, k] = _timestep_for_sigma(sched, sig)
            if k + 1 < F:
                ts[:, k + 1:] = tmax
            ld['noisy_latents'] = noisy
            ld['timesteps'] = ts
            ld['cond_timesteps'] = cond_ts
            ld['latent'] = clean                          # clean stream = generated past (true)
            pred = model(input_dict, train_mode=True)
            v = _v_dense(trainer, pred[0], x0_true)[:, :, k:k + 1]
            xk = xk + v * (sig_next - sig)
        clean[:, :, k:k + 1] = xk                         # fix frame k as clean
    return clean, x0_true


def _v_dense(trainer, video_pred, ref_bcfhw):
    from utils import data_seq_to_patch
    return data_seq_to_patch(trainer.patch_size, video_pred,
                             ref_bcfhw.shape[-3], ref_bcfhw.shape[-2], ref_bcfhw.shape[-1],
                             batch_size=ref_bcfhw.shape[0]).float()


def _tac_v_dense(trainer, tactile_pred, gt_bscfhw):
    """tactile_pred (B', S*F*spatial, 48 patch) -> dense velocity (B,S,C,F,H,W),
    mirroring train.py's tactile-loss un-patchify (data_seq_to_patch over B*S)."""
    from utils import data_seq_to_patch
    B, S, C, F, H, W = gt_bscfhw.shape
    return data_seq_to_patch(
        trainer.patch_size,
        tactile_pred.reshape(B * S, F * H * W, C),
        F, H, W, batch_size=B * S,
    ).reshape(B, S, C, F, H, W).float()


@torch.no_grad()
def sample_tactile_ar(trainer, batch, n_steps=8):
    """Autoregressive diffusion-forcing generation of the GlobalTactile latent,
    co-generated with the video stream exactly as in training. Frame 0 is the
    condition (clean), frames 1..F-1 are denoised frame-by-frame while past frames
    are held clean (matches the chunk-causal mask). Video stream is fed as clean
    context so tactile conditions on the true video (isolates tactile generation).
    Returns (gt_clean[B,S,C,F,H,W] generated, gt_true[B,S,C,F,H,W])."""
    model = trainer.transformer
    sched = trainer.train_scheduler_tactile
    input_dict = trainer._prepare_input_dict(batch)
    input_dict['chunk_size'] = 1
    input_dict['window_size'] = 256
    ld = input_dict['latent_dict']
    ad = input_dict['action_dict']
    dev = ld['latent'].device
    tdtype = ld['timesteps'].dtype

    t0 = _timestep_for_sigma(trainer.train_scheduler_latent, 0.0)
    tmax = _timestep_for_sigma(trainer.train_scheduler_latent,
                               trainer.train_scheduler_latent.sigmas[0].item())
    # video + action as CLEAN context (we only generate tactile here)
    ld['noisy_latents'] = ld['latent'].clone()
    ld['timesteps'] = torch.full_like(ld['timesteps'], t0)
    ld['cond_timesteps'] = torch.full_like(ld['timesteps'], t0)
    ad['noisy_latents'] = ad['latent'].clone()
    ad['timesteps'] = torch.full_like(ad['timesteps'], t0)

    # the dataset may CFG-drop tactile for some batches -> prepare_input_dict then
    # omits the tactile_global_* keys. For rendering we FORCE tactile present from the
    # raw batch latent (and clear the drop flag), so generation always has a target.
    if 'tactile_global_clean_latent' not in ad:
        g = batch['tactile_global_latent'].to(dev, ld['latent'].dtype)
        if g.dim() == 5:
            g = g.unsqueeze(0)                                # -> (B,S,C,F,H,W)
        ad['tactile_global_clean_latent'] = g.clone()
        ad['tactile_global_noisy_latent'] = g.clone()
        ad['tactile_sensor_ids'] = batch.get(
            'tactile_sensor_ids',
            torch.zeros(g.shape[0], g.shape[1], dtype=torch.long, device=dev))
        ad['tactile_cond_drop'] = torch.tensor(False, device=dev)
        Bf, Sf = g.shape[0], g.shape[1]
        ad['tactile_global_timesteps'] = torch.full(
            (Bf, g.shape[3]), t0, device=dev, dtype=tdtype)

    # IMPORTANT: ad['tactile_global_clean_latent'] comes from _add_noise, which with
    # noisy_cond_prob>0 PARTIALLY NOISES the "clean" latent (noisy-cond augmentation)
    # -> decoding it shows snow. For the ground-truth reference we must use the RAW
    # batch latent (never touched by _add_noise). Fall back to ad only if absent.
    if batch.get('tactile_global_latent') is not None:
        gt_true = batch['tactile_global_latent'].to(dev, ld['latent'].dtype).clone()
        if gt_true.dim() == 5:
            gt_true = gt_true.unsqueeze(0)                   # -> (B,S,C,F,H,W)
    else:
        gt_true = ad['tactile_global_clean_latent'].clone()
    B, S, C, F, H, W = gt_true.shape
    # also feed the RAW clean latent as the conditioning/clean stream during sampling
    ad['tactile_global_clean_latent'] = gt_true.clone()
    full_sig = sched.sigmas
    idx = torch.linspace(0, len(full_sig) - 1, n_steps + 1).round().long()
    sigmas = full_sig[idx].tolist()

    gt_clean = gt_true.clone()
    gt_clean[:, :, :, 1:] = gt_true[:, :, :, 0:1]            # only frame0 known
    # tactile_global_timesteps is (B, F) per-frame
    base_ts = torch.full((B, F), t0, device=dev, dtype=ad['tactile_global_timesteps'].dtype)

    for k in range(1, F):
        xk = torch.randn(B, S, C, 1, H, W, device=dev, dtype=gt_true.dtype)
        for i in range(n_steps):
            sig, sig_next = sigmas[i], sigmas[i + 1]
            noisy = gt_clean.clone()
            noisy[:, :, :, k:k + 1] = xk
            if k + 1 < F:
                noisy[:, :, :, k + 1:] = torch.randn_like(noisy[:, :, :, k + 1:])
            ts = base_ts.clone()
            ts[:, k] = _timestep_for_sigma(sched, sig)
            if k + 1 < F:
                ts[:, k + 1:] = tmax
            ad['tactile_global_noisy_latent'] = noisy
            ad['tactile_global_clean_latent'] = gt_clean       # generated past = clean cond
            ad['tactile_global_timesteps'] = ts
            pred = model(input_dict, train_mode=True)
            v = _tac_v_dense(trainer, pred[2], gt_true)[:, :, :, k:k + 1]
            xk = xk + v * (sig_next - sig)
        gt_clean[:, :, :, k:k + 1] = xk
    return gt_clean, gt_true


def save_compare_video(left_b3fhw, right_b3fhw, out_dir, tag, fps=4,
                       left_label="generated", right_label="ground truth",
                       title=None):
    """Save a side-by-side [left | right] comparison as PNG frames + an mp4, with
    a caption banner on top (title + per-side labels) and a white separator."""
    import numpy as np
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    f = min(left_b3fhw.shape[2], right_b3fhw.shape[2])
    sep_w = 4
    banner_h = 28
    frames = []
    for fi in range(f):
        l = left_b3fhw[0, :, fi].permute(1, 2, 0).numpy()
        r = right_b3fhw[0, :, fi].permute(1, 2, 0).numpy()
        sep = np.full((l.shape[0], sep_w, 3), 255, dtype=np.uint8)
        body = np.concatenate([l, sep, r], axis=1)              # [H, Wl+sep+Wr, 3]
        Wtot = body.shape[1]
        banner = np.zeros((banner_h, Wtot, 3), dtype=np.uint8)  # black caption strip
        canvas = np.concatenate([banner, body], axis=0)
        img = Image.fromarray(canvas)
        d = ImageDraw.Draw(img)
        Wl = l.shape[1]
        cap = title or tag
        d.text((4, 2), cap, fill=(255, 255, 0))                 # title (yellow), top-left
        # frame number top-RIGHT (right-aligned) so it never overlaps the title
        fnum = f"frame {fi:02d}"
        try:
            fw = d.textlength(fnum)
        except Exception:
            fw = 8 * len(fnum)
        d.text((Wtot - fw - 4, 2), fnum, fill=(200, 200, 200))
        d.text((4, 15), left_label, fill=(0, 255, 0))           # left label = green
        d.text((Wl + sep_w + 4, 15), right_label, fill=(0, 200, 255))  # right label = cyan
        frames.append(np.asarray(img))
        img.save(os.path.join(out_dir, f"{tag}_f{fi:02d}.png"))
    try:
        import imageio
        imageio.mimsave(os.path.join(out_dir, f"{tag}.mp4"), frames, fps=fps)
        print(f"[compare] {tag}: {f} frames (L={left_label} | R={right_label}) -> {out_dir}/{tag}.mp4")
    except Exception as e:
        print(f"[compare] {tag}: saved {f} PNG frames to {out_dir} (mp4 skip: {e})")


def save_frames(vid_b3fhw, out_dir, tag):
    import numpy as np  # noqa
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    _, _, f, _, _ = vid_b3fhw.shape
    for fi in range(f):
        frame = vid_b3fhw[0, :, fi].permute(1, 2, 0).numpy()    # [H,W,3]
        Image.fromarray(frame).save(os.path.join(out_dir, f"{tag}_f{fi:02d}.png"))
    print(f"[decode] {tag}: video {tuple(vid_b3fhw.shape)} -> saved {f} frames to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="base")
    ap.add_argument("--ckpt", required=True, help="path to .../checkpoint_step_N/transformer")
    ap.add_argument("--n-batches", type=int, default=5)
    ap.add_argument("--decode-check", action="store_true",
                    help="decode the TRUE training video latent -> RGB (validate VAE decode in isolation)")
    ap.add_argument("--tactile-decode", action="store_true",
                    help="try decoding the GlobalTactile latent -> RGB")
    ap.add_argument("--generate", action="store_true", help="i2v sampling -> decode -> frames")
    ap.add_argument("--teacher-force", action="store_true",
                    help="clean-stream = true x0 (plumbing test); else free i2v (self-cond)")
    ap.add_argument("--n-steps", type=int, default=30, help="Euler sampling steps")
    ap.add_argument("--ar", action="store_true",
                    help="faithful autoregressive frame-by-frame diffusion-forcing i2v")
    ap.add_argument("--gen-tactile", action="store_true",
                    help="autoregressively GENERATE GlobalTactile (co-gen with video), "
                         "decode, and save side-by-side [generated | true] compare video")
    ap.add_argument("--fps", type=int, default=15, help="mp4 playback fps (129 frames / fps = duration)")
    ap.add_argument("--vae", default="/path/to/base-model/vae")
    ap.add_argument("--out-dir", default="/path/to/workspace/pretrained/render_out")
    ap.add_argument("--val", action="store_true",
                    help="use the VAL split (config.val_dataset_path) instead of train")
    ap.add_argument("--n-gen", type=int, default=1,
                    help="number of samples to generate per run (--generate); one model load")
    args = ap.parse_args()

    config = TWAM_CONFIGS[args.config_name]
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.enable_wandb = False
    config.load_worker = 0
    dev = "cuda:0"

    if args.val:
        config.dataset_path = config.val_dataset_path
        print(f"[render] using VAL data: {config.dataset_path}")

    # put every render run in its own timestamped subfolder for easy browsing:
    # <out-dir>/<ckptstep>_<mode>_<YYYYmmdd_HHMMSS>/
    import time as _time
    _stamp = _time.strftime("%Y%m%d_%H%M%S")
    _ckstep = "unknown"
    for _p in str(args.ckpt).split("/"):
        if _p.startswith("checkpoint_step_"):
            _ckstep = _p.replace("checkpoint_", "")
    _mode = ("gentac" if args.gen_tactile else "video" if args.generate
             else "tacdec" if args.tactile_decode else "deccheck" if args.decode_check else "consist")
    args.out_dir = os.path.join(args.out_dir, f"{_ckstep}_{_mode}_{_stamp}")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[render] output dir: {args.out_dir}")

    trainer = Trainer(config, inference_only=True)
    if not (args.decode_check or args.tactile_decode):
        print(f"[render] loading MoT checkpoint: {args.ckpt}")
        trainer.transformer = load_mot_checkpoint(args.ckpt, torch_dtype=torch.bfloat16, torch_device=dev)
        trainer.transformer.eval()
        print("[render] expert structure:",
              {n: (trainer.transformer.mot.experts[n].hidden_dim,
                   trainer.transformer.mot.experts[n].narrow)
               for n in trainer.transformer.mot.expert_names})

    ds = MultiLatentLeRobotDataset(config=config)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    it = iter(dl)

    if args.tactile_decode:
        # Try to decode the GlobalTactile latent (per-sensor, WAN-VAE encoded 48ch
        # residual) back to RGB — user wasn't sure it's decodable.
        from models.utils import load_vae
        vae = load_vae(args.vae, torch_dtype=torch.bfloat16, torch_device=dev)
        batch = next(it)
        batch = trainer.convert_input_format(batch)
        gt = batch.get('tactile_global_latent')
        if gt is None:
            print("[tactile] no tactile_global_latent in batch"); return
        if gt.dim() == 5:
            gt = gt.unsqueeze(0)
        B, S, C, F, H, W = gt.shape
        print(f"[tactile] global tactile latent shape: {tuple(gt.shape)} (B,S,C,F,H,W)")
        for s in range(S):
            vid = decode_video_latent(vae, gt[:, s].to(dev))   # [B,48,F,H,W] -> RGB
            save_frames(vid, args.out_dir, f"tactile_s{s}")
        print("TACTILE DECODE DONE")
        return

    if args.decode_check:
        # Stage 0 (lowest risk): decode the TRUE training video latent -> RGB, with
        # NO model / NO sampling. Validates the VAE decode + view layout in isolation.
        from models.utils import load_vae
        vae = load_vae(args.vae, torch_dtype=torch.bfloat16, torch_device=dev)
        batch = next(it)
        batch = trainer.convert_input_format(batch)
        input_dict = trainer._prepare_input_dict(batch)
        x0 = input_dict['latent_dict']['latent']                 # [B, z, F, H, W] clean
        print(f"[decode-check] true video latent shape: {tuple(x0.shape)}")
        vid = decode_video_latent(vae, x0)
        save_frames(vid, args.out_dir, "true")
        print("DECODE CHECK DONE")
        return

    if args.generate:
        from models.utils import load_vae
        vae = load_vae(args.vae, torch_dtype=torch.bfloat16, torch_device=dev)
        for g in range(args.n_gen):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            batch = trainer.convert_input_format(batch)
            if args.ar:
                x_gen, x0_true = sample_video_ar(trainer, batch, n_steps=args.n_steps)
                tag = f"video_gen{g:02d}_ar"
            else:
                x_gen, x0_true = sample_video(trainer, batch, n_steps=args.n_steps,
                                              teacher_force=args.teacher_force)
                tag = (f"video_gen{g:02d}_tf" if args.teacher_force else f"video_gen{g:02d}_free")
            mse = torch.nn.functional.mse_loss(x_gen.float(), x0_true.float()).item()
            # left = generated video, right = ground-truth video, caption on top
            save_compare_video(
                decode_video_latent(vae, x_gen),
                decode_video_latent(vae, x0_true),
                args.out_dir, tag, fps=args.fps,
                left_label="generated", right_label="ground truth",
                title=f"VIDEO  latentMSE={mse:.3f}")
            print(f"[generate {g}] {tag} vs true latent MSE = {mse:.4f}")
        print("GENERATE DONE")
        return

    if args.gen_tactile:
        from models.utils import load_vae
        vae = load_vae(args.vae, torch_dtype=torch.bfloat16, torch_device=dev)
        for g in range(args.n_gen):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            batch = trainer.convert_input_format(batch)
            if batch.get('tactile_global_latent') is None:
                print("[gen-tactile] batch has no tactile_global_latent; skip"); continue
            gt_gen, gt_true = sample_tactile_ar(trainer, batch, n_steps=args.n_steps)
            # gt_*: (B,S,C,F,H,W). Decode each sensor; stack sensors vertically.
            B, S, C, F, H, W = gt_true.shape
            import torch as _t
            mse = _t.nn.functional.mse_loss(gt_gen.float(), gt_true.float()).item()
            for s in range(S):
                vid_gen = decode_video_latent(vae, gt_gen[:, s].to(dev))    # gen residual
                vid_true = decode_video_latent(vae, gt_true[:, s].to(dev))  # true residual
                save_compare_video(
                    vid_gen, vid_true, args.out_dir, f"tactile_gen{g:02d}_s{s}", fps=args.fps,
                    left_label="generated", right_label="true global-residual",
                    title=f"TACTILE global s{s}  latentMSE={mse:.3f}")
            print(f"[gen-tactile {g}] generated vs true GlobalTactile latent MSE = {mse:.4f}")
        print("GEN TACTILE DONE")
        return

    agg = {"latent": [], "action": [], "tactile": [], "vrec": []}
    for i in range(args.n_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        batch = trainer.convert_input_format(batch)
        input_dict = trainer._prepare_input_dict(batch)
        with torch.no_grad():
            pred = trainer.transformer(input_dict, train_mode=True)
            loss_dict = trainer.compute_loss(input_dict, pred)

            # video velocity -> reconstruct x0_hat = x_noisy - sigma * v, compare to clean x0
            video_pred = pred[0]                                  # [B, N_tok, C_patch]
            from utils import data_seq_to_patch
            ld = input_dict['latent_dict']
            x0 = ld['latent'].float()                             # [B,C,F,H,W] clean
            xN = ld['noisy_latents'].float()                      # [B,C,F,H,W] noised
            v = data_seq_to_patch(trainer.patch_size, video_pred,
                                  x0.shape[-3], x0.shape[-2], x0.shape[-1],
                                  batch_size=x0.shape[0]).float()
            sig = _sigma_of(trainer.train_scheduler_latent,
                            ld['timesteps'].flatten()).to(x0.device)   # [B*F]
            sig = sig.view(x0.shape[0], x0.shape[2], 1, 1).permute(0, 2, 1, 3)[..., None]  # [B,1,F,1,1]
            x0_hat = xN - sig * v
            vrec = torch.nn.functional.mse_loss(x0_hat, x0).item()

        la = loss_dict['latent_loss'].item()
        ac = loss_dict.get('action_loss', torch.zeros(())).item()
        ta = loss_dict.get('tactile_loss', torch.zeros(())).item()
        agg["latent"].append(la); agg["action"].append(ac)
        agg["tactile"].append(ta); agg["vrec"].append(vrec)
        print(f"[batch {i}] latent_loss={la:.4f} action_loss={ac:.4f} "
              f"tactile_loss={ta:.4f} | video x0-reconstruction MSE={vrec:.4f}")

    n = len(agg["latent"])
    print("\n==== MEAN over %d batches ====" % n)
    for k in ("latent", "action", "tactile", "vrec"):
        print(f"  {k}: {sum(agg[k])/n:.4f}")
    print("CONSISTENCY CHECK DONE (compare to the ~step-400 training losses; "
          "low = inference computes the same function as training)")


if __name__ == "__main__":
    main()
