#!/usr/bin/env python3
"""Encode LeRobot tactile episodes into GlobalTactile / LocalTactile latents.

For each tactile sensor video:
  1. Decode raw RGB frames (any resolution; fps read from meta/info.json),
     downsample fps and resize to target resolution (default 128x128).
  2. Build two tactile streams in RGB pixel space:
       - GlobalTactile: frame[t] - frame[0]  — residual against the first frame.
         The model *predicts* this stream as a diffusion target.
       - LocalTactile : frame[t]             — the current observed tactile frame
         itself (absolute, streaming-robust), fed to the model as an input
         observation. (A legacy 'residual' form frame[t]-frame[t-1] also exists but
         is deprecated.)
  3. Normalize each stream to [-1, 1] and encode with the Wan2.2 VAE.
  4. Save one `.pth` per (episode, tactile_key, mode) under `latents_tactile/`.

The output layout mirrors `encode_lerobot_n0_latents.py`:
    <dataset>/latents_tactile/global/chunk-XXX/observation.images.tactile_*/episode_*_*_*.pth
    <dataset>/latents_tactile/local /chunk-XXX/observation.images.tactile_*/episode_*_*_*.pth

Usage (recommended):
    python script/encode_tactile_latent.py \
        --dataset-root /path/to/your_lerobot_repo \
        --model-path /path/to/base-model \
        --height 128 --width 128 \
        --target-fps 10 \
        --tactile-keys observation.images.tactile_a observation.images.tactile_b \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio_ffmpeg as ffmpeg
import numpy as np
import pandas as pd
import torch
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.models.utils import load_vae


@dataclass
class EpisodeRequest:
    episode_index: int
    length: int
    source_video: Path
    output_paths: dict        # {"global": Path, "local": Path}
    start_timestamp: float
    end_timestamp: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path containing 'vae/' subdir (use base-model)")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="default: <dataset-root>/latents_tactile")
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--height", type=int, default=128,
                        help="must be multiple of 32 (VAE constraint)")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--tactile-keys", nargs="+", required=True,
                        help="e.g. observation.images.tactile_a observation.images.tactile_b")
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode", choices=["global", "local", "both"], default="both",
                        help="which residual streams to compute")
    parser.add_argument("--local-mode", choices=["current", "residual"], default="current",
                        help="local stream: 'current'=the observed frame itself (default, "
                             "absolute, streaming-robust; input observation for the "
                             "use_local_tactile post-train branch); 'residual'="
                             "frame[t]-frame[t-1] (legacy, fragile under streaming)")
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_info(dataset_root: Path) -> dict:
    with (dataset_root / "meta" / "info.json").open() as f:
        return json.load(f)


def load_episode_table(dataset_root: Path) -> pd.DataFrame:
    episode_files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if episode_files:
        return pd.concat([pd.read_parquet(p) for p in episode_files], ignore_index=True)
    jsonl_path = dataset_root / "meta" / "episodes.jsonl"
    if jsonl_path.exists():
        records = [json.loads(line) for line in jsonl_path.read_text().strip().split("\n")]
        return pd.DataFrame(records)
    raise FileNotFoundError(f"No episode metadata under {dataset_root / 'meta'}")


def build_source_video(dataset_root: Path, video_key: str, chunk_index: int, file_index: int) -> Path:
    return (dataset_root / "videos" / video_key
            / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4")


def normalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=latents.device).view(1, -1, 1, 1, 1)
    return ((latents.float() - latents_mean) * (1.0 / latents_std)).to(latents.dtype)


def extract_frames(
    request: EpisodeRequest, width: int, height: int, target_fps: int, ori_fps: int,
) -> tuple[np.ndarray, list[int]]:
    """Decode + downsample + resize. Returns (frames uint8 (N,H,W,3), local_frame_ids)."""
    ffmpeg_bin = ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin, "-v", "error",
        "-ss", str(request.start_timestamp),
        "-to", str(request.end_timestamp),
        "-i", str(request.source_video),
        "-vf", f"fps={target_fps},scale={width}:{height}:flags=area",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    frame_bytes = width * height * 3
    if len(raw) % frame_bytes != 0:
        raise RuntimeError(f"ffmpeg output size mismatch for {request.source_video}")
    num = len(raw) // frame_bytes
    if num <= 0:
        raise RuntimeError(f"no frames decoded from {request.source_video}")
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(num, height, width, 3).copy()
    local_frame_ids = [
        min(request.length - 1, int(round(i * ori_fps / target_fps)))
        for i in range(num)
    ]
    # Wan VAE wants F % 4 == 1 for clean temporal compression
    while len(frames) > 1 and len(frames) % 4 != 1:
        frames = frames[:-1]
        local_frame_ids.pop()
    if len(frames) <= 0:
        raise RuntimeError(f"no valid Wan frame sequence for {request.source_video}")
    return frames, local_frame_ids


def compute_residuals(frames_u8: np.ndarray, mode: str,
                      local_mode: str = "current") -> dict[str, np.ndarray]:
    """Compute global and/or local tactile streams in float32, normalized to [-1, 1].

    frames_u8: (N, H, W, 3) uint8 in [0, 255]
    Returns dict { "global": (N, H, W, 3) float32 in [-1, 1], "local": ... }

    local_mode:
      - "residual" (legacy): local[t] = frame[t] - frame[t-1] (high-freq inter-frame
        delta). Fragile under streaming inference (needs the contiguous prev frame).
      - "current": local[t] = the current frame itself, mapped [0,255]->[-1,1]. The
        CURRENT tactile latent (absolute), not a delta — robust to streaming (server
        just encodes the current frame, no prev_frames bookkeeping). Pair with the
        use_local_tactile post-train switch.
    """
    # int16 to avoid uint8 wrap-around in subtraction
    frames_i16 = frames_u8.astype(np.int16)
    out: dict[str, np.ndarray] = {}
    if mode in ("global", "both"):
        # global[t] = frame[t] - frame[0]   ∈ [-255, 255]
        # normalize by /255 → [-1, 1]
        residual = (frames_i16 - frames_i16[0:1]).astype(np.float32) / 255.0
        out["global"] = residual
    if mode in ("local", "both"):
        if local_mode == "current":
            # local[t] = current frame -> [-1, 1] (VAE input range), no delta.
            out["local"] = frames_u8.astype(np.float32) / 127.5 - 1.0
        else:
            # local[t] = frame[t] - frame[t-1]; first frame residual = 0
            residual = np.zeros_like(frames_i16, dtype=np.float32)
            if frames_i16.shape[0] > 1:
                residual[1:] = (frames_i16[1:] - frames_i16[:-1]).astype(np.float32) / 255.0
            out["local"] = residual
    return out


def encode_residual_to_latent(
    residual_frames: np.ndarray, vae, device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, int, int, int]:
    """Encode a (N, H, W, 3) residual array to latent.

    residual_frames values are already in [-1, 1] (same range Wan VAE expects).
    Returns (flat_latent (F*H*W, C), latent_F, latent_H, latent_W).
    """
    # Convert to (C, N, H, W) → unsqueeze batch → (1, C, N, H, W)
    video = torch.from_numpy(residual_frames).permute(3, 0, 1, 2).unsqueeze(0)
    # already in [-1, 1] range from compute_residuals, no further normalize needed
    with torch.no_grad():
        posterior = vae.encode(video.to(device=device, dtype=dtype)).latent_dist
        latents = normalize_latents(posterior.mean, vae)[0]   # (C, F, H, W)
    lat_F, lat_H, lat_W = latents.shape[1:]
    flat = rearrange(latents, "c f h w -> (f h w) c").contiguous().cpu()
    return flat, lat_F, lat_H, lat_W


def build_requests(
    dataset_root: Path, output_root: Path, episodes_df: pd.DataFrame, tactile_key: str,
    info: dict, is_v21: bool, modes: list[str],
) -> list[EpisodeRequest]:
    requests = []
    chunks_size = info.get("chunks_size", 1000)
    total_chunks = int(info.get("total_chunks", 1))
    ori_fps = int(round(info["fps"]))
    for row in episodes_df.to_dict(orient="records"):
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        if is_v21:
            # single-chunk repos (converter -> chunk-000, total_chunks=1) must
            # use chunk-000; naive idx//size sends ep>=1000 to a missing
            # chunk-001 and crashes (the giant-repo "wall"). See video encoder.
            chunk_index = 0 if total_chunks <= 1 else episode_index // chunks_size
            source_video = (
                dataset_root / "videos" / f"chunk-{chunk_index:03d}"
                / tactile_key / f"episode_{episode_index:06d}.mp4"
            )
            start_ts = 0.0
            end_ts = length / ori_fps
        else:
            chunk_index = int(row[f"videos/{tactile_key}/chunk_index"])
            file_index = int(row[f"videos/{tactile_key}/file_index"])
            source_video = build_source_video(dataset_root, tactile_key, chunk_index, file_index)
            start_ts = float(row[f"videos/{tactile_key}/from_timestamp"])
            end_ts = float(row[f"videos/{tactile_key}/to_timestamp"])

        output_paths = {
            mode: (
                output_root / mode / f"chunk-{chunk_index:03d}" / tactile_key
                / f"episode_{episode_index:06d}_0_{length}.pth"
            )
            for mode in modes
        }
        for p in output_paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        requests.append(EpisodeRequest(
            episode_index=episode_index, length=length,
            source_video=source_video, output_paths=output_paths,
            start_timestamp=start_ts, end_timestamp=end_ts,
        ))
    return requests


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    model_path = args.model_path.resolve()
    output_root = (args.output_root or (dataset_root / "latents_tactile")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError("Height and width must both be multiples of 32 (Wan VAE constraint)")

    info = load_info(dataset_root)
    ori_fps = int(round(info["fps"]))

    # Filter episodes
    episodes_df = load_episode_table(dataset_root)
    if args.episodes is not None:
        episodes_df = episodes_df[episodes_df["episode_index"].isin(args.episodes)].copy()
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)
    if episodes_df.empty:
        raise ValueError("No episodes selected")

    # Detect dataset format
    first_key = args.tactile_keys[0]
    is_v21 = f"videos/{first_key}/chunk_index" not in episodes_df.columns

    # Set up VAE
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = get_dtype(args.dtype)
    vae = load_vae(model_path / "vae", torch_dtype=dtype, torch_device=device)

    modes = ["global", "local"] if args.mode == "both" else [args.mode]

    # Process each tactile key
    for tactile_key in args.tactile_keys:
        print(f"\n=== Processing tactile_key={tactile_key} ===")
        requests = build_requests(
            dataset_root, output_root, episodes_df, tactile_key, info, is_v21, modes
        )
        for req in requests:
            # Skip if all outputs exist and not overwriting
            if not args.overwrite and all(p.exists() for p in req.output_paths.values()):
                print(f"  [skip] episode {req.episode_index} (all modes present)")
                continue

            if not req.source_video.exists():
                print(f"  [warn] missing source video {req.source_video}")
                continue

            # 1. extract frames
            frames_u8, local_frame_ids = extract_frames(
                req, args.width, args.height, args.target_fps, ori_fps,
            )

            # 2. compute residuals
            residuals = compute_residuals(frames_u8, args.mode, local_mode=args.local_mode)

            # 3. encode each residual stream
            for mode, residual_arr in residuals.items():
                out_path = req.output_paths[mode]
                if not args.overwrite and out_path.exists():
                    continue
                flat_latent, lat_F, lat_H, lat_W = encode_residual_to_latent(
                    residual_arr, vae, device, dtype
                )
                payload = {
                    "latent": flat_latent.to(torch.bfloat16),
                    "latent_num_frames": int(lat_F),
                    "latent_height": int(lat_H),
                    "latent_width": int(lat_W),
                    "frame_ids": local_frame_ids,
                    "start_frame": 0,
                    "end_frame": int(req.length),
                    "fps": int(args.target_fps),
                    "ori_fps": int(ori_fps),
                    "tactile_residual_mode": mode,
                    "tactile_resize_h": int(args.height),
                    "tactile_resize_w": int(args.width),
                }
                # atomic write (see video encoder): temp + rename so a killed
                # process never leaves a truncated *.pth treated as complete.
                _tmp = out_path.parent / (out_path.name + f".tmp.{os.getpid()}")
                torch.save(payload, _tmp)
                os.replace(_tmp, out_path)
                print(f"  [ok] episode {req.episode_index} {mode} -> {out_path.name} "
                      f"latent shape=({lat_F},{lat_H},{lat_W})")

    print(f"\nDone. Outputs under: {output_root}")


if __name__ == "__main__":
    main()
