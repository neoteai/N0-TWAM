#!/usr/bin/env python3
"""Encode LeRobot video episodes into Wan2.2 VAE latents.

It reads the LeRobot v2.1 episode parquet metadata, samples/resizes frames for each episode,
encodes the sampled clip with the Wan2.2 VAE, and saves one `.pth` file per
episode/view under `latents/`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import imageio_ffmpeg as ffmpeg
import numpy as np
import pandas as pd
import torch
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.models.utils import load_text_encoder, load_tokenizer, load_vae


@dataclass
class EpisodeRequest:
    episode_index: int
    length: int
    prompt: str
    source_video: Path
    output_path: Path
    start_timestamp: float
    end_timestamp: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--video-keys", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-episodes-jsonl", action="store_true")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override the text prompt for all episodes instead of using the task string from the dataset.",
    )
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def load_info(dataset_root: Path) -> dict:
    with (dataset_root / "meta" / "info.json").open() as f:
        return json.load(f)


def get_video_keys(info: dict) -> list[str]:
    # RGB cameras only — tactile streams are encoded by encode_tactile_latent.py
    # with their own global/local semantics, not the RGB pipeline.
    return [
        key
        for key, value in info["features"].items()
        if isinstance(value, dict) and value.get("dtype") in ("video", "image")
        and "image" in key and "tactile" not in key
    ]


def resize_frame(frame: torch.Tensor | None, width: int, height: int):
    if frame is None:
        raise ValueError("Missing frame")
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def build_source_video(dataset_root: Path, video_key: str, chunk_index: int, file_index: int) -> Path:
    return (
        dataset_root
        / "videos"
        / video_key
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )


def load_episode_table(dataset_root: Path) -> pd.DataFrame:
    # Try v3.0 parquet format first
    episode_files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if episode_files:
        df = pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)
        # Some converters write the v3 episodes parquet without a `tasks`
        # column; recover it from the v2.1 episodes.jsonl when present so the
        # per-episode prompt lookup below keeps working (else use --prompt).
        jsonl_path = dataset_root / "meta" / "episodes.jsonl"
        if "tasks" not in df.columns and jsonl_path.exists():
            records = [json.loads(line) for line in jsonl_path.read_text().strip().split("\n")]
            tasks_by_ep = {r["episode_index"]: r.get("tasks") for r in records}
            df["tasks"] = df["episode_index"].map(tasks_by_ep)
        return df
    # Fallback to v2.1 jsonl format
    jsonl_path = dataset_root / "meta" / "episodes.jsonl"
    if jsonl_path.exists():
        records = [json.loads(line) for line in jsonl_path.read_text().strip().split("\n")]
        return pd.DataFrame(records)
    raise FileNotFoundError(f"No episode metadata found under {dataset_root / 'meta'}")


def clean_prompt(tasks) -> str:
    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    if isinstance(tasks, tuple) and tasks:
        return str(tasks[0])
    if isinstance(tasks, str):
        return tasks
    return ""


def get_text_emb(
    prompt: str,
    tokenizer,
    text_encoder,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
    cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    prompt = prompt_clean(prompt)
    if prompt in cache:
        return cache[prompt]

    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    mask = text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    enc_device = next(text_encoder.parameters()).device
    prompt_embeds = text_encoder(
        text_input_ids.to(enc_device),
        mask.to(enc_device),
    ).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )[0]
    cache[prompt] = prompt_embeds.cpu()
    return cache[prompt]


def normalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=latents.device).view(1, -1, 1, 1, 1)
    return ((latents.float() - latents_mean) * (1.0 / latents_std)).to(latents.dtype)


def ensure_complete(frames: Iterable) -> None:
    if any(frame is None for frame in frames):
        raise RuntimeError("Missing decoded frame while collecting sampled video frames")


def extract_episode_frames(
    request: EpisodeRequest,
    width: int,
    height: int,
    target_fps: int,
    ori_fps: int,
) -> tuple[list[np.ndarray], list[int]]:
    ffmpeg_bin = ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-ss",
        str(request.start_timestamp),
        "-to",
        str(request.end_timestamp),
        "-i",
        str(request.source_video),
        "-vf",
        f"fps={target_fps},scale={width}:{height}:flags=area",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    frame_bytes = width * height * 3
    if len(raw) % frame_bytes != 0:
        raise RuntimeError(
            f"Unexpected ffmpeg output size for {request.source_video}: {len(raw)} bytes"
        )

    num_frames = len(raw) // frame_bytes
    if num_frames <= 0:
        raise RuntimeError(f"No frames decoded for {request.source_video}")

    frames = np.frombuffer(raw, dtype=np.uint8).reshape(num_frames, height, width, 3).copy()
    local_frame_ids = [
        min(request.length - 1, int(round(i * ori_fps / target_fps)))
        for i in range(num_frames)
    ]

    while len(frames) > 1 and len(frames) % 4 != 1:
        frames = frames[:-1]
        local_frame_ids.pop()

    if len(frames) <= 0:
        raise RuntimeError(f"No valid Wan frame sequence remained for {request.source_video}")

    return list(frames), local_frame_ids


def encode_video(
    frames: list,
    vae,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int, int, int]:
    ensure_complete(frames)
    video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
    video = video.to(torch.float32) / 127.5 - 1.0
    with torch.no_grad():
        posterior = vae.encode(video.to(device=device, dtype=dtype)).latent_dist
        latents = normalize_latents(posterior.mean, vae)[0]
    latent_num_frames, latent_height, latent_width = latents.shape[1:]
    flat_latent = rearrange(latents, "c f h w -> (f h w) c").contiguous().cpu()
    return flat_latent, latent_num_frames, latent_height, latent_width


def maybe_write_episodes_jsonl(dataset_root: Path, episodes_df: pd.DataFrame) -> None:
    output_path = dataset_root / "meta" / "episodes.jsonl"
    lines = []
    for row in episodes_df.to_dict(orient="records"):
        prompt = clean_prompt(row["tasks"])
        item = {
            "episode_index": int(row["episode_index"]),
            "tasks": list(row["tasks"]),
            "length": int(row["length"]),
            "action_config": [
                {
                    "start_frame": 0,
                    "end_frame": int(row["length"]),
                    "action_text": prompt,
                }
            ],
        }
        lines.append(json.dumps(item, ensure_ascii=True))
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    model_path = args.model_path.resolve()
    output_root = (args.output_root or (dataset_root / "latents")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError("Height and width must both be multiples of 32")

    info = load_info(dataset_root)
    ori_fps = int(round(info["fps"]))
    all_video_keys = get_video_keys(info)
    video_keys = args.video_keys or all_video_keys

    missing_keys = sorted(set(video_keys) - set(all_video_keys))
    if missing_keys:
        raise ValueError(f"Unknown video keys: {missing_keys}")

    episodes_df = load_episode_table(dataset_root)
    if args.episodes is not None:
        episodes_df = episodes_df[episodes_df["episode_index"].isin(args.episodes)].copy()
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)
    if episodes_df.empty:
        raise ValueError("No episodes selected")

    if args.write_episodes_jsonl:
        if args.episodes is not None:
            # episodes_df is filtered here — rewriting the jsonl from it would
            # silently DROP every unselected episode from the dataset metadata.
            raise ValueError("--write-episodes-jsonl cannot be combined with "
                             "--episodes (it would rewrite meta/episodes.jsonl "
                             "with only the selected subset)")
        maybe_write_episodes_jsonl(dataset_root, episodes_df)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = get_dtype(args.dtype)
    vae = load_vae(model_path / "vae", torch_dtype=dtype, torch_device=device)
    tokenizer = load_tokenizer(model_path / "tokenizer")
    text_encoder = load_text_encoder(
        model_path / "text_encoder",
        torch_dtype=dtype,
        torch_device="cpu",
    )

    text_cache: dict[str, torch.Tensor] = {}
    # Detect dataset format: v3.0 has per-video columns, v2.1 uses per-episode videos
    is_v21 = f"videos/{video_keys[0]}/chunk_index" not in episodes_df.columns

    for video_key in video_keys:
        requests: list[EpisodeRequest] = []
        for row in episodes_df.to_dict(orient="records"):
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            if args.prompt is not None:
                prompt = args.prompt
            elif "tasks" in row:
                prompt = clean_prompt(row["tasks"])
            else:
                raise KeyError(
                    "episode metadata has no `tasks` (v3 episodes parquet "
                    "without a tasks column and no meta/episodes.jsonl) — "
                    "pass --prompt \"<instruction>\" instead."
                )

            if is_v21:
                # v2.1: videos/chunk-{chunk}/{video_key}/episode_{idx}.mp4
                # Single-chunk repos (our converter writes ALL segments to
                # chunk-000 with total_chunks=1) must use chunk-000 — the
                # naive idx//chunks_size sends episode_index>=1000 to a
                # non-existent chunk-001 and the encode crashes (the giant-repo
                # "wall"). Honor total_chunks; fall back to idx//size otherwise.
                if int(info.get("total_chunks", 1)) <= 1:
                    chunk_index = 0
                else:
                    chunk_index = episode_index // int(info.get("chunks_size", 1000))
                source_video = (
                    dataset_root / "videos" / f"chunk-{chunk_index:03d}"
                    / video_key / f"episode_{episode_index:06d}.mp4"
                )
                start_timestamp = 0.0
                end_timestamp = length / ori_fps
            else:
                # v3.0: videos/{video_key}/chunk-{chunk}/file-{file}.mp4
                chunk_index = int(row[f"videos/{video_key}/chunk_index"])
                file_index = int(row[f"videos/{video_key}/file_index"])
                source_video = build_source_video(dataset_root, video_key, chunk_index, file_index)
                start_timestamp = float(row[f"videos/{video_key}/from_timestamp"])
                end_timestamp = float(row[f"videos/{video_key}/to_timestamp"])

            output_path = (
                output_root
                / f"chunk-{chunk_index:03d}"
                / video_key
                / f"episode_{episode_index:06d}_0_{length}.pth"
            )
            if output_path.exists() and not args.overwrite:
                continue

            requests.append(
                EpisodeRequest(
                    episode_index=episode_index,
                    length=length,
                    prompt=prompt,
                    source_video=source_video,
                    output_path=output_path,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                )
            )

        for request in requests:
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            frames, local_frame_ids = extract_episode_frames(
                request=request,
                width=args.width,
                height=args.height,
                target_fps=args.target_fps,
                ori_fps=ori_fps,
            )
            flat_latent, latent_num_frames, latent_height, latent_width = encode_video(
                frames=frames,
                vae=vae,
                device=device,
                dtype=dtype,
            )
            text_emb = get_text_emb(
                prompt=request.prompt,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                device=device,
                dtype=dtype,
                max_sequence_length=args.max_sequence_length,
                cache=text_cache,
            )
            payload = {
                "latent": flat_latent.to(torch.bfloat16),
                "latent_num_frames": int(latent_num_frames),
                "latent_height": int(latent_height),
                "latent_width": int(latent_width),
                "video_num_frames": int(len(local_frame_ids)),
                "video_height": int(args.height),
                "video_width": int(args.width),
                "text_emb": text_emb.to(torch.bfloat16),
                "text": request.prompt,
                "frame_ids": list(local_frame_ids),
                "start_frame": 0,
                "end_frame": int(request.length),
                "fps": int(args.target_fps),
                "ori_fps": int(ori_fps),
            }
            # atomic write: a killed process must never leave a truncated
            # *.pth that the skip-if-exists logic would treat as complete.
            # write to a per-pid temp (not matching *.pth), then atomic rename.
            _tmp = request.output_path.parent / (
                request.output_path.name + f".tmp.{os.getpid()}")
            torch.save(payload, _tmp)
            os.replace(_tmp, request.output_path)
            print(f"saved {request.output_path}")


if __name__ == "__main__":
    main()
