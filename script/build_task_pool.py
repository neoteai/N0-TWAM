# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Build a single-task post-training pool: train/val symlinks + action norm stats.

  <pool>/{train,val}/<repo> -> symlinks to --data
  <pool>/norm_stat_absee.json (--mode absee) | norm_stat.json (--mode delta)

Windows are sampled on the same latent-anchor grid the training dataset uses;
the two mode outputs are distinct files and must never overwrite each other.
Norm rules: true per-channel q01/q99, gripper lower = data min, dead channels
floored to span 1e-3, unused dims [-1, 1].
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HORIZON = 12
DELTA_IDS_20 = list(range(0, 9)) + list(range(10, 19))
DEAD_SPAN = 1e-4
EPS_HALF = 5e-4


def collect_samples(root: Path, anchor_cam: str, mode: str):
    """Sample per-anchor action windows exactly as the dataset builds them
    (window = frame stride x 4; 12 for the standard 30fps/stride-3 encode)."""
    samples, episodes, used_dim = [], 0, None
    windows = set()
    latent_files = sorted((root / "latents").glob(f"chunk-*/{anchor_cam}/episode_*.pth"))
    assert latent_files, (f"no encoded latents under {root}/latents/chunk-*/{anchor_cam} "
                          "— run the encoders first")
    for lp in latent_files:
        m = re.search(r"episode_(\d{6})", lp.name)
        if not m:
            continue
        ep = int(m.group(1))
        pq = sorted(root.glob(f"data/chunk-*/episode_{ep:06d}.parquet"))
        if not pq:
            print(f"  skip episode {ep}: no parquet found")
            continue
        df = pd.read_parquet(pq[0])
        action = np.stack([np.asarray(x, dtype=np.float32) for x in df["action"].to_numpy()])
        state = np.stack([np.asarray(x, dtype=np.float32)
                          for x in df["observation.state"].to_numpy()])
        seq_len = min(len(action), len(state))
        action, state = action[:seq_len], state[:seq_len]
        used_dim = action.shape[1] if used_dim is None else used_dim

        meta = torch.load(lp, map_location="cpu")
        frame_ids = np.asarray(meta["frame_ids"], dtype=np.int64)
        latent_frame_num = int(meta.get("latent_num_frames", (len(frame_ids) - 1) // 4 + 1))
        start = int(meta.get("start_frame", 0))
        anchor_ids = np.asarray(
            [frame_ids[min(i * 4, len(frame_ids) - 1)] for i in range(latent_frame_num)],
            dtype=np.int64,
        )
        stride = int(frame_ids[1] - frame_ids[0]) if len(frame_ids) > 1 else 3
        window = max(1, stride * 4)      # raw actions per latent frame
        windows.add(window)

        for frame_idx in range(1, latent_frame_num):
            anchor = int(anchor_ids[frame_idx - 1] - start)
            if anchor < 0 or anchor >= seq_len:
                continue
            valid = min(window, seq_len - anchor)
            if valid <= 0:
                continue
            target = action[anchor:anchor + valid].copy()
            if mode == "delta":
                delta_ids = [i for i in DELTA_IDS_20 if i < target.shape[1]]
                if delta_ids:
                    target[:, delta_ids] -= state[anchor, delta_ids]
            out = np.zeros((valid, 20), dtype=np.float32)
            width = min(target.shape[1], 20)
            out[:, :width] = target[:, :width]
            samples.append(out)
        episodes += 1
    if not samples:
        raise RuntimeError(f"no norm samples from {root}")
    window = max(windows) if windows else HORIZON
    if windows != {HORIZON}:
        print(f"WARNING: derived action window(s) {sorted(windows)} != {HORIZON} — "
              "twam_posttrain_cfg pins action_per_frame/horizon to 12; re-encode at "
              "stride 3 (30fps -> 10fps) or adjust the config consciously.")
    return np.concatenate(samples, axis=0), episodes, int(used_dim or 10), window


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="LeRobot v2.1 repo (latents encoded)")
    ap.add_argument("--pool", type=Path, required=True, help="pool dir to create")
    ap.add_argument("--mode", choices=["absee", "delta"], default="absee")
    ap.add_argument("--dual-arm", action="store_true",
                    help="20-dim dual-arm repo (default: single-arm, first 10 dims)")
    ap.add_argument("--anchor-cam", default=None,
                    help="camera key whose latents define the anchor grid "
                         "(default: first non-tactile video key in meta/info.json)")
    args = ap.parse_args()

    # resolve() is load-bearing: symlink_to() writes the given path verbatim, so a
    # relative dataset root would create dangling train/<repo> links (num_samples=0).
    root = args.data.resolve()
    pool = args.pool.resolve()
    repo = root.name

    features = json.loads((root / "meta" / "info.json").read_text())["features"]
    video_keys = [k for k, v in features.items()
                  if k.startswith("observation.images.") and "tactile" not in k
                  and v.get("dtype") == "video"]
    anchor_cam = args.anchor_cam or video_keys[0]

    for split in ("train", "val"):
        d = pool / split
        d.mkdir(parents=True, exist_ok=True)
        link = d / repo
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(root)

    arr, episodes, source_action_dim, window = collect_samples(root, anchor_cam, args.mode)
    source_action_dim = min(source_action_dim, 20)
    used = 20 if args.dual_arm else min(source_action_dim, 10)
    if source_action_dim > 10 and not args.dual_arm:
        print(f"WARNING: source action dim is {source_action_dim} (>10) but --dual-arm "
              "is not set — stats for dims 10-19 (arm B) will be neutral [-1,1]. "
              "If this repo is dual-arm you are about to lose arm B; re-run with --dual-arm.")

    q01_raw = np.full(20, -1.0, dtype=np.float64)
    q99_raw = np.full(20, 1.0, dtype=np.float64)
    q01_raw[:source_action_dim] = np.percentile(arr[:, :source_action_dim], 1, axis=0)
    q99_raw[:source_action_dim] = np.percentile(arr[:, :source_action_dim], 99, axis=0)
    min_raw = np.full(20, -1.0, dtype=np.float64)
    min_raw[:source_action_dim] = arr[:, :source_action_dim].min(axis=0)

    q01, q99 = q01_raw.copy(), q99_raw.copy()

    gripper_ids = [g for g in ((9, 19) if args.dual_arm else (9,)) if g < used]
    for g in gripper_ids:
        q01[g] = min_raw[g]

    floored = []
    for ch in range(used):
        if float(q99[ch] - q01[ch]) < DEAD_SPAN:
            center = 0.5 * float(q01[ch] + q99[ch])
            q01[ch], q99[ch] = center - EPS_HALF, center + EPS_HALF
            floored.append(ch)
    q01[used:], q99[used:] = -1.0, 1.0

    # dual-arm sanity: an all-dead arm-B usually means the converter dropped it
    arm_b_ids = [ch for ch in range(10, 19) if ch < used]
    if (args.dual_arm and args.mode == "delta" and arm_b_ids
            and all(float(q99_raw[ch] - q01_raw[ch]) < DEAD_SPAN for ch in arm_b_ids)):
        print("WARNING ARM_B_FROZEN_SUSPECT: all arm-B delta channels (10-18) are "
              "dead — check the conversion, this usually is a data bug, not a real "
              "frozen arm.")

    stem = "norm_stat_absee" if args.mode == "absee" else "norm_stat"
    norm = {
        "q01": [float(x) for x in q01],
        "q99": [float(x) for x in q99],
        "note": (
            f"mode={args.mode}, horizon={window}; per-channel true q01/q99, no "
            f"global min_span; gripper lower = data min {gripper_ids}; dead "
            f"channels floored to span 1e-3: {floored}; unused dims neutral [-1,1]."
        ),
    }
    pool.mkdir(parents=True, exist_ok=True)
    (pool / f"{stem}.json").write_text(json.dumps(norm, indent=2))
    (pool / f"{stem}_per_robot.json").write_text(
        json.dumps({repo: {"q01": norm["q01"], "q99": norm["q99"]}}, indent=2))
    report = {
        "mode": args.mode,
        "q01_raw": [float(x) for x in q01_raw],
        "q99_raw": [float(x) for x in q99_raw],
        "min_raw": [float(x) for x in min_raw],
        "samples": int(arr.shape[0]),
        "episodes": episodes,
        "horizon": window,
        "gripper_ids": gripper_ids,
        "floored_channels": floored,
        "source_action_dim": source_action_dim,
        "used_dim": used,
        "anchor_cam": anchor_cam,
    }
    (pool / f"{stem}_raw_report.json").write_text(json.dumps(report, indent=2))

    print("pool", pool)
    print("episodes", episodes, "samples", arr.shape[0], "used_dim", used)
    print(f"{args.mode} q01[:10]", [round(x, 4) for x in norm["q01"][:10]])
    print(f"{args.mode} q99[:10]", [round(x, 4) for x in norm["q99"][:10]])
    print("floored_channels", floored)


if __name__ == "__main__":
    main()
