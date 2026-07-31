# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Assemble a serve bundle: symlink the trained transformer/ next to the base
model's vae/ tokenizer/ text_encoder/ (+ assets, empty_emb.pt when present).
The transformer link points INTO the checkpoint dir so the server finds the
train_meta.json consistency snapshot.

Usage:
  python script/make_serve_bundle.py --checkpoint .../checkpoint_step_2000 \
      --base /path/to/base-model --bundle /path/to/serve-bundle
"""
import argparse
from pathlib import Path


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())
    print(f"  {dst.name:14s} -> {src.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="training checkpoint dir (contains transformer/)")
    ap.add_argument("--base", type=Path, required=True,
                    help="base model dir (contains vae/ tokenizer/ text_encoder/)")
    ap.add_argument("--bundle", type=Path, required=True, help="bundle dir to create")
    args = ap.parse_args()

    ckpt_tf = args.checkpoint.resolve() / "transformer"
    assert ckpt_tf.is_dir(), f"no transformer/ under {args.checkpoint}"
    for part in ("vae", "tokenizer", "text_encoder"):
        assert (args.base / part).is_dir(), f"no {part}/ under {args.base}"
    if not (args.checkpoint / "train_meta.json").is_file():
        print("note: checkpoint has no train_meta.json — the server will skip "
              "the train/serve consistency cross-check (pre-release checkpoints).")

    args.bundle.mkdir(parents=True, exist_ok=True)
    print(f"bundle {args.bundle.resolve()}:")
    link(ckpt_tf, args.bundle / "transformer")
    for part in ("vae", "tokenizer", "text_encoder", "assets"):
        if (args.base / part).is_dir():
            link(args.base / part, args.bundle / part)
    for extra in ("empty_emb.pt", "norm_stat_pretrain.json"):
        if (args.base / extra).is_file():
            link(args.base / extra, args.bundle / extra)


if __name__ == "__main__":
    main()
