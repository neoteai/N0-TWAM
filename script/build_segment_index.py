"""Build a global, human-readable segment index over the encoded data tree.

Aggregates every repo's episodes.jsonl into <data>/segment_index.csv with one
row per segment:

    split,task,robot,episode_index,latent_stem,source_episode,source_session,
    source_start_frame,source_end_frame,length,instruction

So "which task / which raw episode / which raw frames does this latent belong
to" is a single grep:  grep episode_000001 segment_index.csv

Also usable as a lookup tool for one file:
    python build_segment_index.py --data <root> --lookup <path/to/latent.pth>
"""
import argparse
import csv
import json
from pathlib import Path


def iter_repos(data: Path):
    for ep_jsonl in sorted(data.glob("*/*/*/meta/episodes.jsonl")):
        repo = ep_jsonl.parent.parent          # .../split/task/robot
        yield repo.parent.parent.name, repo.parent.name, repo.name, repo, ep_jsonl


def build(data: Path) -> Path:
    out = data / "segment_index.csv"
    rows = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "task", "robot", "episode_index", "latent_stem",
                    "source_episode", "source_session",
                    "source_start_frame", "source_end_frame",
                    "length", "instruction"])
        for split, task, robot, repo, ep_jsonl in iter_repos(data):
            for line in ep_jsonl.read_text().splitlines():
                if not line:
                    continue
                e = json.loads(line)
                idx = e["episode_index"]
                sf = e.get("source_frames", [0, e["length"]])
                w.writerow([split, task, robot, idx,
                            f"episode_{idx:06d}_0_{e['length']}",
                            e.get("source_episode", "?"),
                            e.get("source_session", ""),
                            sf[0], sf[1], e["length"],
                            (e.get("tasks") or [""])[0]])
                rows += 1
    print(f"{rows} segments -> {out}")
    return out


def lookup(data: Path, latent_path: Path) -> None:
    # .../<split>/<task>/<robot>/latents*/chunk-XXX/<key>/episode_XXXXXX_S_E.pth
    parts = latent_path.resolve().parts
    try:
        i = parts.index("latents") if "latents" in parts else parts.index("latents_tactile")
    except ValueError:
        raise SystemExit("path does not look like a latent file inside a repo")
    repo = Path(*parts[:i])
    split, task, robot = parts[i - 3], parts[i - 2], parts[i - 1]
    idx = int(latent_path.name.split("_")[1])
    for line in (repo / "meta/episodes.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["episode_index"] == idx:
            sf = e.get("source_frames", "?")
            print(f"latent     : {latent_path.name}")
            print(f"split/task : {split} / {task}")
            print(f"robot      : {robot}")
            print(f"instruction: {(e.get('tasks') or [''])[0]}")
            print(f"source ep  : {e.get('source_episode', '?')}  (session {e.get('source_session', '')})")
            print(f"raw frames : {sf}  (segment length {e['length']})")
            print(f"source path: {e.get('source_path', '?')}")
            return
    raise SystemExit(f"episode_index {idx} not found in {repo}/meta/episodes.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--lookup", type=Path, default=None,
                    help="print provenance for one latent file instead of building the index")
    a = ap.parse_args()
    if a.lookup:
        lookup(a.data, a.lookup)
    else:
        build(a.data)
