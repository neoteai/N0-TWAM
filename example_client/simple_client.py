#!/usr/bin/env python3
"""Minimal N0-TWAM client: reset + a few observation steps -> action chunks.

Sends the bundled example frames (synthetic zero tactile / state) to a running
server and prints the returned action chunks — a smoke test for the serving
stack and a template for wiring a real robot or simulator. For full closed-loop
evaluation use the NeoSim clients (see docs/DEPLOY.md).

  # post-trained checkpoint server (config: posttrain_server)
  python example_client/simple_client.py --port 29601

  # released pretrain checkpoint server (config: twam_server)
  python example_client/simple_client.py --port 29601 --pretrain

NOTE: unset http_proxy / https_proxy in the client shell — Python websockets
does not honor NO_PROXY CIDRs and a proxy breaks the handshake.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(REPO))

from n0_twam.utils.Simple_Remote_Infer.deploy.websocket_client_policy import (  # noqa: E402
    WebsocketClientPolicy,
)


def load_frames(img_dir: Path, cams, name_map):
    frames = {}
    for k in cams:
        p = img_dir / name_map[k]
        if p.is_file():
            frames[k] = np.array(Image.open(p).convert("RGB").resize((256, 256)))
        else:
            print(f"warn: {p} not found, sending zeros")
            frames[k] = np.zeros((256, 256, 3), np.uint8)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29601)
    ap.add_argument("--pretrain", action="store_true",
                    help="talk to the released-pretrain server (twam_server config) "
                         "instead of a post-trained one (posttrain_server config)")
    ap.add_argument("--prompt", default="pick up the object")
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()

    if args.pretrain:
        # released checkpoint: 3 pretrain camera slots; tactile is synthesized
        # server-side when twam_server_cfg.synthetic_tactile_data = True
        cams = ["observation.images.third_view",
                "observation.images.left_wrist_view",
                "observation.images.right_wrist_view"]
        img_dir = REPO / "example_client" / "cam_test"
        name_map = {k: f"{k}.png" for k in cams}
        tacs = []
    else:
        # posttrain defaults (single-arm NeoSim tasks): top + wrist_l, 2 tactile
        cams = ["observation.images.top", "observation.images.wrist_l"]
        img_dir = REPO / "example_client" / "twam"
        name_map = {k: f"{k}.png" for k in cams}
        tacs = ["observation.images.tactile_a", "observation.images.tactile_b"]

    frame = load_frames(img_dir, cams, name_map)
    tactile = {k: np.zeros((256, 256, 3), np.uint8) for k in tacs}

    client = WebsocketClientPolicy(args.host, args.port)
    print("reset ->", client.infer({"reset": True, "prompt": args.prompt}))

    for step in range(args.steps):
        obs = {"obs": frame, "current_state": np.zeros(20, np.float32)}
        if tacs:
            obs["tactile"] = tactile
        t0 = time.time()
        action = np.asarray(client.infer(obs)["action"])
        print(f"step {step}: action {action.shape} "
              f"range [{action.min():+.4f}, {action.max():+.4f}] "
              f"finite={np.isfinite(action).all()} "
              f"latency {time.time() - t0:.1f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
