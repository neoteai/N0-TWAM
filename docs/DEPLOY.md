# Deploying N0-TWAM as an inference server

The server loads a trained checkpoint and answers observation → action requests over a
websocket. A client sends `reset` (with a language prompt) then per-step observations,
and receives an action chunk back.

## 1. Assemble a serve bundle

The server loads the VAE, tokenizer, text-encoder, and transformer from a single
directory (`wan22_pretrained_model_name_or_path`). Your training checkpoint only
contains `transformer/`, so combine it with the base model's other components:

```bash
python script/make_serve_bundle.py --checkpoint /path/to/checkpoint_step_2000 \
    --base /path/to/base-model --bundle /path/to/serve-bundle
```

(Equivalent to symlinking `transformer/` from the checkpoint plus `vae/`,
`tokenizer/`, `text_encoder/` from the base model into one directory.)

## 2. Point the server config at your bundle

For a **post-trained** checkpoint use `posttrain_server`
(`n0_twam/configs/twam_posttrain_server_cfg.py`): it inherits your training
config — action space, norm stats, camera/tactile keys, ablation switches — so
serving cannot drift from training. Set `wan22_pretrained_model_name_or_path`
to the bundle and `save_root` to a writable dump dir.

At startup the server cross-checks the config against the checkpoint's
`train_meta.json` snapshot (norm stats, action mode, tactile switches) and
**refuses to start** on a mismatch or on placeholder norm stats — fix the
config rather than bypassing the check.

### Serving a multi-task checkpoint

A multi-task pool trains every task with its **own** normalization stats (the
`<norm>_per_robot.json` table from `script/build_task_pool.py`) — there is no
single serve norm, and de-normalizing with the pool-level file (a training-only
fallback envelope) scales actions wrongly. For these checkpoints — including
the released `NeoteAI/n0-twam-univtac-{absee,delta}` and
`NeoteAI/n0-twam-neosim-{absee,delta}` — use `multitask_server`
(`n0_twam/configs/twam_multitask_server_cfg.py`): it selects one task and
wires the stats, camera/tactile keys, action channels and prompt that task
trained with, all read from the pool itself.

```bash
TWAM_SERVE_POOL=/path/to/pools/my_tasks TWAM_SERVE_TASK=my_task \
TWAM_SERVE_ACTION_MODE=absee TWAM_SERVE_BUNDLE=/path/to/serve-bundle \
TWAM_SERVE_OUT=/path/to/serve-output \
  python -m n0_twam.n0_twam_server --config-name multitask_server --port 29601
```

One server instance serves one task — run several instances (different
`TWAM_SERVE_TASK` / `--port`) to evaluate several tasks in parallel. The
startup cross-check adapts accordingly: the live norm is verified against the
pool's per-task table, and the served task's action channels must be a subset
of the training union recorded in `train_meta.json`.

For the released pretrain checkpoint use `twam_server`
(`n0_twam/configs/twam_server_cfg.py`):

```python
twam_server_cfg.wan22_pretrained_model_name_or_path = "/path/to/serve-bundle"
twam_server_cfg.save_root = "/path/to/serve-output"     # must be writable
twam_server_cfg.port = 29601
twam_server_cfg.num_inference_steps = 10                # fewer = faster, more = higher fidelity
twam_server_cfg.action_num_inference_steps = 10
```

If your data has no tactile sensors, set `twam_server_cfg.synthetic_tactile_data = True`
to feed zero-valued tactile automatically.

**GPU memory.** Serving keeps the transformer (~14 GB), VAE (~3 GB) and umT5
text encoder (~11 GB) resident — plan for a ≥40 GB GPU. `enable_offload = True`
loads the VAE and text encoder on **CPU and leaves them there**: it does save
VRAM, but every per-frame VAE encode then runs on CPU (minutes per chunk), so
treat it as a debug mode, not a way to serve from a small GPU.

## 3. Launch the server

```bash
export PYTHONPATH=$PWD:$PWD/n0_twam
export CUDA_VISIBLE_DEVICES=0
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29988

python -m n0_twam.n0_twam_server --config-name posttrain_server --port 29601
```

Wait for `server listening on 0.0.0.0:29601` in the log. For multi-GPU inference,
launch with `torchrun --nproc_per_node=N` and `WORLD_SIZE=N`.

## 4. Query it from a client

A runnable version of this section ships at
[`example_client/simple_client.py`](../example_client/simple_client.py) — it sends the bundled
example frames with synthetic tactile/state and prints the returned action
chunks (`--pretrain` targets a `twam_server` instance, default targets
`posttrain_server`).

```python
import numpy as np
from n0_twam.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy("127.0.0.1", 29601)

# Start an episode with a language instruction (omit "prompt" to use the
# config's prompt).
client.infer({"reset": True, "prompt": "pick up the object"})

# Send an observation: per-camera frames, tactile frames and the current robot
# state. The keys MUST match the serve config's obs_cam_keys / tactile_keys —
# the values below match twam_posttrain_cfg's single-arm defaults.
cams = ["observation.images.top", "observation.images.wrist_l"]
tacs = ["observation.images.tactile_a", "observation.images.tactile_b"]
frame = {k: np.zeros((256, 256, 3), np.uint8) for k in cams}    # your RGB frames here
tactile = {k: np.zeros((256, 256, 3), np.uint8) for k in tacs}  # your tactile frames here
state = np.zeros(20, np.float32)                                # your current EE state

out = client.infer({"obs": frame, "tactile": tactile, "current_state": state})
action = out["action"]        # shape (action_dim, frame_chunk, actions_per_frame)
                              # = (20, 2, 12) for post-trained, (20, 2, 16) for the pretrain ckpt
```

The returned action is already de-normalized by the server (`postprocess_action`)
using the training norm stats: with `server_action_output_format = "absolute"`
(the `posttrain_server` default) each slot is a directly executable absolute EE
target. Orientation is encoded as rot6d (the first two rotation-matrix columns),
so if your robot API takes quaternions, apply the standard rot6d→quaternion
mapping on the client — `rot6d10_to_pose` in
[`example_client/closed_loop_client.py`](../example_client/closed_loop_client.py) does this.
IK and execution are client-side by design.

## 5. Close the loop

The snippet above is **open-loop**, and that is not how the model is meant to
run. The server is streaming and stateful: the chunk it returns is based on a
future it *imagined*. After executing a chunk you must send back what was
actually observed, so its KV cache is re-grounded on reality. Skip that and the
policy silently degrades — nothing raises, the numbers just get worse.

[`example_client/closed_loop_client.py`](../example_client/closed_loop_client.py) implements
that loop without a simulator, for driving your own robot. Like
`simple_client.py` it needs only numpy + websockets + msgpack — no torch:

```python
from example_client.closed_loop_client import TwamClient

client = TwamClient("127.0.0.1", 29601, prompt="pick up the object", num_arms=1)
client.reset(seed=100)

while not robot.done:
    cams, tactile = robot.observe()       # {"top": HxWx3}, {"tactile_a": HxWx3}
    state = client.encode_state([robot.ee_pose()])
    result = client.run_chunk(cams, tactile, state,
                              execute=robot.move_to,   # execute(poses, ctx) -> abort?
                              observe=robot.observe)
    if result.stopped:
        break
```

`run_chunk` performs one full bracket — **infer → execute the chunk → re-ground
the KV cache**. Your `execute(poses, ctx)` callback gets one `EEPose` (position +
`wxyz` quaternion + gripper) per arm and returns truthy to abort; `observe()` is
called at each keyframe. Short camera/tactile names are mapped to the
`observation.images.*` keys and validated, so a misspelled key raises client-side
instead of reaching the server as a silently wrong observation. A failed
re-grounding raises: treat that episode as invalid rather than continuing.

Run it standalone against a live server for a closed-loop smoke test:

```bash
python example_client/closed_loop_client.py --port 29601 --chunks 2
```

Knobs worth knowing: `num_arms` (1 or 2 — selects default keys and how each
action vector splits), `cam_names` / `tactile_names` (non-default serve config),
`video_keyframes_per_frame` (observations fed back per frame, default 4),
`exec_slots` (execute only the first N slots of a frame). To adapt it to your
hardware, subclass and override `pack_images` (image pipeline),
`is_keyframe_slot` (sampling cadence) or `ActionChunk.poses_at` (non-rot6d action
space) — the file's docstring lists the full set.

For the NeoSim benchmark, use its `eval/` clients instead — see below.

## Serving for NeoSim evaluation

[NeoSim](https://github.com/neoteai/NeoSim)'s policy-server clients
(`eval/eval_twam_ee_cl.py` single-arm, `eval/eval_twam_ee_dual_cl.py` dual-arm,
`eval/run_eval.sh` driver) implement this server's full streaming protocol —
reset, chunked inference, and KV-cache re-grounding on executed observations.
NeoSim's `eval/README.md` documents the clients and their knobs; on this side
four things must line up:

- **Observation keys.** The `posttrain` defaults match NeoSim single-arm tasks
  (`observation.images.top`, `wrist_l`, `tactile_a`, `tactile_b`). Dual-arm
  tasks add `wrist_r`, use the 4-key tactile set, and the dual client.
- **Tactile representation.** The client-side `UNIVTAC_TAC_KEY` selects which
  tactile image is sent (`gel_particle` default, `rgb`, ...). It must equal the
  representation the checkpoint was trained on — a mismatch is silent
  out-of-distribution input, not an error.
- **Prompt and seeds.** Pass the training instruction verbatim
  (`meta/tasks.jsonl`); the held-out protocol is 20 episodes, seeds 100–119.
  Long-horizon tasks may need a larger per-episode budget (`EVAL_STEP_LIM`).
- **Cross-machine serving.** The websocket link works across machines and SSH
  tunnels; unset `http_proxy`/`https_proxy` on the client host (Python
  websockets does not honor `NO_PROXY` CIDRs).
