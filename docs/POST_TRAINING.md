# Post-training N0-TWAM on your own data

This guide takes you from robot demonstrations to a checkpoint fine-tuned from the
released N0-TWAM weights, and to closed-loop evaluation. The flow is:

```
your demos → LeRobot v2.1 → encode latents → build pool + norm stats
           → edit the posttrain config → train → serve → closed-loop eval
```

## 1. Put your data in LeRobot v2.1 format

The dataset loader consumes [LeRobot](https://github.com/huggingface/lerobot) v2.1
repositories. Each episode needs:

- **RGB cameras** under `observation.images.<cam_name>` (e.g. `top`, `wrist_l`).
- **Tactile videos** under `observation.images.tactile_*` (e.g. `tactile_a`,
  `tactile_b` for a single-arm gripper; four streams for dual-arm).
- **Action / state** columns — end-effector poses in the 20-dim dual-arm schema
  `[left xyz, rot6d, grip | right xyz, rot6d, grip]`; single-arm robots fill the
  first 10 dims and mask the rest.
  **The `action` column must hold ABSOLUTE EE target poses** (per-step-delta
  actions silently produce wrong `absee` stats and a broken policy — sanity-check
  the `*_raw_report.json` in step 3: xyz quantiles should look like workspace
  coordinates, rot6d components like ±1, not ±0.01).
- **Complete `meta/`**: `info.json`, `episodes.jsonl`, `tasks.jsonl` **and
  `episodes_stats.jsonl`** — the LeRobot loader hard-requires the last one; when
  it is missing the repo is treated as remote and fails with a misleading
  HuggingFace-Hub 401/authentication error. Repos written by
  `LeRobotDataset.create()` have all four; hand-converted repos must too.

## 2. Encode latents

Encode the RGB and tactile videos into Wan2.2 VAE latents once, up front. Point
`--model-path` at the base model directory (it must contain `vae/`, `tokenizer/`
and `text_encoder/` — the video encoder also precomputes the text embeddings).

```bash
# video latents (RGB cameras; tactile streams are handled by the next script)
python script/encode_lerobot_n0_latents.py --dataset-root /path/to/your_repo \
    --model-path /path/to/base-model

# tactile latents (global = diffusion target, local = observed input).
# --local-mode current is part of the recipe: the local stream feeds the action
# expert the CURRENT observed tactile frame.
python script/encode_tactile_latent.py --dataset-root /path/to/your_repo \
    --model-path /path/to/base-model --mode both --local-mode current \
    --tactile-keys observation.images.tactile_a observation.images.tactile_b
```

Run each with `--help` for the full option list. Encoded latents are written next to
the dataset under `latents/` and `latents_tactile/`.

## 3. Build the task pool + normalization stats

```bash
python script/build_task_pool.py --data /path/to/your_repo --pool /path/to/pools/my_task
```

This creates `<pool>/train|val/<repo>` symlinks and computes the action norm stats
in the same anchor/horizon-12 windows training uses. Two action representations are
supported, each with its **own** stats file (they must never overwrite each other):

| mode (`--mode`) | action space | stats file |
|---|---|---|
| `absee` (default, the validated recipe) | absolute EE pose | `norm_stat_absee.json` |
| `delta` | π0.5-style horizon delta | `norm_stat.json` |

Add `--dual-arm` for 20-dim dual-arm repos. Keep the stats **fixed** for the whole
run. The `*_raw_report.json` audit file records the raw quantiles and every rule
applied — check it once before training (dead channels, gripper bounds).

> `script/build_segment_index.py` can additionally build a human-readable index
> over multi-repo buckets; single-task post-training does not need it.

## 4. Configure

Edit the `EDIT ME` block of
[`n0_twam/configs/twam_posttrain_cfg.py`](../n0_twam/configs/twam_posttrain_cfg.py)
(registered as `posttrain`): pool path, base model, released checkpoint, prompt
(**verbatim** from your `meta/tasks.jsonl`), camera/tactile keys.

Two switches live in the same block:

- `_ACTION_MODE` — `"absee"` (default) or `"pi05_delta"`; must match the
  `--mode` you used in step 3.
- **Ablation switches**, default = full model: `_USE_LOCAL_TACTILE = False`
  knocks out the local (observed) stream; `_TACTILE_GLOBAL_ZERO = True` knocks
  out the global (imagined) stream. The serve config inherits both, so an
  ablation is trained *and served* consistently.

Everything else in the file is the validated recipe (MoT experts, LocalTactile
"current", horizon 12, drop probabilities 0) and is asserted at the bottom —
prefer not to touch it.

## 5. Launch post-training

```bash
NGPU=8 bash run_posttrain.sh
```

Training resumes from the released weights (the LocalTactile branch is built
zero-init on top of the local-off pretrain checkpoint) and adapts them to your
data. Checkpoints land under `<save_root>/checkpoints/checkpoint_step_*/`, each
with a `train_meta.json` snapshot of the serve-critical settings — the inference
server cross-checks it at startup and refuses to serve a mismatched config.

## 6. Deploy and evaluate closed-loop

Assemble a serve bundle and start the paired server config
(`posttrain_server` inherits your training config):

```bash
python script/make_serve_bundle.py --checkpoint /path/to/checkpoint_step_2000 \
    --base /path/to/base-model --bundle /path/to/serve-bundle
python -m n0_twam.n0_twam_server --config-name posttrain_server --port 29601
```

Serving details, the raw client protocol, and closed-loop evaluation with the
[NeoSim](https://github.com/neoteai/NeoSim) benchmark clients are covered in
[DEPLOY.md](DEPLOY.md).
