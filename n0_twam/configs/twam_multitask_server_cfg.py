# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Serve ONE task of a multi-task checkpoint (registered "multitask_server").

A multi-task pool trains every task with its OWN normalization stats (the
``<norm>_per_robot.json`` table build_task_pool.py writes, one entry per repo).
The checkpoint therefore has no single serve norm: de-normalizing with the
pool-level file (an envelope kept only as an unreachable training fallback)
scales actions wrongly, and the train_meta.json snapshot stores that same
envelope — so a mismatch would even pass the naive cross-check. This config
selects a task and wires up everything that task trained with:

  * q01/q99 from the pool's ``<norm>_per_robot.json`` entry,
  * camera / tactile keys from the task repo's ``meta/info.json``
    (declaration order — the order the task trained with),
  * action channels from the repo's action dim (10 = single-arm half of the
    20-d schema, 20 = dual-arm),
  * the warm-up prompt from the repo's ``meta/tasks.jsonl`` (training text,
    verbatim; a client reset with a prompt overrides it).

Everything else (recipe, ablation switches, inference knobs) inherits from
``posttrain_server`` unchanged. The server recognises ``serve_task`` and
cross-checks the live norm against the per-task table instead of the envelope.

One server instance serves ONE task; run several instances for several tasks:

    TWAM_SERVE_POOL=/path/to/pools/my_tasks TWAM_SERVE_TASK=my_task \\
      python -m n0_twam.n0_twam_server --config-name multitask_server --port 29601
"""
import json
import os
from pathlib import Path

from easydict import EasyDict

from .twam_posttrain_server_cfg import twam_posttrain_server_cfg

# ───────── EDIT ME (each value can also be set via the env var) ─────────
_POOL = Path(os.environ.get("TWAM_SERVE_POOL", "/path/to/pools/my_tasks"))
_TASK = os.environ.get("TWAM_SERVE_TASK", "my_task")   # repo name under <pool>/train/
_ACTION_MODE = os.environ.get("TWAM_SERVE_ACTION_MODE", "absee")  # "absee" | "delta"
_BUNDLE = os.environ.get("TWAM_SERVE_BUNDLE", "/path/to/serve-bundle")
_SAVE_ROOT = os.environ.get("TWAM_SERVE_OUT", "/path/to/serve-output")
# ───────── end EDIT ME ─────────

assert _ACTION_MODE in ("absee", "delta"), _ACTION_MODE
_NORM_NAME = "norm_stat_absee" if _ACTION_MODE == "absee" else "norm_stat"

s = EasyDict()
s.update(twam_posttrain_server_cfg)
s.__name__ = f"Config: N0-TWAM multi-task SERVER ({_ACTION_MODE}, task={_TASK})"

s.wan22_pretrained_model_name_or_path = _BUNDLE
s.save_root = _SAVE_ROOT
s.action_delta_mode = "none" if _ACTION_MODE == "absee" else "pi05_delta"

# multi-task markers — the server's consistency check keys off serve_task.
s.serve_task = _TASK
s.multitask_norm_path = str(_POOL / f"{_NORM_NAME}_per_robot.json")

_repo_meta = _POOL / "train" / _TASK / "meta"
_per_path = Path(s.multitask_norm_path)

if _POOL.is_dir():
    # ── per-task norm: the exact stats this task trained with ──
    assert _per_path.is_file(), (
        f"multi-task pool {_POOL} has no {_per_path.name} — per-task stats are "
        "required; the pool-level file is a training-only fallback (see "
        "script/build_task_pool.py)")
    _per = json.loads(_per_path.read_text())
    assert _TASK in _per, (
        f"task {_TASK!r} not in {_per_path.name} — available: {sorted(_per)}")
    s.norm_stat = {"q01": list(_per[_TASK]["q01"]), "q99": list(_per[_TASK]["q99"])}
    s.norm_stat_path = f"{_per_path}[{_TASK}]"

    # ── observation contract + prompt, from the task repo itself ──
    _info = json.loads((_repo_meta / "info.json").read_text())
    _video_keys = [k for k, v in _info["features"].items()
                   if k.startswith("observation.images.") and v.get("dtype") == "video"]
    s.obs_cam_keys = [k for k in _video_keys if "tactile" not in k]
    s.tactile_keys = [k for k in _video_keys if "tactile" in k]
    s.tactile_sensor_id_map = {k: i for i, k in enumerate(s.tactile_keys)}

    _action_dim = int(_info["features"]["action"]["shape"][0])
    s.used_action_channel_ids = list(range(min(_action_dim, s.action_dim)))
    _inverse = [len(s.used_action_channel_ids)] * s.action_dim
    for _i, _j in enumerate(s.used_action_channel_ids):
        _inverse[_j] = _i
    s.inverse_used_action_channel_ids = _inverse

    _task_row = json.loads(
        (_repo_meta / "tasks.jsonl").read_text().splitlines()[0])
    s.prompt = _task_row["task"]
    s.eval_prompt = s.prompt
else:  # keep import safe pre-data-prep; runtime guards refuse the placeholder
    s.norm_stat = {"q01": [-1.0] * 20, "q99": [1.0] * 20}
    s.norm_stat_path = str(_per_path) + " (POOL MISSING)"

# ── self-check ──
assert s.action_delta_mode in ("none", "pi05_delta")
assert s.server_action_output_format == "absolute"
assert len(s.norm_stat["q01"]) == 20 and len(s.norm_stat["q99"]) == 20
if _POOL.is_dir():
    assert s.obs_cam_keys and s.tactile_keys, \
        f"no camera/tactile video keys found in {_repo_meta}/info.json"
    # the selected stats must be the task's own, never the pool envelope
    _pool_file = _POOL / f"{_NORM_NAME}.json"
    if _pool_file.is_file():
        _env = json.loads(_pool_file.read_text())
        assert s.norm_stat["q01"] != _env["q01"] or s.norm_stat["q99"] != _env["q99"] \
            or len(_per) == 1, \
            "per-task norm equals the pool envelope — wrong stats wired"

twam_multitask_server_cfg = s
