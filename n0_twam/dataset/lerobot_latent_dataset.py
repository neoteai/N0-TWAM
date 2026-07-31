# Copyright 2025-2026 NeoteAI Team. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import json
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.video_utils import decode_video_frames
import logging

def recursive_find_file(directory, filename='info.json'):
    result = []
    ignored_dirs = {"data", "videos", "latents", ".cache", "__pycache__"}
    try:
        for root, dirs, files in os.walk(directory, followlinks=True):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    # Tolerate broken/in-flight repos (conversion interrupted mid-episode,
    # parquet missing, meta inconsistent): skip with a warning instead of
    # killing the whole multi-dataset init. Essential for train-while-convert.
    try:
        return LatentLeRobotDataset(
            repo_id=repo_id,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Skipping unusable repo %s: %s", repo_id, exc)
        return None

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    repo_list = sorted(repo_list)
    if not repo_list:
        return []
    num_init_worker = max(1, min(int(num_init_worker), len(repo_list)))
    if num_init_worker <= 1 or len(repo_list) <= 1:
        datasets_out_lst = [construct_func(repo_id) for repo_id in repo_list]
    else:
        with Pool(num_init_worker) as pool:
            datasets_out_lst = pool.map(construct_func, repo_list)
    skipped = sum(1 for d in datasets_out_lst if d is None)
    if skipped:
        logging.warning("construct_lerobot: skipped %d unusable repos", skipped)
    return [d for d in datasets_out_lst if d is not None]

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        num_init_worker = int(getattr(config, "num_init_worker", num_init_worker))
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        try:
            self.meta = LeRobotDatasetMetadata(
                self.repo_id, self.root, self.revision, force_cache_sync=False
            )
        except Exception:
            # LeRobot v3.0 (+ lerobot>=0.3) rejects a full-path repo_id ("must be
            # 'repo_name'") and tries to resolve a pinned revision via the hub.
            # Retry with the basename and no revision -> uses the local snapshot.
            # v2.1 repos succeed on the first call above, so they are untouched.
            self.meta = LeRobotDatasetMetadata(
                Path(self.repo_id).name, self.root, None, force_cache_sync=False
            )

        self.repo_name = Path(self.repo_id).name
        self.config = config
        per_repo_obs_cam_keys = getattr(config, 'per_repo_obs_cam_keys', None) or {}
        self.used_video_keys = list(per_repo_obs_cam_keys.get(self.repo_name, config.obs_cam_keys))
        # Mixed-robot pretraining: tactile sensor sets differ per repo (4/2/0
        # streams). per_repo_tactile_keys overrides the global list; an empty
        # list means this repo has no tactile at all.
        per_repo_tactile_keys = getattr(config, 'per_repo_tactile_keys', None) or {}
        self.used_tactile_keys = list(
            per_repo_tactile_keys.get(self.repo_name, getattr(config, 'tactile_keys', []))
        )
        self.has_tactile_condition = bool(self.used_tactile_keys)
        # When True, episodes whose tactile latents are missing fall back to the
        # CFG tactile-drop path (model's zero-anchor keeps grads sane) instead of
        # raising — required when tactile-less repos mix into pretraining.
        self.tactile_optional = bool(getattr(config, 'tactile_optional', False))
        self.synthetic_tactile_data = bool(getattr(config, 'synthetic_tactile_data', False))
        self.tactile_channels = int(getattr(config, 'tactile_in_channels', 3))
        
        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError) as exc:
            raise FileNotFoundError(f"Incomplete local LeRobot dataset under {self.root}") from exc
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = Path(repo_id) / 'latents'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False).detach()
        self.cfg_prob = config.cfg_prob
        per_repo_used_action_channel_ids = getattr(config, 'per_repo_used_action_channel_ids', None) or {}
        self.used_action_channel_ids = list(
            per_repo_used_action_channel_ids.get(
                self.repo_name,
                getattr(config, 'used_action_channel_ids', []),
            )
        )
        if self.used_action_channel_ids:
            action_dim = int(getattr(config, 'action_dim', 30))
            inverse_ids = [len(self.used_action_channel_ids)] * action_dim
            for i, j in enumerate(self.used_action_channel_ids):
                inverse_ids[j] = i
            self.inverse_used_action_channel_ids = inverse_ids
        else:
            self.inverse_used_action_channel_ids = list(config.inverse_used_action_channel_ids)
        # per-robot (embodiment) norm: per_repo_norm_stat maps repo basename
        # ("arx5") or its robot base ("ur" for "ur_3cam") to {q01,q99}; falls
        # back to the global norm_stat when absent.
        #
        # robot_base must find the robot token ANYWHERE in repo_name, not just the
        # first token: some repos are named "<task>_<robot>" (e.g.
        # "<task>_<robot>_<n>cam" variants) where split("_")[0] is
        # the TASK ("stack"/"scoop") and would miss the per-robot stat -> silent
        # global fallback (wrong scale, esp. grippers).
        # So we match any known-robot key (from per_repo_norm) appearing as a token.
        _norm_stat = config.norm_stat
        _per_repo_norm = getattr(config, 'per_repo_norm_stat', None) or {}
        if _per_repo_norm:
            _tokens = set(self.repo_name.split("_"))
            _known_robots = sorted(k for k in _per_repo_norm if k)  # drop '' key
            _robot_base = next((k for k in _known_robots if k in _tokens),
                               self.repo_name.split("_")[0])
            _norm_stat = _per_repo_norm.get(
                self.repo_name, _per_repo_norm.get(_robot_base, _norm_stat))
        self.q01 = np.array(_norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(_norm_stat['q99'], dtype='float')[None]
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=['action'],
                output_all_columns=False
            )
        self._hf_tactile_view = None
        available_columns = set(getattr(self.hf_dataset, "column_names", []))
        tactile_columns = [key for key in self.used_tactile_keys if key in available_columns]
        if self.has_tactile_condition and tactile_columns:
            self._hf_tactile_view = self.hf_dataset.with_format(
                type='torch',
                columns=tactile_columns,
                output_all_columns=False,
            )
        if self.has_tactile_condition and self._hf_tactile_view is None and self.synthetic_tactile_data:
            logging.warning(
                "Using synthetic tactile videos for training. "
                "Fake tactile streams are enabled because tactile columns were not found in dataset %s.",
                self.repo_id,
            )
        self.filter_mismatched_latents = bool(
            getattr(config, 'filter_mismatched_latents', True)
        )
        self._latent_frame_count_cache = {}
        self._meta_filter_counts = {}
        self.parse_meta()

    def _episode_chunk_candidates(self, episode_index: int) -> list[int]:
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        candidates = [episode_chunk]
        if episode_chunk != 0:
            candidates.append(0)
        return candidates

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = []
        for ep_idx in episodes:
            data_path = self.meta.get_data_file_path(ep_idx)
            full_path = self.root / data_path
            if not full_path.is_file():
                fallback = self.root / "data" / "chunk-000" / Path(data_path).name
                if fallback.is_file():
                    data_path = fallback.relative_to(self.root)
            fpaths.append(str(data_path))
        return fpaths

    def _resolve_latent_file(self, episode_index: int, start_frame: int, end_frame: int, key: str) -> Path:
        filename = f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        for chunk_index in self._episode_chunk_candidates(episode_index):
            latent_file = self.latent_path / f"chunk-{chunk_index:03d}" / key / filename
            if latent_file.exists():
                return latent_file
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        return self.latent_path / f"chunk-{episode_chunk:03d}" / key / filename

    def _resolve_tactile_latent_file(
        self,
        episode_index: int,
        start_frame: int,
        end_frame: int,
        key: str,
        mode: str,
        raise_on_ambiguous: bool = True,
    ) -> Path | None:
        tactile_root_name = getattr(self.config, 'tactile_latent_root_name', 'latents_tactile')
        tactile_root = self.root / tactile_root_name
        if not tactile_root.exists():
            return None
        filename = f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        glob_name = f"episode_{episode_index:06d}_*.pth"
        chunks_size = int(self.meta.info.get('chunks_size', 1000))
        chunk_candidates = [episode_index // chunks_size]
        for chunk_index in self._episode_chunk_candidates(episode_index):
            if chunk_index not in chunk_candidates:
                chunk_candidates.append(chunk_index)

        for chunk_index in chunk_candidates:
            directory = tactile_root / mode / f'chunk-{chunk_index:03d}' / key
            if not directory.exists():
                continue
            exact = directory / filename
            if exact.exists():
                return exact
            candidates = list(directory.glob(glob_name))
            if not candidates:
                continue
            if len(candidates) > 1:
                if raise_on_ambiguous:
                    raise FileNotFoundError(
                        f"Tactile latent segment {filename} not found in "
                        f"{directory}, and the episode glob matches "
                        f"{len(candidates)} files — cannot pick one safely. "
                        "Re-encode tactile latents per segment "
                        "(script/encode_tactile_latent.py) so names match the "
                        "video latents."
                    )
                return None
            return candidates[0]
        return None

    def _latent_frame_count(self, latent_file: Path) -> int:
        latent_file = Path(latent_file)
        cached = self._latent_frame_count_cache.get(latent_file)
        if cached is not None:
            return cached
        # mmap=True reads the tensor lazily, so reading only the small
        # 'latent_num_frames' metadata field doesn't pull the whole latent off
        # shared storage — ~4x faster for the validation pass. Fall back if unsupported.
        try:
            payload = torch.load(latent_file, map_location='cpu',
                                 weights_only=False, mmap=True)
        except Exception:
            payload = torch.load(latent_file, map_location='cpu', weights_only=False)
        count = int(payload['latent_num_frames'])
        self._latent_frame_count_cache[latent_file] = count
        return count

    def _reject_meta(self, reason: str) -> bool:
        self._meta_filter_counts[reason] = self._meta_filter_counts.get(reason, 0) + 1
        return False

    def _valid_seg_cache_path(self):
        return Path(self.root) / ".valid_seg_cache.json"

    def _repo_validation_signature(self):
        """Cheap (no torch.load) fingerprint of the repo's data + filter config.
        The valid-segment cache is reused only when this matches, so any latent
        add/remove (re-encode / gapfill), parquet change, camera/tactile-key
        change, or filter-flag flip forces a fresh validation -> never a stale
        cache. Counting latent files is os-level (fast), unlike _check_meta's
        per-segment torch.load."""
        root = Path(self.root)
        vp = root / "latents" / "chunk-000"
        nvid = 0
        if vp.is_dir():
            for cam in vp.iterdir():
                if cam.is_dir():
                    nvid += sum(1 for _ in cam.glob("*.pth"))
        tname = getattr(self.config, 'tactile_latent_root_name', 'latents_tactile')
        tp = root / tname
        ntac = sum(1 for _ in tp.rglob("*.pth")) if tp.is_dir() else 0
        pp = root / "data" / "chunk-000"
        npq = sum(1 for _ in pp.glob("*.parquet")) if pp.is_dir() else 0
        return {
            "v": 1,
            "parquet": npq,
            "video_latents": nvid,
            "tactile_latents": ntac,
            "video_keys": sorted(self.used_video_keys),
            "tactile_keys": sorted(self.used_tactile_keys),
            "filter": bool(self.filter_mismatched_latents),
        }

    def _load_valid_seg_cache(self):
        """Cached set of valid (episode, start, end) keys iff the on-disk cache
        matches the current data signature; else None (-> full validation)."""
        if not bool(getattr(self.config, 'use_valid_seg_cache', True)):
            return None
        p = self._valid_seg_cache_path()
        if not p.is_file():
            return None
        try:
            blob = json.loads(p.read_text())
            if blob.get("signature") != self._repo_validation_signature():
                return None
            return {tuple(int(v) for v in k) for k in blob.get("valid", [])}
        except Exception:
            return None

    def _save_valid_seg_cache(self, valid_set):
        if not bool(getattr(self.config, 'use_valid_seg_cache', True)):
            return
        try:
            p = self._valid_seg_cache_path()
            tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps({
                "signature": self._repo_validation_signature(),
                "valid": [[int(k[0]), int(k[1]), int(k[2])] for k in valid_set],
            }))
            os.replace(tmp, p)  # atomic; concurrent ranks write identical content
        except Exception as exc:
            logging.warning("valid-seg cache write failed for %s: %s", self.repo_id, exc)

    def parse_meta(self):
        # One-time per data state: validating every segment (existence + frame
        # consistency of each latent via torch.load) is the slow part of init,
        # and all ranks repeat it. Cache the validated valid-segment set keyed by
        # a data signature so subsequent launches/resumes skip the torch.load
        # pass. Build it once up front.
        cached_valid = self._load_valid_seg_cache()
        out = []
        total = 0
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                total += 1
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                if cached_valid is not None:
                    check_statu = (int(episode_index),
                                   int(cur_meta["start_frame"]),
                                   int(cur_meta["end_frame"])) in cached_valid
                else:
                    check_statu = self._check_meta(
                        cur_meta["start_frame"],
                        cur_meta["end_frame"],
                        cur_meta["episode_index"],
                    )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out
        if cached_valid is None:
            self._save_valid_seg_cache({
                (int(m["episode_index"]), int(m["start_frame"]), int(m["end_frame"]))
                for m in out})
            if self._meta_filter_counts:
                logging.warning(
                    "Filtered %d/%d latent segments for repo=%s: %s",
                    total - len(out),
                    total,
                    self.repo_id,
                    self._meta_filter_counts,
                )
        else:
            logging.info(
                "valid-seg cache hit: %d/%d segments for repo=%s",
                len(out), total, self.repo_id)

    def _check_meta(self, start_frame, end_frame, episode_index):
        expected_frames = None
        for key in self.used_video_keys:
            latent_file = self._resolve_latent_file(episode_index, start_frame, end_frame, key)
            if not os.path.exists(latent_file):
                return self._reject_meta('missing_video_latent')
            if self.filter_mismatched_latents:
                try:
                    frame_count = self._latent_frame_count(latent_file)
                except Exception as exc:
                    logging.warning("Failed to read video latent metadata %s: %s", latent_file, exc)
                    return self._reject_meta('bad_video_latent')
                if expected_frames is None:
                    expected_frames = frame_count
                elif frame_count != expected_frames:
                    return self._reject_meta('video_frame_mismatch')

        if (
            self.filter_mismatched_latents
            and self.has_tactile_condition
            and self.used_tactile_keys
            and not self.synthetic_tactile_data
        ):
            for key in self.used_tactile_keys:
                global_file = self._resolve_tactile_latent_file(
                    episode_index, start_frame, end_frame, key, 'global',
                    raise_on_ambiguous=False)
                local_file = self._resolve_tactile_latent_file(
                    episode_index, start_frame, end_frame, key, 'local',
                    raise_on_ambiguous=False)
                if global_file is None or local_file is None:
                    return self._reject_meta('missing_tactile_latent')
                try:
                    global_frames = self._latent_frame_count(global_file)
                    local_frames = self._latent_frame_count(local_file)
                except Exception as exc:
                    logging.warning(
                        "Failed to read tactile latent metadata repo=%s episode=%s key=%s: %s",
                        self.repo_id, episode_index, key, exc,
                    )
                    return self._reject_meta('bad_tactile_latent')
                if global_frames != local_frames:
                    return self._reject_meta('tactile_local_global_mismatch')
                if expected_frames is not None and global_frames != expected_frames:
                    return self._reject_meta('tactile_video_frame_mismatch')
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        out = {}
        for key in self.used_video_keys:
            latent_file = self._resolve_latent_file(episode_index, start_frame, end_frame, key)
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"].detach()
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='edge')

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def _load_tactile_latents(
        self,
        episode_index: int,
        local_start_frame: int,
        local_end_frame: int,
        latent_frame_ids,
        truncate_start: int | None = None,
        truncate_end: int | None = None,
        expected_full_frames: int | None = None,
    ) -> dict | None:
        """Load pre-computed GlobalTactile + LocalTactile latents for an episode.

        Expects files produced by script/encode_tactile_latent.py at:
            <dataset_root>/latents_tactile/{global,local}/chunk-XXX/<tactile_key>/episode_*_*.pth

        Returns dict {
            'global':      torch.Tensor (S, C=48, F_lat_truncated, H_lat, W_lat),
            'local':       torch.Tensor (S, C=48, F_lat_truncated, H_lat, W_lat),
            'sensor_ids':  torch.LongTensor (S,)  — sensor index per slot,
        } or None if any sensor's latent is missing.

        truncate_start/truncate_end: indices into F_lat to slice (matches the
        truncation applied to video latents to keep temporal alignment).
        expected_full_frames: the VIDEO latent's full (pre-truncation) frame
        count. Tactile is frame-aligned with video in the attention mask, so a
        mismatched tactile F would silently misalign every tactile frame —
        assert instead of training on wrong semantics.
        """
        sensor_id_map = getattr(self.config, 'tactile_sensor_id_map', None)
        tactile_root_name = getattr(self.config, 'tactile_latent_root_name', 'latents_tactile')
        tactile_root = self.root / tactile_root_name
        if not tactile_root.exists():
            return None

        globals_list = []
        locals_list = []
        sensor_ids_list = []
        for key in self.used_tactile_keys:
            global_file = self._resolve_tactile_latent_file(
                episode_index, local_start_frame, local_end_frame, key, 'global')
            local_file = self._resolve_tactile_latent_file(
                episode_index, local_start_frame, local_end_frame, key, 'local')
            if global_file is None or local_file is None:
                # one sensor missing — skip whole tactile for this batch
                return None

            g_payload = torch.load(global_file, map_location='cpu', weights_only=False)
            l_payload = torch.load(local_file, map_location='cpu', weights_only=False)
            g_flat = g_payload['latent']                            # (F*H*W, C)
            l_flat = l_payload['latent']
            F_lat = int(g_payload['latent_num_frames'])
            H_lat = int(g_payload['latent_height'])
            W_lat = int(g_payload['latent_width'])
            F_lat_local = int(l_payload.get('latent_num_frames', F_lat))
            if F_lat_local != F_lat:
                raise ValueError(
                    f"Tactile local/global frame mismatch for {key} "
                    f"episode {episode_index}: local={F_lat_local}, "
                    f"global={F_lat}."
                )
            # Tactile is frame-aligned with video tokens (same frame_id in the
            # attention mask), so the FULL tactile F must equal the video's
            # full latent F — otherwise the shared truncate window slices a
            # shifted/short range and every tactile frame silently pairs with
            # the wrong video frame.
            if expected_full_frames is not None and F_lat != expected_full_frames:
                raise ValueError(
                    f"Tactile/video latent frame mismatch for {key} episode "
                    f"{episode_index}: tactile F={F_lat}, video F="
                    f"{expected_full_frames}. Re-encode tactile latents with "
                    "the same --target-fps/frame policy as the video latents."
                )
            # Reshape (F*H*W, C) → (F, H, W, C) → (C, F, H, W)
            g_5d = g_flat.reshape(F_lat, H_lat, W_lat, -1).permute(3, 0, 1, 2).contiguous()
            l_5d = l_flat.reshape(F_lat, H_lat, W_lat, -1).permute(3, 0, 1, 2).contiguous()

            # Slice along F to match the truncated video latent window
            if truncate_start is not None and truncate_end is not None:
                g_5d = g_5d[:, truncate_start:truncate_end]
                l_5d = l_5d[:, truncate_start:truncate_end]

            globals_list.append(g_5d)
            locals_list.append(l_5d)

            # Resolve sensor_id from cfg map, default to enumeration order
            if sensor_id_map and key in sensor_id_map:
                sensor_ids_list.append(int(sensor_id_map[key]))
            else:
                sensor_ids_list.append(len(sensor_ids_list))

        return {
            'global': torch.stack(globals_list, dim=0).contiguous(),       # (S, C, F, H, W)
            'local': torch.stack(locals_list, dim=0).contiguous(),
            'sensor_ids': torch.tensor(sensor_ids_list, dtype=torch.long),  # (S,)
        }

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        num_latent_frames = ori_data_dict[f"{self.used_video_keys[0]}.latent_num_frames"]

        # Truncate long episodes to avoid CUDA OOM
        start_lat = None        # init so tactile loader can reference outside the if
        end_lat = None
        max_latent_frames = int(getattr(self.config, 'max_latent_frames', 0))
        if max_latent_frames > 0 and num_latent_frames > max_latent_frames:
            start_lat = torch.randint(0, num_latent_frames - max_latent_frames + 1, (1,)).item()
            end_lat = start_lat + max_latent_frames
            # Truncate latent tensors for each video key
            for key in self.used_video_keys:
                F = ori_data_dict[f"{key}.latent_num_frames"]
                H = ori_data_dict[f"{key}.latent_height"]
                W = ori_data_dict[f"{key}.latent_width"]
                latent = ori_data_dict[f"{key}.latent"].reshape(F, H * W, -1)
                ori_data_dict[f"{key}.latent"] = latent[start_lat:end_lat].reshape(-1, latent.shape[-1])
                ori_data_dict[f"{key}.latent_num_frames"] = max_latent_frames
            # Truncate frame_ids: action code uses (len(frame_ids)-1)//4+1 as latent_frame_num
            # so we need exactly (max_latent_frames-1)*4+1 video frame entries
            needed_vid_frames = (max_latent_frames - 1) * 4 + 1
            vid_start = start_lat * 4
            vid_start = min(vid_start, max(0, len(latent_frame_ids) - needed_vid_frames))
            vid_end = vid_start + needed_vid_frames
            latent_frame_ids = latent_frame_ids[vid_start:vid_end]

        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)
        # ─── NEW: load pre-computed GlobalTactile / LocalTactile latents ───
        # (replaces the old RGB-video → CNN tactile pipeline)
        if self.has_tactile_condition and self.used_tactile_keys:
            tactile_payload = self._load_tactile_latents(
                episode_index=episode_index,
                local_start_frame=local_start_frame,
                local_end_frame=local_end_frame,
                latent_frame_ids=latent_frame_ids,
                truncate_start=start_lat,
                truncate_end=end_lat,
                # video's FULL latent frame count (num_latent_frames is read
                # before truncation) — tactile must match it frame-for-frame.
                expected_full_frames=int(num_latent_frames),
            )
            if tactile_payload is None and not self.tactile_optional:
                raise FileNotFoundError(
                    "Missing tactile latents for cond branch: "
                    f"repo={self.repo_id} episode={episode_index} "
                    f"frames={local_start_frame}:{local_end_frame} "
                    f"keys={self.used_tactile_keys}"
                )

            if tactile_payload is None:
                # tactile_optional: missing tactile falls back to the CFG-drop
                # path (model's zero-anchor keeps gradients FSDP-safe).
                out_dict['tactile_cond_drop'] = torch.tensor(True, dtype=torch.bool)
            else:
                # Keep the real tensors in the sample so missing files are still
                # caught above. The model uses this explicit flag to skip tactile
                # tokens entirely for CFG drop, so tactile modules get no gradient.
                tactile_cfg_prob = float(getattr(self.config, 'tactile_cfg_prob', 0.1))
                tactile_cond_drop = torch.rand(1).item() < tactile_cfg_prob

                out_dict['tactile_global_latent'] = tactile_payload['global']
                out_dict['tactile_local_latent'] = tactile_payload['local']
                out_dict['tactile_sensor_ids'] = tactile_payload['sensor_ids']
                out_dict['tactile_cond_drop'] = torch.tensor(
                    tactile_cond_drop, dtype=torch.bool)
        else:
            # Repo with no tactile sensors at all (per_repo_tactile_keys = []):
            # explicit drop flag so the model takes the zero-anchor path.
            out_dict['tactile_cond_drop'] = torch.tensor(True, dtype=torch.bool)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from n0_twam.configs import TWAM_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        TWAM_CONFIGS['base']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
