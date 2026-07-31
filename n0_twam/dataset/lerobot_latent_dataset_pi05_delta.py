# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""LeRobot latent dataset variant with OpenPI/pi0.5-style horizon deltas.

The source parquet stores one-step deltas for xyz + rot6d and absolute gripper
targets. This dataset keeps the parquet unchanged and builds
training targets as:

    action_delta[k, delta_dims] = absolute_target[t + k] - state[t]

for every latent-frame anchor t. The first action frame is reserved as the
streaming condition slot and masked out, matching n0_twam_server.py's
action_cond=0 behavior at frame_st_id == 0.
"""

import logging
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch

from .lerobot_latent_dataset import LatentLeRobotDataset, recursive_find_file


# ───────── 6D rotation (first two columns of R, Zhou et al.) <-> matrix ─────────
def sixd_to_R(m6):
    """(...,6)=[col0(3),col1(3)] -> (...,3,3) via Gram-Schmidt (columns)."""
    a1 = m6[..., 0:3]; a2 = m6[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-9)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2p / (np.linalg.norm(a2p, axis=-1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def R_to_sixd(R):
    """(...,3,3) -> (...,6) first two COLUMNS."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def ee_arm_groups(action_dim, dims_per_arm=10, rot_offset=3):
    """For a dual/single-arm action laid out [pos3, rot6d, grip1] per 10-dim arm,
    return list of (pos_ids[3], rot6d_ids[6]) per arm present within action_dim."""
    groups = []
    for a in range(0, action_dim, dims_per_arm):
        rot = list(range(a + rot_offset, a + rot_offset + 6))
        if rot[-1] < action_dim:
            groups.append(([a, a + 1, a + 2], rot))
    return groups


def construct_lerobot_pi05_delta(repo_id, config):
    # Skip broken/in-flight repos (interrupted conversion, missing parquet)
    # instead of failing the whole multi-dataset init — train-while-convert.
    #
    # BUT under highly concurrent multi-worker init, shared/network storage can throw TRANSIENT read failures
    # ("Expected to be able to read N bytes for message body", "Tried reading schema
    # message, was null", "Cannot find data file", "Incomplete local LeRobot dataset")
    # that are NOT corruption — the same parquet reads fine single-process. A single try/except would permanently
    # drop an otherwise-valid repo. Retry with backoff so a transient
    # read doesn't lose an otherwise-valid repo; only give up after repeated failure.
    import time as _t
    last_exc = None
    attempts = max(1, int(getattr(config, "repo_init_retries", 4)))
    for attempt in range(attempts):
        try:
            return Pi05DeltaLatentLeRobotDataset(
                repo_id=repo_id,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < attempts - 1:
                _t.sleep(2.0 * (attempt + 1))  # 2s,4s,6s — let shared storage contention drain
    logging.warning("Skipping unusable repo %s after %d attempts: %s",
                    repo_id, attempts, last_exc)
    return None


def construct_lerobot_pi05_delta_multi_processor(config, num_init_worker=8):
    construct_func = partial(
        construct_lerobot_pi05_delta,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, "info.json")
    repo_list = [v.split("/meta/info.json")[0] for v in repo_list]
    repo_list = sorted(repo_list)
    if not repo_list:
        return []
    num_init_worker = max(1, min(int(num_init_worker), len(repo_list)))
    if num_init_worker <= 1 or len(repo_list) <= 1:
        out = [construct_func(repo_id) for repo_id in repo_list]
    else:
        with Pool(num_init_worker) as pool:
            out = pool.map(construct_func, repo_list)
    skipped = sum(1 for d in out if d is None)
    if skipped:
        logging.warning("pi05 construct: skipped %d unusable repos", skipped)
    return [d for d in out if d is not None]


class MultiLatentLeRobotPi05DeltaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        num_init_worker = int(getattr(config, "num_init_worker", num_init_worker))
        self._datasets = construct_lerobot_pi05_delta_multi_processor(
            config,
            num_init_worker,
        )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(self):
        return sum(len(v) for v in self._datasets)

    @property
    def sample_signatures(self):
        """Per-global-index (n_cam, n_tac) shape signature for batch bucketing.

        Built in global-index order (dataset 0's items, then dataset 1's, ...),
        matching _get_item_id_to_dataset_id's sequential idx assignment, so
        ``sample_signatures[i]`` is the signature of sample ``i``. Samples with
        the same signature have identical tensor shapes and are collatable into
        a batch (cameras 1/3 and tactile streams 0/2/4 are what vary; frames and
        action dims are uniform pool-wide).
        """
        sigs = []
        for dset in self._datasets:
            sig = (len(dset.used_video_keys), len(dset.used_tactile_keys))
            sigs.extend([sig] * len(dset))
        return sigs

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        idx = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[idx] = dset_id
                idx += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def _getitem_one(self, idx):
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        # Robust to TRANSIENT shared storage read failures in a sub-dataset's hot path (empty/
        # partial parquet reads -> e.g. "action must be a 2D array, got shape (0,)")
        # which otherwise crash the whole multi-node run. Retry the same item (transient
        # read recovers), then step to a neighbour (skip a persistently-bad segment).
        n = len(self)
        last = None
        for off in range(0, 8):
            j = (idx + off) % n
            for _attempt in range(2):
                try:
                    return self._getitem_one(j)
                except Exception as exc:  # noqa: BLE001
                    last = exc
            if off == 0:
                logging.warning("pi05 getitem idx=%s failed (%s); stepping to neighbour",
                                idx, last)
        raise last


class Pi05DeltaLatentLeRobotDataset(LatentLeRobotDataset):
    """LatentLeRobotDataset with OpenPI/pi0.5 action target construction."""

    def __init__(self, repo_id, config=None):
        super().__init__(repo_id=repo_id, config=config)

        self.pi05_state_column = getattr(
            config, "pi05_state_column", "observation.state"
        )
        self.pi05_action_horizon = int(
            getattr(config, "pi05_action_horizon", getattr(config, "action_per_frame", 16))
        )
        self.pi05_source_action_is_delta = bool(
            getattr(config, "pi05_source_action_is_delta", True)
        )
        self.pi05_condition_first_frame_zero = bool(
            getattr(config, "pi05_condition_first_frame_zero", True)
        )
        self.pi05_condition_first_frame_loss = bool(
            getattr(config, "pi05_condition_first_frame_loss", False)
        )
        self.pi05_invalid_horizon_mask = bool(
            getattr(config, "pi05_invalid_horizon_mask", True)
        )

        if hasattr(config, "pi05_delta_channel_ids"):
            self.pi05_delta_channel_ids = [
                int(v) for v in getattr(config, "pi05_delta_channel_ids")
            ]
        else:
            delta_dims = int(getattr(config, "pi05_delta_dims", 9))
            self.pi05_delta_channel_ids = list(range(delta_dims))
        # When True, the rotation-6D dims of each EE arm use a RELATIVE rotation
        # delta (R_anchor^T · R_target -> first two columns) instead of element-wise
        # subtraction (geometrically correct on SO(3); position/gripper unchanged).
        self.pi05_rot6d_relative_delta = bool(
            getattr(config, "pi05_rot6d_relative_delta", False))
        self.pi05_ee_dims_per_arm = int(getattr(config, "pi05_ee_dims_per_arm", 10))
        self.pi05_ee_rot_offset = int(getattr(config, "pi05_ee_rot_offset", 3))

        available_columns = set(getattr(self.hf_dataset, "column_names", []))
        if self.pi05_state_column not in available_columns:
            raise KeyError(
                f"pi05 state column {self.pi05_state_column!r} not found in "
                f"{self.repo_id}; available columns: {sorted(available_columns)}"
            )

        self._hf_torch_view = self.hf_dataset.with_format(
            type="torch",
            columns=["action", self.pi05_state_column],
            output_all_columns=False,
        )

    @staticmethod
    def _to_numpy_2d(value, name):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        else:
            value = np.asarray(value)
        if value.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got shape {value.shape}")
        return value.astype(np.float32, copy=False)

    @staticmethod
    def _latent_anchor_frame_ids(latent_frame_ids):
        if torch.is_tensor(latent_frame_ids):
            frame_ids = latent_frame_ids.detach().cpu().numpy().astype(np.int64)
        else:
            frame_ids = np.asarray(latent_frame_ids, dtype=np.int64)
        latent_frame_num = (len(frame_ids) - 1) // 4 + 1
        anchor_ids = np.asarray(
            [frame_ids[min(i * 4, len(frame_ids) - 1)] for i in range(latent_frame_num)],
            dtype=np.int64,
        )
        return anchor_ids, latent_frame_num

    def _absolute_targets_from_source(self, action, state):
        absolute = action.copy()
        if self.pi05_source_action_is_delta:
            delta_ids = [i for i in self.pi05_delta_channel_ids if i < action.shape[1]]
            if delta_ids:
                absolute[:, delta_ids] = state[:, delta_ids] + action[:, delta_ids]
        return absolute

    def _build_pi05_delta_actions(
        self,
        local_start_frame,
        latent_frame_ids,
        action,
        state,
    ):
        action = self._to_numpy_2d(action, "action")
        state = self._to_numpy_2d(state, self.pi05_state_column)

        seq_len = min(action.shape[0], state.shape[0])
        action = action[:seq_len]
        state = state[:seq_len]

        absolute_targets = self._absolute_targets_from_source(action, state)
        anchor_ids, latent_frame_num = self._latent_anchor_frame_ids(latent_frame_ids)

        horizon = self.pi05_action_horizon
        action_dim = action.shape[1]
        out = np.zeros((latent_frame_num, horizon, action_dim), dtype=np.float32)
        mask = np.zeros((latent_frame_num, horizon, action_dim), dtype=bool)

        if not self.pi05_condition_first_frame_zero:
            first_pred_frame = 0
        else:
            first_pred_frame = 1
            if self.pi05_condition_first_frame_loss and latent_frame_num > 0:
                mask[0, :, :] = True

        delta_ids = [i for i in self.pi05_delta_channel_ids if i < action_dim]
        # Split delta dims into rotation-6D groups (relative transform) and the rest
        # (position; world-frame subtraction). gripper is not a delta dim.
        rel_rot = self.pi05_rot6d_relative_delta
        rot_groups = []
        if rel_rot:
            for _pos, rot in ee_arm_groups(action_dim, self.pi05_ee_dims_per_arm,
                                           self.pi05_ee_rot_offset):
                if all(i in delta_ids for i in rot):
                    rot_groups.append(rot)
            rot_id_set = {i for g in rot_groups for i in g}
            sub_ids = [i for i in delta_ids if i not in rot_id_set]   # position dims
        else:
            sub_ids = delta_ids
        for frame_idx in range(first_pred_frame, latent_frame_num):
            anchor_frame_idx = frame_idx - 1 if self.pi05_condition_first_frame_zero else frame_idx
            anchor = int(anchor_ids[anchor_frame_idx] - local_start_frame)
            if anchor < 0 or anchor >= seq_len:
                continue

            if self.pi05_invalid_horizon_mask:
                valid = min(horizon, seq_len - anchor)
                if valid <= 0:
                    continue
                target = absolute_targets[anchor : anchor + valid].copy()
                valid_slice = slice(0, valid)
            else:
                target_indices = np.minimum(
                    np.arange(anchor, anchor + horizon),
                    seq_len - 1,
                )
                target = absolute_targets[target_indices].copy()
                valid_slice = slice(0, horizon)

            if sub_ids:
                target[:, sub_ids] -= state[anchor, sub_ids]          # position (world frame)
            for rot in rot_groups:                                     # rotation: relative SO(3)
                R_anchor = sixd_to_R(state[anchor, rot])               # (3,3)
                R_tgt = sixd_to_R(target[:, rot])                      # (V,3,3)
                target[:, rot] = R_to_sixd(np.matmul(R_anchor.T, R_tgt))  # R_anchor^T R_target
            out[frame_idx, valid_slice, :] = target
            mask[frame_idx, valid_slice, :] = True

        return out, mask

    def _action_post_process(
        self,
        local_start_frame,
        local_end_frame,
        latent_frame_ids,
        action,
        state,
    ):
        del local_end_frame
        action, action_mask = self._build_pi05_delta_actions(
            local_start_frame=local_start_frame,
            latent_frame_ids=latent_frame_ids,
            action=action,
            state=state,
        )

        latent_frame_num, horizon, action_dim = action.shape
        action = action.reshape(latent_frame_num * horizon, action_dim)
        action_mask = action_mask.reshape(latent_frame_num * horizon, action_dim)

        action_paded = np.pad(action, ((0, 0), (0, 1)), mode="constant", constant_values=0)
        action_mask_padded = np.pad(
            action_mask,
            ((0, 0), (0, 1)),
            mode="constant",
            constant_values=0,
        )

        action_aligned = action_paded[:, self.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
            self.q99 - self.q01 + 1e-6
        ) * 2.0 - 1.0
        action_aligned = action_aligned.reshape(latent_frame_num, horizon, -1)
        action_mask_aligned = action_mask_aligned.reshape(latent_frame_num, horizon, -1)
        action_aligned = np.transpose(action_aligned, (2, 0, 1))[:, :, :, None]
        action_mask_aligned = np.transpose(action_mask_aligned, (2, 0, 1))[:, :, :, None]
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

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

        start_lat = None
        end_lat = None
        max_latent_frames = int(getattr(self.config, "max_latent_frames", 0))
        if max_latent_frames > 0 and num_latent_frames > max_latent_frames:
            start_lat = torch.randint(0, num_latent_frames - max_latent_frames + 1, (1,)).item()
            end_lat = start_lat + max_latent_frames
            for key in self.used_video_keys:
                f_num = ori_data_dict[f"{key}.latent_num_frames"]
                height = ori_data_dict[f"{key}.latent_height"]
                width = ori_data_dict[f"{key}.latent_width"]
                latent = ori_data_dict[f"{key}.latent"].reshape(f_num, height * width, -1)
                ori_data_dict[f"{key}.latent"] = latent[start_lat:end_lat].reshape(
                    -1,
                    latent.shape[-1],
                )
                ori_data_dict[f"{key}.latent_num_frames"] = max_latent_frames
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

        if self.has_tactile_condition and self.used_tactile_keys:
            tactile_payload = self._load_tactile_latents(
                episode_index=episode_index,
                local_start_frame=local_start_frame,
                local_end_frame=local_end_frame,
                latent_frame_ids=latent_frame_ids,
                truncate_start=start_lat,
                truncate_end=end_lat,
                # tactile is frame-aligned with video — full lengths must match
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
                # tactile_optional: fall back to the CFG-drop path.
                out_dict["tactile_cond_drop"] = torch.tensor(True, dtype=torch.bool)
            else:
                # Keep the real tensors in the sample so missing files are still
                # caught above. The model uses this explicit flag to skip tactile
                # tokens entirely for CFG drop, so tactile modules get no gradient.
                tactile_cfg_prob = float(getattr(self.config, "tactile_cfg_prob", 0.1))
                tactile_cond_drop = torch.rand(1).item() < tactile_cfg_prob

                out_dict["tactile_global_latent"] = tactile_payload["global"]
                out_dict["tactile_local_latent"] = tactile_payload["local"]
                out_dict["tactile_sensor_ids"] = tactile_payload["sensor_ids"]
                out_dict["tactile_cond_drop"] = torch.tensor(
                    tactile_cond_drop, dtype=torch.bool)
        else:
            # Repo with no tactile sensors (per_repo_tactile_keys = []).
            out_dict["tactile_cond_drop"] = torch.tensor(True, dtype=torch.bool)

        state = ori_data_dict[self.pi05_state_column]
        out_dict["actions"], out_dict["actions_mask"] = self._action_post_process(
            local_start_frame,
            local_end_frame,
            latent_frame_ids,
            ori_data_dict["action"],
            state,
        )

        out_dict["latents"] = out_dict["latents"].permute(3, 0, 1, 2)
        return out_dict


__all__ = [
    "Pi05DeltaLatentLeRobotDataset",
    "MultiLatentLeRobotPi05DeltaDataset",
    "construct_lerobot_pi05_delta",
    "construct_lerobot_pi05_delta_multi_processor",
]
