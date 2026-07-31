#!/usr/bin/env python3
"""Closed-loop N0-TWAM client: drive the policy from your own robot or simulator.

`simple_client.py` is an **open-loop** smoke test — it sends a fixed frame a few
times and prints the action chunks. That is enough to check the serving stack,
but it is not how the model is meant to be run.

The server is *streaming and stateful*: the action chunk it returns is based on
a future it **imagined**. After executing that chunk you must feed back what was
actually observed, so the model's KV cache is re-grounded on reality. Skip that
step and the policy silently degrades to open-loop — nothing raises, the numbers
just quietly get worse.

This file implements that loop without a simulator, so you can run it from your
own robot. It needs only **numpy, websockets and msgpack** — no torch, no Isaac.
(For the NeoSim benchmark, use its `eval/` clients instead; see docs/DEPLOY.md.)

Per episode::

    reset  ->  [ infer_chunk  ->  execute slots  ->  commit_kv_cache ]  x N

:meth:`TwamClient.run_chunk` performs one bracketed iteration, calling back into
your code to execute an action and to read an observation::

    from example_client.closed_loop_client import TwamClient

    client = TwamClient("127.0.0.1", 29601, prompt="Lift the can", num_arms=1)
    client.reset(seed=100)

    while not robot.done:
        cams, tactile = robot.observe()      # {"top": HxWx3}, {"tactile_a": HxWx3}
        state = client.encode_state([robot.ee_pose()])
        result = client.run_chunk(cams, tactile, state,
                                  execute=robot.move_to, observe=robot.observe)
        if result.stopped:
            break

``robot.observe()`` returns ``(cams, tactile)`` dicts keyed by short names
(``top`` / ``wrist_l`` / ``wrist_r``, ``tactile_a`` / ``tactile_b`` ...); this
client prefixes them into the ``observation.images.*`` keys the serve config
declares. ``robot.move_to(poses, ctx)`` executes one action slot and returns
``True`` to abort the chunk (task solved, step budget exhausted, ...).

Run it standalone for a closed-loop smoke test against a live server::

    python example_client/closed_loop_client.py --port 29601 --chunks 2

Alignment requirements (silent-failure territory — verify before trusting any
number):

* the camera/tactile keys must match the serve config's ``obs_cam_keys`` /
  ``tactile_keys``;
* the tactile images must be the *same representation* the checkpoint was
  trained on;
* ``prompt`` must be the training instruction verbatim.

NOTE: unset http_proxy / https_proxy in the client shell — Python websockets
does not honor NO_PROXY CIDRs and a proxy breaks the handshake, with an error
("did not receive a valid HTTP response") that points nowhere near the cause.

Extending it
------------
This is a starting point to build on, not a sealed library. Adapt it by
subclassing :class:`TwamClient` and overriding one of these — the surrounding
sequencing (which is the part that fails silently) keeps working:

===================================  =====================================
Override                             When
===================================  =====================================
:meth:`TwamClient.pack_images`       your camera pipeline needs resizing,
                                     cropping, color conversion, de-bayering
:meth:`TwamClient.is_keyframe_slot`  sample observations on contact events
                                     or at a rate your hardware sustains
:meth:`TwamClient.first_frame_index` a different cold-start convention
:meth:`ActionChunk.poses_at`         a non ``[xyz, rot6d, gripper]`` action
                                     space (e.g. joint targets)
``arm_dim`` / ``state_dim``          a checkpoint with a different width
``cam_names`` / ``tactile_names``    a serve config with different keys
                                     (constructor arg, no subclass needed)
===================================  =====================================

Two things to preserve if you rewrite :meth:`TwamClient.run_chunk` wholesale: a
chunk must be re-grounded via ``compute_kv_cache`` after execution, and a failed
commit must abort the episode. Both failures are silent.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Import the transport as a TOP-LEVEL package (rather than
# `n0_twam.utils.Simple_Remote_Infer...`) so that neither `n0_twam/__init__.py`
# nor `n0_twam/utils/__init__.py` runs — both eagerly import the training stack
# and would drag in torch/diffusers. This client needs only numpy + websockets
# + msgpack, and must stay importable on the machine driving the robot, which
# typically cannot host the training dependencies at all.
#
# APPEND, never insert(0): `n0_twam/utils/` contains a `logging.py`, which at the
# front of sys.path shadows the standard library's and breaks every `import
# logging` in the process (including the transport's own).
sys.path.append(str(REPO / "n0_twam" / "utils"))

from Simple_Remote_Infer.deploy.websocket_client_policy import (  # noqa: E402
    WebsocketClientPolicy,
)

__all__ = [
    "EEPose",
    "SlotContext",
    "ActionChunk",
    "ChunkResult",
    "TwamClient",
    "rot6d10_to_pose",
    "pose_to_rot6d10",
    "to_uint8",
]

#: Dims one arm occupies in the action/state vector: xyz(3) + rot6d(6) + gripper(1).
ARM_DIM = 10

#: Server-side state/action vector width (``cfg.action_dim``). Two arms fit; a
#: single-arm setup uses the first 10 dims and leaves the rest zero.
STATE_DIM = 20

#: Short name -> serve-config key. Mirrors ``twam_posttrain_cfg``'s defaults.
IMAGE_KEY_PREFIX = "observation.images."

DEFAULT_CAM_NAMES: Dict[int, Tuple[str, ...]] = {
    1: ("top", "wrist_l"),
    2: ("top", "wrist_l", "wrist_r"),
}

DEFAULT_TACTILE_NAMES: Dict[int, Tuple[str, ...]] = {
    1: ("tactile_a", "tactile_b"),
    2: ("tactile_ll", "tactile_lr", "tactile_rl", "tactile_rr"),
}


# --------------------------------------------------------------------------
# rot6d <-> pose
# --------------------------------------------------------------------------
def _matrix_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), Shepperd's method.

    Hand-rolled rather than via scipy so that a client needs nothing beyond
    numpy; branch selection on the largest diagonal term keeps it stable near
    the 180-degree singularities a naive trace-only formula blows up on.
    """
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = (0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = ((m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _quat_wxyz_to_matrix(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"quaternion must have 4 elements, got {q.shape[0]}")
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


@dataclass
class EEPose:
    """One arm's end-effector target: position, orientation, gripper.

    ``quat_wxyz`` is scalar-first. ``gripper`` carries whatever convention the
    training data used (the UniVTAC/NeoSim lines use a normalized opening).
    """
    xyz: np.ndarray
    quat_wxyz: np.ndarray
    gripper: float

    def as_tuple(self) -> Tuple[float, ...]:
        """Flat ``(x, y, z, qw, qx, qy, qz, gripper)`` — what most IK APIs take."""
        return (*(float(v) for v in self.xyz),
                *(float(v) for v in self.quat_wxyz), float(self.gripper))


def rot6d10_to_pose(vec: Sequence[float]) -> EEPose:
    """Decode one arm's 10-dim ``[xyz(3), rot6d(6), gripper(1)]`` slice into a pose.

    The 6D rotation is the first two columns of the rotation matrix; they are
    re-orthonormalized (Gram-Schmidt) because the model's continuous output is
    only approximately orthogonal. Column 3 is their cross product, which fixes
    the handedness — do not drop this step, a raw reshape yields a non-rotation
    matrix and a garbage quaternion.
    """
    d = np.asarray(vec, dtype=np.float64).reshape(-1)
    if d.shape[0] < ARM_DIM:
        raise ValueError(f"expected >= {ARM_DIM} dims per arm, got {d.shape[0]}")
    xyz = d[:3].copy()
    c1, c2 = d[3:6].copy(), d[6:9].copy()
    b1 = c1 / (np.linalg.norm(c1) + 1e-8)
    c2 = c2 - (b1 @ c2) * b1
    b2 = c2 / (np.linalg.norm(c2) + 1e-8)
    b3 = np.cross(b1, b2)
    rot = np.stack([b1, b2, b3], axis=1)
    return EEPose(xyz=xyz, quat_wxyz=_matrix_to_quat_wxyz(rot),
                  gripper=float(d[9]))


def pose_to_rot6d10(pose: EEPose) -> np.ndarray:
    """Encode a pose back into a 10-dim ``[xyz(3), rot6d(6), gripper(1)]`` slice.

    Inverse of :func:`rot6d10_to_pose`; used to build ``current_state``.
    """
    rot = _quat_wxyz_to_matrix(pose.quat_wxyz)
    out = np.zeros(ARM_DIM, dtype=np.float32)
    out[:3] = np.asarray(pose.xyz, dtype=np.float64).reshape(-1)[:3]
    out[3:9] = rot[:, :2].T.reshape(-1)   # [col0(3), col1(3)]
    out[9] = float(pose.gripper)
    return out


# --------------------------------------------------------------------------
# image normalization
# --------------------------------------------------------------------------
def to_uint8(img, scale: str = "auto") -> np.ndarray:
    """Coerce an image buffer to contiguous ``uint8`` HxWx3.

    ``scale`` controls how float inputs are mapped:

    * ``"auto"`` — infer from the value range. Float camera buffers usually
      arrive in [0, 1] while some tactile renderers emit [0, 255] floats; a
      blind ``*255`` would overflow the latter into solid white. The heuristic
      is ``max() > 1.5 => already byte-ranged`` (the same test the server
      applies to tactile), which **misfires on a genuinely dark [0, 1] image
      whose max is small only by coincidence** — indistinguishable without
      knowing the source, so pass ``"unit"``/``"byte"`` when you do know.
    * ``"unit"`` — input is [0, 1], multiply by 255.
    * ``"byte"`` — input is already [0, 255], only clip and cast.
    """
    x = np.asarray(img.cpu() if hasattr(img, "cpu") else img)
    x = np.squeeze(x)
    if x.ndim == 3 and x.shape[-1] > 3:
        x = x[..., :3]           # drop alpha
    if x.dtype == np.uint8:
        return np.ascontiguousarray(x)
    if scale == "auto":
        factor = 1.0 if float(x.max(initial=0.0)) > 1.5 else 255.0
    elif scale == "unit":
        factor = 255.0
    elif scale == "byte":
        factor = 1.0
    else:
        raise ValueError(f"unknown scale mode {scale!r}")
    return np.ascontiguousarray((x * factor).clip(0, 255).astype(np.uint8))


# --------------------------------------------------------------------------
# chunk types
# --------------------------------------------------------------------------
@dataclass
class SlotContext:
    """Where an action slot sits inside the chunk, handed to the execute callback."""
    frame: int              #: frame index within the chunk
    slot: int               #: action slot within the frame
    slot_index: int         #: running index over the slots actually executed
    is_keyframe: bool       #: an observation is sampled right after this slot


@dataclass
class ActionChunk:
    """A decoded server response: raw ``(C, F, H)`` actions plus per-arm poses.

    ``C`` = action dims (20), ``F`` = frames, ``H`` = action slots per frame.
    Values are already de-normalized by the server; with the default
    ``server_action_output_format = "absolute"`` each slot is a directly
    executable absolute EE target.
    """
    raw: np.ndarray
    num_arms: int
    server_timing: Optional[Dict] = None
    #: Dims per arm. Override together with the client's ``arm_dim`` if your
    #: checkpoint uses a different per-arm layout.
    arm_dim: int = ARM_DIM

    @property
    def num_frames(self) -> int:
        return int(self.raw.shape[1])

    @property
    def slots_per_frame(self) -> int:
        return int(self.raw.shape[2])

    def poses_at(self, frame: int, slot: int) -> List[EEPose]:
        """Decode one slot into one pose per arm.

        Override (with a subclass) if your action space is not
        ``[xyz, rot6d, gripper]`` per arm — e.g. joint targets.
        """
        vec = np.asarray(self.raw[:, frame, slot]).reshape(-1)
        return [
            rot6d10_to_pose(vec[a * self.arm_dim:(a + 1) * self.arm_dim])
            for a in range(self.num_arms)
        ]


@dataclass
class ChunkResult:
    """Outcome of one :meth:`TwamClient.run_chunk` bracket."""
    chunk: ActionChunk
    executed_slots: int
    #: True if the execute callback aborted the chunk (task done / budget spent).
    stopped: bool = False
    #: False when the abort happened before the KV cache could be re-grounded.
    committed: bool = False
    keyframes: List[Dict] = field(default_factory=list)


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------
class TwamClient:
    """Streaming closed-loop client for the N0-TWAM policy server.

    Parameters
    ----------
    host, port
        Where the server listens. Cross-machine and SSH-tunnelled links work;
        unset ``http_proxy``/``https_proxy`` on the client host first.
    prompt
        The training instruction, **verbatim**. Sent with every request; a
        paraphrase silently changes the conditioning.
    num_arms
        1 or 2. Selects the default key sets and how many 10-dim slices each
        action vector is split into.
    cam_names, tactile_names
        Override the short names (and hence the ``observation.images.*`` keys).
        Must match the serve config's ``obs_cam_keys`` / ``tactile_keys``.
    video_keyframes_per_frame
        Observations sampled per frame while executing, fed back for
        re-grounding. 4 matches the NeoSim reference client.
    tactile_keyframes
        Keep only the last N tactile keyframes at commit time (0 = all). Note
        that trimming makes the tactile stream shorter than the video stream;
        they advance separate streaming VAEs, so their time axes then differ.
    exec_slots
        Execute only the first N slots of each frame (0 = all). Truncating means
        re-planning more often, at higher inference cost.
    image_scale
        Passed to :func:`to_uint8` for every image.
    client
        Inject a pre-built transport (or a test double). Constructed from
        ``host``/``port`` when omitted.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        *,
        prompt: str,
        num_arms: int = 1,
        cam_names: Optional[Sequence[str]] = None,
        tactile_names: Optional[Sequence[str]] = None,
        video_keyframes_per_frame: int = 4,
        tactile_keyframes: int = 0,
        exec_slots: int = 0,
        image_scale: str = "auto",
        api_key: Optional[str] = None,
        client: Optional[WebsocketClientPolicy] = None,
    ) -> None:
        if num_arms not in (1, 2):
            raise ValueError(f"num_arms must be 1 or 2, got {num_arms}")
        if not prompt:
            raise ValueError(
                "prompt is required and must be the training instruction "
                "verbatim — there is no safe default")

        self.prompt = prompt
        self.num_arms = num_arms
        self.cam_names = tuple(cam_names or DEFAULT_CAM_NAMES[num_arms])
        self.tactile_names = tuple(tactile_names
                                   or DEFAULT_TACTILE_NAMES[num_arms])
        self.video_keyframes_per_frame = max(1, int(video_keyframes_per_frame))
        self.tactile_keyframes = int(tactile_keyframes)
        self.exec_slots = int(exec_slots)
        self.image_scale = image_scale
        # Instance attributes, not module constants, so a subclass serving a
        # checkpoint with a different action layout can change them without
        # forking this file.
        self.arm_dim = ARM_DIM
        self.state_dim = STATE_DIM
        self.image_key_prefix = IMAGE_KEY_PREFIX

        self._client = client if client is not None else WebsocketClientPolicy(
            host=host, port=port, api_key=api_key)
        # After a reset the server holds a 1-frame cold seed that already
        # corresponds to frame 0 of the next chunk; executing frame 0 again
        # would replay the current pose as a target and stall the arm.
        self._cold_chunk = True

    # -- transport -------------------------------------------------------
    @property
    def server_metadata(self) -> Dict:
        return self._client.get_server_metadata()

    def full_keys(self, names: Sequence[str]) -> List[str]:
        """Short names -> the keys the serve config declares."""
        return [
            n if n.startswith(self.image_key_prefix) else
            self.image_key_prefix + n for n in names
        ]

    def pack_images(self, images: Dict, names: Sequence[str],
                    kind: str) -> Dict[str, np.ndarray]:
        """Short-name dict -> ``observation.images.*`` dict of uint8 frames.

        Missing or unexpected keys raise here rather than reaching the server as
        a silently wrong observation: the server takes each configured key out
        of the dict, so an extra key is **ignored without any report** and only
        a missing one fails.

        **Override point.** Every image the client sends passes through here, so
        this is where to plug in resizing, cropping, color conversion or
        de-bayering. Keep the contract: return a dict keyed by
        :meth:`full_keys` whose values are contiguous ``uint8`` HxWx3 arrays in
        [0, 255] — the server maps them with ``/255*2-1``.
        """
        packed = {}
        for name, full in zip(names, self.full_keys(names)):
            if name in images:
                value = images[name]
            elif full in images:
                value = images[full]
            else:
                raise KeyError(
                    f"{kind} observation is missing {name!r} (expected keys: "
                    f"{list(names)}, got: {sorted(images)})")
            packed[full] = to_uint8(value, self.image_scale)
        extra = set(images) - set(names) - set(self.full_keys(names))
        if extra:
            raise KeyError(
                f"{kind} observation has unexpected keys {sorted(extra)}; "
                f"expected exactly {list(names)}. A key the serve config does "
                f"not declare is dropped server-side, not reported.")
        return packed

    # -- state helpers ---------------------------------------------------
    def encode_state(self, poses: Sequence[EEPose]) -> np.ndarray:
        """Build the 20-dim ``current_state`` vector from one pose per arm.

        Arm 0 occupies dims 0-9, arm 1 dims 10-19; unused dims stay zero.
        """
        if len(poses) != self.num_arms:
            raise ValueError(
                f"expected {self.num_arms} pose(s), got {len(poses)}")
        state = np.zeros(self.state_dim, dtype=np.float32)
        for i, pose in enumerate(poses):
            state[i * self.arm_dim:(i + 1) * self.arm_dim] = pose_to_rot6d10(pose)
        return state

    def _as_state_list(self, state) -> List[float]:
        vec = np.asarray(state, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.state_dim:
            raise ValueError(
                f"current_state must be {self.state_dim}-dim, got {vec.shape[0]}")
        return [float(v) for v in vec]

    # -- protocol steps --------------------------------------------------
    def reset(self, prompt: Optional[str] = None,
              seed: Optional[int] = None) -> None:
        """Start a new episode: clear the server's KV cache and re-encode the prompt.

        ``seed`` makes the diffusion noise reproducible for that episode
        (honored when the serve config leaves ``deterministic_episode_seed`` on).
        """
        if prompt is not None:
            self.prompt = prompt
        request = {"reset": True, "prompt": self.prompt}
        if seed is not None:
            request["seed"] = int(seed)
        self._client.infer(request)
        self._cold_chunk = True

    def infer_chunk(self, cams: Dict, tactile: Dict, current_state) -> ActionChunk:
        """Request one action chunk for the current observation.

        Sends a single (current) frame per camera and tactile sensor: the
        server's cold seed takes exactly one frame, and more would break its
        streaming VAE shortcut. Multi-frame history goes through
        :meth:`commit_kv_cache` instead.
        """
        response = self._client.infer({
            "obs": self.pack_images(cams, self.cam_names, "camera"),
            "tactile": self.pack_images(tactile, self.tactile_names, "tactile"),
            "current_state": self._as_state_list(current_state),
            "prompt": self.prompt,
        })
        if "action" not in response:
            raise RuntimeError(
                f"server returned no 'action' field (keys: {sorted(response)})")
        raw = np.asarray(response["action"], dtype=np.float32)
        if raw.ndim != 3:
            raise RuntimeError(
                f"expected action shape (C, F, H), got {raw.shape}")
        if raw.shape[0] < self.num_arms * self.arm_dim:
            raise RuntimeError(
                f"action has {raw.shape[0]} channels, too few for "
                f"{self.num_arms} arm(s) (need {self.num_arms * self.arm_dim})")
        return ActionChunk(raw=raw, num_arms=self.num_arms,
                           server_timing=response.get("server_timing"),
                           arm_dim=self.arm_dim)

    def commit_kv_cache(self, video_keyframes: List[Dict],
                        tactile_keyframes: List[Dict], action: np.ndarray,
                        current_state) -> None:
        """Re-ground the KV cache on what was actually observed and executed.

        This is what closes the loop. Skipping it — or swallowing its failure —
        leaves the model running on its own imagined future, i.e. open-loop,
        with no error raised anywhere. Any failure therefore propagates: treat
        the episode as invalid rather than continuing.

        This request advances the server's ``frame_st_id``. The bundled
        transport does not retry, which is exactly the semantics needed: a
        replay after a lost response would double-count the time axis. **If you
        add retries to** ``WebsocketClientPolicy``, exempt this call.
        """
        if self.tactile_keyframes > 0:
            tactile_keyframes = tactile_keyframes[-self.tactile_keyframes:]
        try:
            self._client.infer({
                "obs": video_keyframes,
                "tactile": tactile_keyframes,
                "state": np.asarray(action, dtype=np.float32),
                "current_state": self._as_state_list(current_state),
                "compute_kv_cache": True,
                "imagine": False,
                "prompt": self.prompt,
            })
        except Exception as e:
            raise RuntimeError(
                f"compute_kv_cache commit failed ({type(e).__name__}: {e}); "
                "the episode is now open-loop and must be discarded") from e

    # -- orchestration hooks ---------------------------------------------
    def is_keyframe_slot(self, slot: int, slots_per_frame: int) -> bool:
        """Whether to sample an observation right after executing ``slot``.

        Default: evenly spaced, ``video_keyframes_per_frame`` per frame.
        Override to sample on contact events, on gripper transitions, or at a
        cadence your hardware can actually sustain — the keyframes are what the
        KV cache is re-grounded on, so this is the main quality/latency knob.
        """
        every = max(1, slots_per_frame // self.video_keyframes_per_frame)
        return (slot + 1) % every == 0

    def first_frame_index(self) -> int:
        """Which frame of the chunk to start executing at.

        Frame 0 of the first chunk after a reset is the server's cold seed —
        already the robot's current pose — so executing it replays the present
        and stalls the arm. Every later chunk starts at 0.
        """
        return 1 if self._cold_chunk else 0

    # -- orchestration ---------------------------------------------------
    def run_chunk(
        self,
        cams: Dict,
        tactile: Dict,
        current_state,
        execute: Callable[[List[EEPose], SlotContext], Optional[bool]],
        observe: Callable[[], Tuple[Dict, Dict]],
    ) -> ChunkResult:
        """One full closed-loop bracket: infer -> execute -> re-ground.

        ``execute(poses, ctx)`` receives one :class:`EEPose` per arm and returns
        truthy to abort the chunk (task solved, step budget exhausted, ...).
        ``observe()`` returns the current ``(cams, tactile)`` dicts and is called
        at each keyframe to collect what the arm actually saw.

        On abort the KV cache is deliberately **not** committed: the executed
        prefix does not correspond to a complete chunk, and grounding on it
        would desynchronize the server's time axis from the robot's.
        """
        chunk = self.infer_chunk(cams, tactile, current_state)
        # Freeze the chunk-start state: the commit must carry the SAME anchor
        # the inference used, or the server's delta<->absolute round-trip on
        # pi05_delta channels no longer cancels.
        anchor_state = self._as_state_list(current_state)

        slots = (min(chunk.slots_per_frame, self.exec_slots)
                 if self.exec_slots > 0 else chunk.slots_per_frame)
        start_frame = self.first_frame_index()
        self._cold_chunk = False

        video_keyframes: List[Dict] = []
        tactile_keyframes: List[Dict] = []
        executed = 0

        for frame in range(start_frame, chunk.num_frames):
            for slot in range(slots):
                ctx = SlotContext(
                    frame=frame, slot=slot, slot_index=executed,
                    is_keyframe=self.is_keyframe_slot(slot,
                                                      chunk.slots_per_frame))
                if execute(chunk.poses_at(frame, slot), ctx):
                    return ChunkResult(chunk=chunk, executed_slots=executed + 1,
                                       stopped=True, committed=False,
                                       keyframes=video_keyframes)
                executed += 1
                if ctx.is_keyframe:
                    k_cams, k_tactile = observe()
                    video_keyframes.append(
                        self.pack_images(k_cams, self.cam_names, "camera"))
                    tactile_keyframes.append(
                        self.pack_images(k_tactile, self.tactile_names,
                                         "tactile"))

        committed = False
        if video_keyframes:
            self.commit_kv_cache(video_keyframes, tactile_keyframes, chunk.raw,
                                 anchor_state)
            committed = True
        return ChunkResult(chunk=chunk, executed_slots=executed,
                           stopped=False, committed=committed,
                           keyframes=video_keyframes)

    def run_episode(
        self,
        observe: Callable[[], Tuple[Dict, Dict]],
        get_state: Callable[[], np.ndarray],
        execute: Callable[[List[EEPose], SlotContext], Optional[bool]],
        max_chunks: int = 1000,
        seed: Optional[int] = None,
    ) -> int:
        """Reset, then loop :meth:`run_chunk` until ``execute`` aborts.

        Returns the number of chunks run. Thin convenience wrapper — drive
        :meth:`run_chunk` yourself when you need per-chunk bookkeeping.
        """
        self.reset(seed=seed)
        for i in range(max_chunks):
            cams, tactile = observe()
            if self.run_chunk(cams, tactile, get_state(), execute,
                              observe).stopped:
                return i + 1
        return max_chunks


# --------------------------------------------------------------------------
# standalone smoke test
# --------------------------------------------------------------------------
def _load_example_frames(cam_names, size=256):
    """Bundled example frames; zeros for anything missing."""
    from PIL import Image

    img_dir = REPO / "example_client" / "twam"
    frames = {}
    for name in cam_names:
        p = img_dir / f"{IMAGE_KEY_PREFIX}{name}.png"
        if p.is_file():
            frames[name] = np.array(
                Image.open(p).convert("RGB").resize((size, size)))
        else:
            print(f"warn: {p} not found, sending zeros")
            frames[name] = np.zeros((size, size, 3), np.uint8)
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29601)
    ap.add_argument("--prompt", default="pick up the object",
                    help="MUST be the training instruction verbatim")
    ap.add_argument("--num-arms", type=int, default=1, choices=(1, 2))
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    client = TwamClient(args.host, args.port, prompt=args.prompt,
                        num_arms=args.num_arms)
    print("connected:", client.server_metadata)
    print("cameras:", list(client.cam_names))
    print("tactile:", list(client.tactile_names))

    # No robot here: replay the bundled frames, zero tactile, identity pose.
    cams = _load_example_frames(client.cam_names)
    tactile = {n: np.zeros((256, 256, 3), np.uint8)
               for n in client.tactile_names}
    identity = EEPose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 0.0)
    state = client.encode_state([identity] * args.num_arms)

    def execute(poses, ctx):
        if ctx.slot_index == 0:
            p = poses[0]
            print(f"    slot0 pose: xyz={np.round(p.xyz, 4).tolist()} "
                  f"quat={np.round(p.quat_wxyz, 4).tolist()} "
                  f"grip={p.gripper:+.4f}")
        return False        # a real robot would move here and may abort

    def observe():
        return cams, tactile

    client.reset(seed=args.seed)
    print(f"reset (seed={args.seed})")

    for i in range(args.chunks):
        t0 = time.time()
        r = client.run_chunk(cams, tactile, state, execute, observe)
        a = r.chunk.raw
        print(f"chunk {i}: action {a.shape} "
              f"range [{a.min():+.4f}, {a.max():+.4f}] "
              f"finite={np.isfinite(a).all()} | "
              f"executed {r.executed_slots} slots, "
              f"{len(r.keyframes)} keyframes, committed={r.committed} | "
              f"{time.time() - t0:.1f}s", flush=True)

    print("done — closed loop ran, KV cache was re-grounded each chunk")


if __name__ == "__main__":
    main()
