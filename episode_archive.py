"""Disk-backed completed episodes used by periodic return-policy training.

Only the current unfinished episode lives in each recorder. The archive keeps
compressed completed episodes on disk and a small list of IDs awaiting the next
round. Consuming a round removes IDs from that list, never their source files.
"""

import io
import json
import os
from pathlib import Path
import pickle

import numpy as np
from PIL import Image


def resize_frame(frame):
    """Copy an environment RGB image to the standalone classifier's RGB84 format."""
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("Episode frames must be uint8 HWC RGB images")
    if frame.shape[:2] != (84, 84):
        frame = np.asarray(Image.fromarray(frame).resize((84, 84), Image.Resampling.BILINEAR))
    return np.array(frame, dtype=np.uint8, copy=True, order="C")


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} as archive metadata")


class EpisodeRecorder:
    """Record one actual reset-to-end trajectory; incomplete episodes cannot finish."""

    def __init__(self, initial_frame, **metadata):
        self.metadata = dict(metadata)
        self.frames = [resize_frame(initial_frame)]
        self.actions, self.rewards = [], []
        self.terminated, self.truncated = [], []
        self.ended = False

    def step(self, action, next_frame, reward=0., terminated=False, truncated=False):
        if self.ended:
            raise ValueError("Cannot append to an ended episode")
        action = _integer(action, "Action")
        if action < 0 or not np.isfinite(reward):
            raise ValueError("Action must be nonnegative and reward finite")
        frame = resize_frame(next_frame)
        self.frames.append(frame)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.ended = bool(terminated or truncated)

    def finish(self, **metadata):
        if not self.ended:
            raise ValueError("Only complete environment episodes can be archived")
        return dict(self.metadata | metadata,
                    frames=np.stack(self.frames),
                    actions=np.asarray(self.actions, dtype=np.int64),
                    rewards=np.asarray(self.rewards, dtype=np.float32),
                    terminated=np.asarray(self.terminated, dtype=np.bool_),
                    truncated=np.asarray(self.truncated, dtype=np.bool_),
                    complete=True)


class EpisodeArchive:
    """Append completed episodes and export the pending set in standalone format.

    This is a single-writer store. ``snapshot`` returns the precise IDs exported;
    pass those IDs to ``consume`` only after their round succeeds. Episodes added
    after the snapshot (including its 100 assisted rollouts) remain pending.
    """

    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.records = self.directory / "episodes"
        self.records.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        self.metadata = json.loads(json.dumps(metadata, default=_json_default, allow_nan=False))
        self._pending, self._next_id = [], 0
        if self.manifest_path.exists():
            state = json.loads(self.manifest_path.read_text())
            if state.get("format_version") != 1 or state.get("metadata") != self.metadata:
                raise ValueError("Existing episode archive has different format or metadata")
            self._pending = list(state["pending_ids"])
            self._next_id = int(state["next_id"])
            if (len(set(self._pending)) != len(self._pending)
                    or any(type(i) is not int or not 0 <= i < self._next_id for i in self._pending)):
                raise ValueError("Invalid pending episode IDs in archive manifest")
            if any(not self._path(i).is_file() for i in self._pending):
                raise ValueError("Archive manifest references missing episodes")
        # Recover a completed atomic record write interrupted before manifest commit.
        existing = sorted(int(path.stem) for path in self.records.glob("*.pkl")
                          if path.stem.isdigit() and int(path.stem) >= self._next_id)
        for episode_id in existing:
            if episode_id != self._next_id:
                raise ValueError("Episode archive has a gap in uncommitted IDs")
            record = self._read_record(episode_id)
            if record.get("id") != episode_id or not record.get("complete"):
                raise ValueError("Invalid uncommitted episode record")
            self._pending.append(episode_id)
            self._next_id += 1
        self._save_manifest()

    @property
    def pending_ids(self):
        return tuple(self._pending)

    def _path(self, episode_id):
        return self.records / f"{episode_id:09d}.pkl"

    def _save_manifest(self):
        state = dict(format_version=1, metadata=self.metadata,
                     next_id=self._next_id, pending_ids=self._pending)
        encoded = (json.dumps(state, indent=2, allow_nan=False) + "\n").encode()
        _atomic_write(self.manifest_path, lambda stream: stream.write(encoded))

    def _read_record(self, episode_id):
        with self._path(episode_id).open("rb") as stream:
            return pickle.load(stream)

    def append(self, episode):
        """Persist an aligned complete trajectory and return its monotonically new ID."""
        frames = np.asarray(episode.get("frames"))
        if (episode.get("complete") is not True or frames.dtype != np.uint8
                or frames.ndim != 4 or frames.shape[1:] != (84, 84, 3) or len(frames) < 2):
            raise ValueError("Archive needs a complete trajectory of uint8 RGB84 frames")
        arrays = {"frames": np.ascontiguousarray(frames)}
        length = len(frames) - 1
        for key in ("actions", "rewards", "terminated", "truncated"):
            values = np.asarray(episode.get(key))
            if values.shape != (length,):
                raise ValueError(f"{key} must align with frames[t] -> frames[t+1]")
            arrays[key] = values
        if (not np.issubdtype(arrays["actions"].dtype, np.integer)
                or np.any(arrays["actions"] < 0)):
            raise ValueError("Actions must be nonnegative integer indices")
        action_count = self.metadata.get("num_actions")
        if action_count is not None and np.any(arrays["actions"] >= action_count):
            raise ValueError("Action lies outside the archived action space")
        if not np.isfinite(arrays["rewards"]).all():
            raise ValueError("Rewards must be finite")
        if any(arrays[key].dtype != np.bool_ for key in ("terminated", "truncated")):
            raise ValueError("Termination flags must be boolean")
        endings = arrays["terminated"] | arrays["truncated"]
        if not endings[-1] or endings[:-1].any():
            raise ValueError("A complete episode must end exactly at its final transition")
        compressed = io.BytesIO()
        np.savez_compressed(compressed, **arrays)
        record = {key: value for key, value in episode.items()
                  if key not in arrays and key not in ("payload_npz", "id", "frame_count")}
        episode_id = self._next_id
        record.update(id=episode_id, frame_count=len(frames), payload_npz=compressed.getvalue())
        _atomic_write(self._path(episode_id), lambda stream: pickle.dump(record, stream, protocol=5))
        self._pending.append(episode_id)
        self._next_id += 1
        self._save_manifest()
        return episode_id

    def load(self, episode_id):
        """Decode one persisted episode, including episodes consumed by earlier rounds."""
        episode_id = _integer(episode_id, "Episode ID")
        record = dict(self._read_record(episode_id))
        payload = record.pop("payload_npz")
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            record.update({key: arrays[key] for key in arrays.files})
        return record

    def snapshot(self, path, ids=None):
        """Export a stable pending subset; does not consume it or include unfinished work."""
        ids = self.pending_ids if ids is None else tuple(_integer(i, "Episode ID") for i in ids)
        if not ids or len(set(ids)) != len(ids) or not set(ids).issubset(self._pending):
            raise ValueError("Snapshot needs a nonempty distinct subset of pending episode IDs")
        destination = Path(path).resolve()
        if destination == self.manifest_path.resolve() or destination.parent == self.records.resolve():
            raise ValueError("Snapshot must not replace archive bookkeeping or episode records")
        payload = dict(kind="atari_episode_archive", format_version=1,
                       metadata=self.metadata, episodes=[self._read_record(i) for i in ids])
        _atomic_write(destination, lambda stream: pickle.dump(payload, stream, protocol=5))
        return ids

    def consume(self, ids):
        """Remove exactly a successful round's source IDs from the pending pool."""
        ids = tuple(_integer(i, "Episode ID") for i in ids)
        if len(set(ids)) != len(ids) or not set(ids).issubset(self._pending):
            raise ValueError("Can only consume distinct pending episode IDs")
        consumed = set(ids)
        self._pending = [i for i in self._pending if i not in consumed]
        self._save_manifest()
