"""Completed episodes shared directly by the in-process auxiliary trainers.

Pending episodes stay compressed in RAM during collection. At training time
each episode is decoded once, replacing its compressed record; classifier and
policy training then share those same arrays. Consuming source IDs releases
the pool's references without creating trajectory files or IPC snapshots.
"""

import hashlib
import io
import json

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


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} as archive metadata")


def _json_bytes(value):
    return json.dumps(value, default=_json_default, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode()


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
    """Keep pending episodes in RAM and share one decoded copy with trainers.

    ``materialize`` returns direct references to decoded records, replacing
    their compressed payloads one episode at a time. Returned records are for
    read-only use; their arrays are non-writeable. ``consume`` releases exactly
    the selected pool references. Callers must drop their own references when
    fitting and prefix preparation finish to release the decoded source data.
    """

    def __init__(self, metadata):
        self.metadata = json.loads(json.dumps(metadata, default=_json_default, allow_nan=False))
        self._records, self._next_id = {}, 0
        self._digests = {}

    @property
    def pending_ids(self):
        return tuple(self._records)

    def append(self, episode):
        """Compress a complete trajectory in RAM and return its monotonically new ID."""
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
            arrays[key] = np.ascontiguousarray(values)
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
        record = {key: value for key, value in episode.items()
                  if key not in arrays and key not in ("payload_npz", "id", "frame_count")}
        episode_id = self._next_id
        record.update(id=episode_id, frame_count=len(frames))
        # Own metadata and hash source contents without serializing another
        # full archive or allocating a bytes copy of the image arrays.
        record = json.loads(_json_bytes(record))
        digest = hashlib.sha256(_json_bytes(record))
        for key, values in arrays.items():
            digest.update(_json_bytes((key, values.dtype.str, values.shape)))
            digest.update(memoryview(values).cast('B'))
        compressed = io.BytesIO()
        np.savez_compressed(compressed, **arrays)
        record['payload_npz'] = compressed.getvalue()
        self._records[episode_id] = record
        self._digests[episode_id] = digest.hexdigest()
        self._next_id += 1
        return episode_id

    def load(self, episode_id):
        """Return a shared decoded record, decoding it at most once.

        A failed decode leaves its compressed source intact. After success the
        pool retains only the decoded arrays, not a second compressed copy.
        """
        episode_id = _integer(episode_id, "Episode ID")
        record = self._records[episode_id]
        if 'payload_npz' not in record:
            return record
        with np.load(io.BytesIO(record['payload_npz']), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for values in arrays.values():
            values.setflags(write=False)
        record.update(arrays)
        del record['payload_npz']
        return record

    def _selected_ids(self, ids):
        ids = self.pending_ids if ids is None else tuple(_integer(i, "Episode ID") for i in ids)
        if not ids or len(set(ids)) != len(ids) or not set(ids).issubset(self._records):
            raise ValueError("Need a nonempty distinct subset of pending episode IDs")
        return ids

    def materialize(self, ids=None):
        """Share selected episode dictionaries and arrays with in-process trainers."""
        return [self.load(episode_id) for episode_id in self._selected_ids(ids)]

    def fingerprint(self, ids=None):
        """Identify selected source contents independently of their RAM encoding."""
        ids = self._selected_ids(ids)
        digest = hashlib.sha256(_json_bytes(self.metadata))
        for episode_id in ids:
            digest.update(_json_bytes((episode_id, self._digests[episode_id])))
        return digest.hexdigest()

    def consume(self, ids):
        """Release exactly the completed round's source records from RAM."""
        ids = tuple(_integer(i, "Episode ID") for i in ids)
        if len(set(ids)) != len(ids) or not set(ids).issubset(self._records):
            raise ValueError("Can only consume distinct pending episode IDs")
        for episode_id in ids:
            del self._records[episode_id]
            del self._digests[episode_id]
