"""Shared Atari archive readers, frame conversion, and content fingerprints."""

import hashlib
import io
import math
from pathlib import Path
import pickle
import struct

import numpy as np
import torch


FRAME_SHAPES = ((210, 160, 3), (84, 84, 3))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_indexed_archive(stream):
    """Map v2 pixels between the pickle header and the trailing episode index."""
    pixel_offset = stream.tell()
    footer_offset = stream.seek(-8, 2)
    index_offset, = struct.unpack("<Q", stream.read(8))
    if not pixel_offset < index_offset < footer_offset:
        raise ValueError("Invalid native archive index offset")
    stream.seek(index_offset)
    index = pickle.load(stream)
    if stream.tell() != footer_offset:
        raise ValueError("Invalid native archive index size")
    meta = index["metadata"]
    shape = tuple(meta["frame_shape"])
    count = meta["frame_count"]
    if (shape not in FRAME_SHAPES or type(count) is not int or count <= 0
            or count * math.prod(shape) != index_offset - pixel_offset):
        raise ValueError("Native archive frame dimensions disagree with stored pixels")
    frames = np.memmap(stream, mode="r", offset=pixel_offset, dtype=np.uint8, shape=(count, *shape))
    start = 0
    for record in index["episodes"]:
        length = record["frame_count"]
        if type(length) is not int or length < 2 or start + length > count:
            raise ValueError("Invalid native archive episode length")
        record["frames"] = frames[start:start + length]
        start += length
    if start != count:
        raise ValueError("Episode index does not cover the native archive")
    return index


def load_archive(path):
    """Read a trusted Atari archive or a list of episode dictionaries.

    Frames are native Atari RGB or legacy RGB84. Optional actions have N entries
    for N+1 frames, including the reset and last observed screen. The
    classifier does not require actions, but the supervised agent does.
    """
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
        if (isinstance(payload, dict) and payload.get("kind") == "atari_episode_archive"
                and payload.get("format_version") == 2):
            payload = _load_indexed_archive(stream)
    if isinstance(payload, list):
        records, metadata = payload, {}
    elif isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        if payload.get("kind") == "atari_episode_archive" and payload.get("format_version") != 1:
            raise ValueError("Unsupported Atari archive version")
        records, metadata = payload["episodes"], dict(payload.get("metadata", {}))
    else:
        raise ValueError("Expected an episode list or a dictionary with an episodes list")
    episodes, ids = [], set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("Each episode must be a dictionary")
        episode = {key: value for key, value in record.items() if key != "payload_npz"}
        if "payload_npz" in record:
            with np.load(io.BytesIO(record["payload_npz"]), allow_pickle=False) as arrays:
                episode.update({key: arrays[key] for key in arrays.files})
        frames = np.asarray(episode.get("frames"))
        if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] not in FRAME_SHAPES
                or len(frames) < 2):
            raise ValueError("Each trajectory needs at least two uint8 RGB frames of shape 210x160x3 or 84x84x3")
        if not episode.get("complete", True) and not episode.get("collection_cutoff", False):
            raise ValueError("An incomplete trajectory must be marked as a collection cutoff")
        source_id = episode.get("id", index)
        if (not isinstance(source_id, (int, np.integer)) or isinstance(source_id, (bool, np.bool_))
                or int(source_id) in ids):
            raise ValueError("Episode IDs must be distinct integers")
        episode.update(id=int(source_id), frames=np.ascontiguousarray(frames), complete=episode.get("complete", True))
        ids.add(episode["id"])
        for key in ("actions", "rewards", "terminated", "truncated"):
            if episode.get(key) is None:
                continue
            values = np.asarray(episode[key])
            if values.shape != (len(frames) - 1,):
                raise ValueError(f"{key} must align with frames[t] -> frames[t+1]")
            if key == "actions" and (not np.issubdtype(values.dtype, np.integer) or np.any(values < 0)):
                raise ValueError("Actions must be nonnegative integer indices")
            if key == "rewards" and not np.isfinite(values).all():
                raise ValueError("Rewards must be finite")
            if key in ("terminated", "truncated") and values.dtype != np.bool_:
                raise ValueError("Termination flags must be boolean")
            episode[key] = values
        if "frame_count" in episode and int(episode["frame_count"]) != len(frames):
            raise ValueError("Compressed archive frame count differs from its payload")
        episodes.append(episode)
    if not episodes:
        raise ValueError("The archive contains no episodes")
    if all("end_step" in episode for episode in episodes):
        episodes.sort(key=lambda episode: (episode["end_step"], episode["id"]))
    return episodes, metadata


def frames_to_tensor(frames, device="cpu"):
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] not in FRAME_SHAPES:
        raise ValueError("Expected uint8 RGB frames [B,210,160,3] or [B,84,84,3]")
    return torch.from_numpy(np.array(frames, copy=True)).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).div_(255)


def state_digest(digest, state):
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}:".encode())
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest
