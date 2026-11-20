"""Cache the verified final-checkpoint Dreamer posterior in archive frame order."""

from contextlib import closing
import errno
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

from atari_data import file_sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dreamer_latents import replay_latents


def _write_latents(path, episode_index, batches):
    frame_count = sum(length for _, length in episode_index)
    rows = None
    global_row = ep_row = ep_position = 0
    for ep_id, start, batch in batches:
        if ep_position >= len(episode_index):
            raise ValueError("Dreamer replay exceeded the archive length")
        expected_id, length = episode_index[ep_position]
        if (ep_id != expected_id or start != ep_row or batch.ndim != 2 or not len(batch)
                or start + len(batch) > length):
            raise ValueError("Dreamer replay does not align with the archived frames")
        if (batch.dtype != torch.float32 or batch.device.type != "cpu" or not batch.shape[1]
                or not torch.isfinite(batch).all()):
            raise ValueError("Dreamer replay must supply finite float32 vectors")
        if rows is None:
            rows = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                             shape=(frame_count, batch.shape[1]))
        if rows.shape[1] != batch.shape[1]:
            raise ValueError("Dreamer replay changed latent width")
        rows[global_row:global_row + len(batch)] = batch.numpy()
        global_row += len(batch)
        ep_row += len(batch)
        if ep_row == length:
            print(f"Verified and encoded episode {ep_id} ({global_row}/{frame_count} frames)", flush=True)
            ep_position += 1
            ep_row = 0
    if rows is None or ep_position != len(episode_index):
        raise ValueError("Dreamer replay ended before the final archived frame")
    rows.flush()
    return rows.shape[1]


def load_or_replay_latents(cache_dir, episodes, metadata, archive_sha256, checkpoint, config,
                          device="cpu", batch_size=64):
    """Return a copy-on-write CPU tensor backed by the cache and its run fingerprints.

    Publish the complete cache atomically. A rerun with the same archive, config,
    and checkpoint skips the environment and the model entirely.
    """
    cache_dir = Path(cache_dir)
    provenance = dict(archive_sha256=archive_sha256, checkpoint_sha256=file_sha256(checkpoint),
                      config_sha256=file_sha256(config), representation="rssm_deter_and_posterior_mode_v1")
    episode_index = [[ep["id"], len(ep["frames"])] for ep in episodes]
    expected = dict(format_version=1, kind="dreamer_latents", provenance=provenance,
                    episodes=episode_index)
    if not cache_dir.exists():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{cache_dir.name}-", dir=cache_dir.parent) as tmp:
            staged = Path(tmp) / "cache"
            staged.mkdir()
            with closing(replay_latents(episodes, metadata, checkpoint, config, device, batch_size)) as batches:
                width = _write_latents(staged / "latents.npy", episode_index, batches)
            (staged / "manifest.json").write_text(json.dumps(dict(expected, width=width)) + "\n")
            try:
                staged.rename(cache_dir)
            except OSError as exc:
                # Another process may have published its complete cache first.
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise

    manifest = json.loads((cache_dir / "manifest.json").read_text())
    width = manifest.pop("width", None)
    if manifest != expected or type(width) is not int or width <= 0:
        raise ValueError(f"Dreamer latent cache differs from the requested run or episode order: {cache_dir}")
    # PyTorch can wrap the writable mapping. A caller's edits stay private.
    rows = np.load(cache_dir / "latents.npy", mmap_mode="c", allow_pickle=False)
    if rows.dtype != np.float32 or rows.shape != (sum(length for _, length in episode_index), width):
        raise ValueError(f"Dreamer latent cache shape or dtype disagrees with its manifest: {cache_dir}")
    print(f"Loaded {len(rows)} cached Dreamer latents from {cache_dir}", flush=True)
    return torch.from_numpy(rows), provenance
