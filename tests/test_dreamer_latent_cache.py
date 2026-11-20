from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "variants-supervised-agent"))
import dreamer_latent_cache as cache


def test_cached_replay_reuses_ordered_rows_and_checks_identity(tmp_path, monkeypatch):
    episodes = [dict(id=10, frames=range(3)), dict(id=4, frames=range(2))]
    checkpoint, config = tmp_path / "checkpoint", tmp_path / "config"
    checkpoint.write_bytes(b"model")
    config.write_bytes(b"config")
    path = tmp_path / "latents"
    all_rows = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    calls = []

    def replay(*args):
        calls.append(1)
        yield 10, 0, all_rows[:2]
        yield 10, 2, all_rows[2:3]
        yield 4, 0, all_rows[3:]

    monkeypatch.setattr(cache, "replay_latents", replay)

    def load(records=episodes, archive_hash="one"):
        return cache.load_or_replay_latents(path, records, {}, archive_hash, checkpoint, config)

    first, provenance = load()
    torch.testing.assert_close(first, all_rows)
    first[0, 0] = 42
    cached, cached_provenance = load()
    torch.testing.assert_close(cached, all_rows)
    assert provenance == cached_provenance

    with pytest.raises(ValueError, match="cache differs"):
        load(episodes[::-1])
    with pytest.raises(ValueError, match="cache differs"):
        load(archive_hash="two")
    checkpoint.write_bytes(b"retrained-model")
    with pytest.raises(ValueError, match="cache differs"):
        load()
    checkpoint.write_bytes(b"model")
    config.write_bytes(b"new-config")
    with pytest.raises(ValueError, match="cache differs"):
        load()
    assert len(calls) == 1


def test_partial_or_misaligned_replay_is_never_published(tmp_path, monkeypatch):
    episodes = [dict(id=4, frames=range(3))]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"model")
    path = tmp_path / "latents"
    rows = torch.ones((1, 3))

    def partial(*args):
        yield 4, 0, rows

    def misaligned(*args):
        yield 4, 1, rows

    for replay, message in ((partial, "ended before"), (misaligned, "does not align")):
        monkeypatch.setattr(cache, "replay_latents", replay)
        with pytest.raises(ValueError, match=message):
            cache.load_or_replay_latents(path, episodes, {}, "one", checkpoint, checkpoint)
        assert list(tmp_path.iterdir()) == [checkpoint]
