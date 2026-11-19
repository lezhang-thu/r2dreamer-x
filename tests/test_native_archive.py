"""Check native screens and action alignment against the live Montezuma emulator."""

from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf

from envs import make_env
from envs.native_archive import NativeEpisodeArchive

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "variants-supervised-agent"))
from train_gru_classifier import load_archive


def test_native_archive_preserves_time_limit_and_unfinished_prefix(tmp_path):
    config = OmegaConf.load(ROOT / "configs/env/atari100k.yaml")
    config.update(task="atari_montezuma_revenge", seed=7, device="cpu", time_limit=12)
    env = make_env(config, 0)
    path = tmp_path / "episodes.pkl"
    recorder = NativeEpisodeArchive(env, path, OmegaConf.to_container(config, resolve=True))
    expected = []
    done = True
    try:
        for action in range(1, 6):
            if done:
                obs = recorder.reset()
                assert obs["image"].shape == (64, 64, 3)
                expected.append([env.unwrapped.ale.getScreenRGB().copy()])
            obs, _, done, _ = recorder.step(np.eye(env.action_space.shape[0])[action])
            assert obs["image"].shape == (64, 64, 3)
            expected[-1].append(env.unwrapped.ale.getScreenRGB().copy())
        summary = recorder.finish_archive()
    finally:
        env.close()
    assert list(tmp_path.iterdir()) == [path]
    assert summary == dict(episodes=2, transitions=5, frames=7, bytes=path.stat().st_size)
    records, meta = load_archive(path)
    assert meta["frame_shape"] == (210, 160, 3)
    assert meta["env_config"]["archive_frame_source"] == "ale_rgb"
    assert [ep["complete"] for ep in records] == [True, False]
    assert [ep["collection_cutoff"] for ep in records] == [False, True]
    assert [ep["actions"].tolist() for ep in records] == [[1, 2, 3], [4, 5]]
    assert [ep["truncated"].tolist() for ep in records] == [[False, False, True], [False, False]]
    for ep, screens in zip(records, expected):
        assert not ep["terminated"].any()
        assert not ep["frames"].flags.writeable
        np.testing.assert_array_equal(ep["frames"], screens)
