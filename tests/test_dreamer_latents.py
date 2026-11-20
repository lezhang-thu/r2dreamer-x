"""Check offline latents against online posterior modes on the collector's inputs."""

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from dreamer_latents import replay_latents
from envs import make_env
from envs.native_archive import NativeEpisodeArchive
from networks import MultiEncoder
from rssm import RSSM


def test_replay_reproduces_online_posterior_across_chunks_and_resets(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "variants-supervised-agent"))
    from atari_data import load_archive

    env_config = OmegaConf.load(root / "configs/env/atari100k.yaml")
    env_config.update(task="atari_montezuma_revenge", seed=7, device="cpu", time_limit=12)
    model_config = OmegaConf.load(root / "configs/model/_base_.yaml")
    model_config.update(deter=16, hidden=8, discrete=4, depth=2, act="SiLU", norm=True)
    config = OmegaConf.create(dict(device="cpu", model=model_config, env=env_config))
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(config, config_path, resolve=False)
    env = make_env(config.env, 0)
    encoder = MultiEncoder(config.model.encoder, {"image": env.observation_space["image"].shape}).eval()
    num_actions = env.action_space.shape[0]
    rssm = RSSM(config.model.rssm, encoder.out_dim, num_actions).eval()
    weights = nn.ModuleDict(dict(encoder=encoder, rssm=rssm)).state_dict()
    checkpoint = tmp_path / "latest.pt"
    torch.save(dict(agent_state_dict=weights), checkpoint)

    path = tmp_path / "episodes.pkl"
    recorder = NativeEpisodeArchive(env, path, OmegaConf.to_container(config.env, resolve=True))
    expected = []

    @torch.no_grad()
    def observe(obs, prev, state):
        embed = encoder({"image": torch.from_numpy(obs["image"])[None].float() / 255})
        stoch, deter, _ = rssm.obs_step(*state, torch.from_numpy(prev)[None], embed,
                                        torch.tensor([bool(obs["is_first"])]), sample=False)
        expected.append(rssm.get_feat(stoch, deter))
        return stoch, deter

    done = True
    try:
        for action in range(1, 6):
            if done:
                obs = recorder.reset()
                state = observe(obs, np.zeros(num_actions, np.float32), rssm.initial(1))
            one_hot = np.eye(num_actions, dtype=np.float32)[action]
            obs, _, done, _ = recorder.step(one_hot)
            state = observe(obs, one_hot, state)
        recorder.finish_archive()
    finally:
        env.close()

    episodes, metadata = load_archive(path)
    rows = list(replay_latents(episodes, metadata, checkpoint, config_path, batch_size=2))
    assert [(eid, start, len(batch)) for eid, start, batch in rows] == [
        (0, 0, 2), (0, 2, 2), (1, 0, 2), (1, 2, 1)]
    torch.testing.assert_close(torch.cat([batch for _, _, batch in rows]), torch.cat(expected))

    episodes[1]["frames"] = episodes[1]["frames"].copy()
    episodes[1]["frames"][2, 0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="screen mismatch at episode 1 frame 2"):
        list(replay_latents(episodes, metadata, checkpoint, config_path))
