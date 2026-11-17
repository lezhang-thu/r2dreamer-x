"""Run real auxiliary optimizers and assisted collection in one CPU process."""

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from buffer import Buffer
from episode_archive import EpisodeRecorder
from return_exploration import ReturnExploration
from test_return_rollout import FakeDreamer, FakeEnvironment


class CheckpointableDreamer(torch.nn.Module, FakeDreamer):
    def __init__(self):
        torch.nn.Module.__init__(self)
        FakeDreamer.__init__(self)
        self.dummy = torch.nn.Parameter(torch.zeros(1))


class Environment(FakeEnvironment):
    def close(self):
        pass


def test_two_real_training_cycles_reuse_live_optimizer_without_data_exchange(tmp_path, monkeypatch):
    config = OmegaConf.create(dict(
        interval=3, episodes=2, initial_policy=None,
        classifier=dict(architecture='small', embedding_dim=8, hidden_dim=8,
                        visual_steps=1, gru_steps=1, batch_size=2, eval_pairs=2,
                        eval_every=1, seed=3, threads=1),
        policy=dict(updates=1, batch_size=1, unroll_length=2, score_every=1,
                    hidden_dim=8, attention_heads=2, seed=13, threads=1)))
    env_config = OmegaConf.create(dict(env_num=1, gray=False, lives='unused', seed=7,
                                       task='atari_test', action_repeat=4))
    replay = Buffer(OmegaConf.create(dict(device='cpu', storage_device='cpu', batch_size=1,
                                         batch_length=1, max_size=128, online=True,
                                         update_latents=False)))
    cycle = ReturnExploration(config, env_config, tmp_path, 3, 'cpu')
    recorder = EpisodeRecorder(np.zeros((64, 64, 3), np.uint8), source='ordinary')
    for index in range(1, 4):
        recorder.step(index % 3, np.full((64, 64, 3), index, np.uint8),
                      float(index), terminated=index == 3)
    cycle.archive.append(recorder.finish())
    agent = CheckpointableDreamer()
    monkeypatch.setattr('envs.make_env', lambda *args: Environment())

    def forbidden(*args, **kwargs):
        raise AssertionError('In-process training must not launch subprocesses or reload checkpoints')

    monkeypatch.setattr('subprocess.run', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    assert cycle.run(agent, replay, 3)
    policy, optimizer = cycle.policy_state.policy, cycle.policy_state.optimizer
    assert cycle.run(agent, replay, 6)
    assert cycle.policy_state.policy is policy
    assert cycle.policy_state.optimizer is optimizer
    assert all(int(state['step']) == 2 for state in optimizer.state.values())
    assert cycle.policy_state.metadata['optimizer_continuation'] == 'memory'
    assert next(policy.parameters()).device.type == 'cpu'
    assert cycle.archive.pending_ids == (3, 4)
    assert cycle.assisted_interactions == 12
    assert replay.count() == 16  # Four three-action episodes plus reset observations.
    assert not list(Path(tmp_path).rglob('*.pkl'))
    assert len(list(Path(tmp_path).rglob('*.pt'))) == 4
    assert agent.training
