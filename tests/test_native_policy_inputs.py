"""Check offline and online policy inputs at native Atari resolution."""

from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'variants-supervised-agent'))
import train_supervised_agent as supervised


def test_native_rgb_policy_and_early_terminal(monkeypatch):
    # The game ends on repeat 1 of 4, before the legacy wrapper would pool frames.
    screen = np.zeros((210, 160, 3), dtype=np.uint8)
    ale = SimpleNamespace(getScreenRGB=screen.copy)

    def step(action):
        screen[-1, 0] = (3, 2, 1)
        return screen.copy(), 1., True, False, {}

    game = SimpleNamespace(unwrapped=SimpleNamespace(ale=ale, get_action_meanings=lambda: ['NOOP', 'FIRE']),
                           action_space=gym.spaces.Discrete(2), reset=lambda seed: None, step=step)
    monkeypatch.setattr(gym, 'make', lambda *args, **kwargs: game)
    env = supervised.AtariEnv(dict(task='atari_montezuma_revenge', archive_frame_source='ale_rgb',
                                   action_repeat=4, pooling=2, size=[64, 64], noops=0))
    initial = env.reset(17)
    terminal, reward, ended, truncated = env.step(1)
    assert initial.shape == terminal.shape == (210, 160, 3)
    assert not initial.any() and terminal[-1, 0].tolist() == [3, 2, 1]
    assert reward == 1. and ended and not truncated

    demo = dict(frames=np.stack([initial, terminal]), actions=np.array([1]), goals=terminal[None])
    policy = supervised.GoalPolicy(2, hidden_dim=8, attention_heads=2)
    logits, labels = supervised.segment_logits(policy, [demo], [(0, 0, 1)])
    assert labels.tolist() == [[1]]
    assert logits.shape == (1, 1, 2) and torch.isfinite(logits).all()
