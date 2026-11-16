"""Behavioral tests for the actual return-to-Dreamer handoff loop."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict

from return_exploration import collect_assisted_episode


class FakeEnvironment:
    def __init__(self, pixels=(0, 1, 2, 3), terminal=True):
        self.pixels = pixels
        self.terminal = terminal
        self.index = None
        self.reset_calls = 0
        self.actions = []

    def observation(self):
        done = self.index == len(self.pixels) - 1
        return dict(image=np.full((64, 64, 3), self.pixels[self.index], dtype=np.uint8),
                    is_first=np.bool_(self.index == 0), is_last=np.bool_(done),
                    is_terminal=np.bool_(done and self.terminal))

    def reset(self):
        self.reset_calls += 1
        self.index = 0
        return self.observation()

    def step(self, action):
        if self.index >= len(self.pixels) - 1:
            raise AssertionError("Environment stepped after episode end")
        index = int(np.argmax(action))
        np.testing.assert_array_equal(action, np.eye(3)[index])
        self.actions.append(index)
        self.index += 1
        return self.observation(), float(self.index), self.index == len(self.pixels) - 1, {}


class FakeDreamer:
    def __init__(self):
        self.history = []

    def get_initial_state(self, batch):
        return TensorDict(dict(prev_action=torch.zeros(batch, 3),
                               stoch=torch.zeros(batch, 1),
                               deter=torch.zeros(batch, 1)), batch_size=(batch,))

    def act(self, observation, state, eval=False):
        self.history.append(dict(pixel=int(observation['image'][0, 0, 0, 0]),
                                 previous_action=state['prev_action'].clone(),
                                 is_first=bool(observation['is_first'][0])))
        action = torch.tensor([[0., 0., 1.]])
        # The latent value identifies which actions actually conditioned the RNN.
        action_value = (state['prev_action'] * torch.tensor([1., 2., 3.])).sum(-1, keepdim=True)
        latent = state['deter'] + action_value
        result = TensorDict(dict(prev_action=action.clone(), deter=latent,
                                 stoch=observation['image'][:, 0, 0, :1].float()), batch_size=(1,))
        return action, result


class FakeReturnPolicy:
    num_actions = 3

    def __init__(self):
        self.goals = None
        self.history = []

    def encode_goals(self, goals):
        self.goals = goals
        return torch.zeros(1, len(goals[0]), 1), torch.tensor([len(goals[0])])

    def step(self, frame, memory, lengths, hidden):
        self.history.append((int(frame[0, 0, 0]), hidden))
        return torch.tensor([0., 10., 0.]), (0 if hidden is None else hidden) + 1


class FakeClassifier:
    def __init__(self):
        self.history = []

    def encode_step(self, frame, hidden):
        pixel = frame[0, 0, 0, 0].float()
        self.history.append((float(pixel), None if hidden is None else float(hidden)))
        context = pixel if hidden is None else pixel + hidden
        return context.reshape(1, 1), context

    def pair_logits(self, left, right):
        # Matching needs the complete recurrent context, not just the current RGB.
        return torch.where((left - right).abs().sum(-1) < .001, 20., -20.)


class ReturnRolloutTest(unittest.TestCase):
    def collect(self, target, pixels=(0, 1, 2, 3), terminal=True):
        env = FakeEnvironment(pixels, terminal)
        agent, policy, classifier = FakeDreamer(), FakeReturnPolicy(), FakeClassifier()
        goals = np.zeros((2, 84, 84, 3), dtype=np.uint8)
        module = SimpleNamespace(frames_to_tensor=lambda frames, device: torch.as_tensor(frames, device=device))
        with patch('return_exploration.variant_modules', return_value=(module, None)):
            episode, rows, references, result = collect_assisted_episode(
                agent, policy, classifier, env, goals, np.array([target], dtype=np.float32),
                .5, -7, 'cpu')
        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(len(rows), len(pixels))
        self.assertEqual(result['interactions'], len(pixels) - 1)
        self.assertEqual(episode['source'], 'assisted')
        self.assertTrue(episode['complete'])
        np.testing.assert_array_equal(episode['actions'], env.actions)
        np.testing.assert_array_equal(episode['frames'][:, 0, 0, 0], pixels)
        np.testing.assert_array_equal(episode['rewards'], np.arange(1, len(pixels)))
        for index, row in enumerate(rows):
            self.assertEqual(int(row['episode'][0]), -7)
            self.assertEqual(int(row['image'][0, 0, 0, 0]), pixels[index])
            self.assertEqual(float(row['reward'][0]), float(index))
            self.assertEqual(bool(row['is_first'][0]), index == 0)
            self.assertEqual(bool(row['is_last'][0]), index == len(pixels) - 1)
            self.assertEqual(bool(row['is_terminal'][0]), index == len(pixels) - 1 and terminal)
            if index < len(env.actions):
                np.testing.assert_array_equal(row['action'][0].numpy(), np.eye(3)[env.actions[index]])
            else:
                self.assertEqual(float(row['action'].sum()), 0.)
        # Every observation is processed by Dreamer, with executed prior actions.
        self.assertEqual([row['pixel'] for row in agent.history], list(pixels))
        np.testing.assert_array_equal(agent.history[0]['previous_action'], np.zeros((1, 3)))
        for index, previous in enumerate(agent.history[1:]):
            np.testing.assert_array_equal(previous['previous_action'][0], np.eye(3)[env.actions[index]])
        # The classifier starts from reset and includes the final image.
        contexts = np.cumsum(pixels)
        self.assertEqual(classifier.history[0], (float(pixels[0]), None))
        self.assertEqual(classifier.history[1:],
                         [(float(pixels[i]), float(contexts[i - 1])) for i in range(1, len(pixels))])
        return env, agent, policy, episode, rows, references, result

    def test_middle_hit_hands_over_same_env_with_actual_action_history(self):
        env, agent, policy, episode, rows, references, result = self.collect(3)
        self.assertEqual(env.actions, [1, 1, 2])
        self.assertEqual(result['reached_at'], 2)
        self.assertTrue(result['takeover'])
        self.assertEqual(result['return_actions'], 2)
        self.assertEqual(result['dreamer_actions'], 1)
        self.assertEqual(policy.history, [(0, None), (1, 1)])
        self.assertEqual([float(row['deter'][0]) for row in rows], [0., 2., 4., 7.])
        self.assertEqual(result['anchor_indices'], [1, 2, 3])
        np.testing.assert_array_equal(references[:, 0], [1, 3, 6])
        np.testing.assert_array_equal(episode['terminated'], [False, False, True])

    def test_no_hit_runs_return_policy_through_complete_episode(self):
        env, agent, policy, episode, rows, references, result = self.collect(100)
        self.assertEqual(env.actions, [1, 1, 1])
        self.assertIsNone(result['reached_at'])
        self.assertFalse(result['reached'])
        self.assertFalse(result['takeover'])
        self.assertEqual(result['return_actions'], 3)
        self.assertEqual(result['dreamer_actions'], 0)
        self.assertEqual(policy.history, [(0, None), (1, 1), (2, 2)])

    def test_terminal_hit_has_no_dreamer_takeover(self):
        env, agent, policy, episode, rows, references, result = self.collect(6)
        self.assertEqual(env.actions, [1, 1, 1])
        self.assertEqual(result['reached_at'], 3)
        self.assertTrue(result['reached'])
        self.assertFalse(result['takeover'])
        self.assertEqual(result['dreamer_actions'], 0)
        self.assertEqual(result['anchor_indices'][-1], 3)

    def test_initial_hit_immediately_uses_dreamer_without_reset(self):
        env, agent, policy, episode, rows, references, result = self.collect(0)
        self.assertEqual(env.actions, [2, 2, 2])
        self.assertEqual(result['reached_at'], 0)
        self.assertTrue(result['takeover'])
        self.assertEqual(result['return_actions'], 0)
        self.assertEqual(result['dreamer_actions'], 3)
        self.assertEqual(policy.history, [])

    def test_truncation_and_empty_rollout_anchors(self):
        env, agent, policy, episode, rows, references, result = self.collect(100, pixels=(0, 0, 0), terminal=False)
        self.assertEqual(result['anchor_indices'], [])
        self.assertEqual(references.shape, (0, 1))
        self.assertEqual(result['rollout_anchor_count'], 0)
        np.testing.assert_array_equal(episode['terminated'], [False, False])
        np.testing.assert_array_equal(episode['truncated'], [False, True])


if __name__ == '__main__':
    unittest.main()
