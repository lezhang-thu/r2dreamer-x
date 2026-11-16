"""Two return rounds with real archive/bank logic and inexpensive model stand-ins."""

import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from episode_archive import EpisodeRecorder
from return_anchors import AnchorBank
from return_exploration import ReturnExploration, observation_batch


class FakeClassifier:
    def __init__(self, round_number):
        self.round_number = round_number

    def pair_logits(self, left, right):
        equal = (left == right).all(-1)
        return torch.where(equal, 10., -10.)


class FakePolicy(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(num_actions))


class ReturnCycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config = OmegaConf.create(dict(interval=100, episodes=2, initial_policy=None,
                                       classifier=dict(window=20), policy=dict(updates=10000)))
        env_config = OmegaConf.create(dict(env_num=1, gray=False, lives='unused', seed=7,
                                           task='atari_montezuma_revenge', action_repeat=4))
        self.cycle = ReturnExploration(config, env_config, self.root, num_actions=3, device='cpu')
        self.events, self.training_calls, self.classifiers, self.environments = [], [], [], []
        self.replay = SimpleNamespace(add_transition=lambda row: self.events.append(('replay', row)))
        self.agent = torch.nn.Linear(1, 1)

    def frame(self, value):
        return np.full((64, 64, 3), value, dtype=np.uint8)

    def ordinary(self, value, interactions, reset=False, done=False):
        observation = dict(image=self.frame(value), is_first=reset, is_terminal=done)
        transition = observation_batch(observation, float(value), 'cpu')
        action = torch.nn.functional.one_hot(torch.tensor([1]), 3).float()
        self.cycle.record_ordinary(transition, action, torch.tensor([reset]),
                                   torch.tensor([done]), interactions)

    @staticmethod
    def argument(arguments, flag):
        return arguments[arguments.index(flag) + 1]

    @staticmethod
    def load_archive(path):
        with Path(path).open('rb') as stream:
            payload = pickle.load(stream)
        episodes = []
        for record in payload['episodes']:
            record = dict(record)
            with np.load(io.BytesIO(record.pop('payload_npz')), allow_pickle=False) as arrays:
                record.update({key: arrays[key] for key in arrays.files})
            episodes.append(record)
        return episodes, payload['metadata']

    def train(self, module, arguments, logfile):
        self.training_calls.append((module, arguments))
        episodes, _ = self.load_archive(self.argument(arguments, '--archive'))
        output = Path(self.argument(arguments, '--output'))
        logfile.write_text('Fake auxiliary training for orchestration test.\n')
        if module == 'train_gru_classifier.py':
            self.assertIn('--all-data', arguments)
            self.assertNotIn('--resume', arguments)
            self.assertNotIn('--finetune', arguments)
            torch.save(dict(round_number=self.cycle.round, threshold=.5,
                            training_episode_ids=[episode['id'] for episode in episodes]), output)
        else:
            self.assertEqual(self.argument(arguments, '--updates'), 10000)
            self.assertIn('--skip-empty-anchors', arguments)
            policy = FakePolicy(3)
            if '--finetune' in arguments:
                previous = torch.load(self.argument(arguments, '--finetune'), weights_only=True)
                policy.load_state_dict(previous['model_state'])
            with torch.no_grad():
                policy.weight.add_(1.)
            output.mkdir()
            for name in ('best.pt', 'latest.pt'):
                torch.save(dict(model_config=dict(num_actions=3), model_state=policy.state_dict()), output / name)

    def load_classifier(self, path, device):
        checkpoint = torch.load(path, weights_only=True)
        classifier = FakeClassifier(checkpoint['round_number'])
        self.classifiers.append(classifier)
        return classifier, checkpoint

    def demonstrations(self, episodes, ids, classifier, threshold, device, skip_empty):
        self.assertTrue(skip_empty)
        self.assertEqual(ids, [episode['id'] for episode in episodes])
        return [dict(id=episode['id'], references=np.array([[0.]], dtype=np.float32),
                     goals=episode['frames'][1:2], anchor_indices=np.array([1]))
                for episode in episodes]

    def make_env(self, config, index):
        env = SimpleNamespace(closed=False)
        env.close = lambda: setattr(env, 'closed', True)
        self.environments.append(env)
        return env

    def collect(self, agent, policy, classifier, env, goals, target, threshold, segment, device):
        self.assertFalse(agent.training)
        self.assertFalse(policy.training)
        self.assertEqual(classifier.round_number, self.cycle.round)
        self.assertEqual(goals.shape, (1, 84, 84, 3))
        self.assertEqual(self.cycle.next_milestone, self.cycle.round * 100)
        self.assertTrue(self.cycle.due(self.cycle.round * 100))
        self.assertFalse(self.cycle.due(self.cycle.round * 100 - 1))
        self.events.append(('collect', self.cycle.round))
        recorder = EpisodeRecorder(self.frame(50), source='assisted')
        recorder.step(0, self.frame(51), 0.)
        recorder.step(1, self.frame(52), 0., terminated=True)
        result = dict(env_return=0., interactions=2, return_actions=1, dreamer_actions=1,
                      reached=True, takeover=True, reached_at=1,
                      anchor_indices=[1], rollout_anchor_count=1)
        return recorder.finish(), [self.cycle.round], np.array([[0.]], dtype=np.float32), result

    def test_two_cycles_reuse_policy_rotate_sources_and_update_counts_before_sampling(self):
        # Only episode 0 completes before the first milestone. The second is
        # paused mid-episode and must survive both auxiliary model fitting and rollout.
        self.ordinary(0, 0, reset=True)
        self.ordinary(1, 1, done=True)
        self.ordinary(10, 99, reset=True)
        self.ordinary(11, 100)
        paused = self.cycle.recorder
        paused_frames = np.stack(paused.frames)
        self.assertEqual(self.cycle.archive.pending_ids, (0,))

        original_sample, original_update = AnchorBank.sample, AnchorBank.update

        def sample(bank, rng):
            self.events.append(('sample', self.cycle.round, bank.counts.tolist()))
            return original_sample(bank, rng)

        def update(bank, references):
            self.assertEqual(self.events[-1][0], 'replay', 'Replay insertion precedes count updates')
            result = original_update(bank, references)
            self.events.append(('update', self.cycle.round, bank.counts.tolist()))
            return result

        classifier_module = SimpleNamespace(load_classifier=self.load_classifier, load_archive=self.load_archive)
        supervised_module = SimpleNamespace(prepare_demonstrations=self.demonstrations, GoalPolicy=FakePolicy)
        output = io.StringIO()
        with patch.object(self.cycle, '_train', side_effect=self.train), \
                patch('return_exploration.variant_modules', return_value=(classifier_module, supervised_module)), \
                patch('return_exploration.collect_assisted_episode', side_effect=self.collect), \
                patch('envs.make_env', side_effect=self.make_env), \
                patch.object(AnchorBank, 'sample', new=sample), \
                patch.object(AnchorBank, 'update', new=update), redirect_stdout(output):
            self.cycle.run(self.agent, self.replay, ordinary_interactions=100)
            self.assertIs(self.cycle.recorder, paused)
            np.testing.assert_array_equal(np.stack(paused.frames), paused_frames)
            self.assertEqual(paused.actions, [1])
            self.assertFalse(paused.ended)
            self.assertEqual(self.cycle.archive.pending_ids, (1, 2))
            self.assertEqual(self.cycle.next_milestone, 200)
            self.assertEqual(self.cycle.assisted_interactions, 4)
            self.assertTrue(self.agent.training)

            # Completing the paused ordinary trajectory makes it episode 3.
            self.ordinary(12, 110, done=True)
            completed = self.cycle.archive.load(3)
            np.testing.assert_array_equal(completed['frames'][:, 0, 0, 0], [10, 11, 12])
            self.ordinary(20, 111, reset=True)
            paused_second = self.cycle.recorder
            self.assertEqual(self.cycle.archive.pending_ids, (1, 2, 3))
            self.cycle.run(self.agent, self.replay, ordinary_interactions=200)
            self.assertIs(self.cycle.recorder, paused_second)
            self.assertEqual(len(paused_second.frames), 1)
            self.assertEqual(paused_second.actions, [])

        first = self.cycle.root / 'round_000001'
        second = self.cycle.root / 'round_000002'
        first_sources, _ = self.load_archive(first / 'episodes.pkl')
        second_sources, _ = self.load_archive(second / 'episodes.pkl')
        self.assertEqual([episode['id'] for episode in first_sources], [0])
        self.assertEqual([episode['source'] for episode in first_sources], ['ordinary'])
        self.assertEqual([episode['id'] for episode in second_sources], [1, 2, 3])
        self.assertEqual([episode['source'] for episode in second_sources], ['assisted', 'assisted', 'ordinary'])
        self.assertEqual(self.cycle.archive.pending_ids, (4, 5))
        self.assertEqual(self.cycle.next_milestone, 300)
        self.assertEqual(self.cycle.assisted_interactions, 8)
        self.assertFalse(self.cycle.due(200))
        self.assertTrue(self.cycle.due(300))

        for directory, source_count, ordinary, assisted in ((first, 1, 100, 4), (second, 3, 200, 8)):
            report = json.loads((directory / 'report.json').read_text())
            self.assertEqual(report['generated_episodes'], 2)
            self.assertEqual(report['source_episodes'], source_count)
            self.assertEqual(report['ordinary_interactions'], ordinary)
            self.assertEqual(report['assisted_interactions'], assisted)
            self.assertEqual(report['total_interactions'], ordinary + assisted)
            self.assertEqual(report['next_milestone'], ordinary + 100)
        self.assertEqual([event for event in self.events if event[0] == 'sample'],
                         [('sample', 1, [1]), ('sample', 1, [2]),
                          ('sample', 2, [3]), ('sample', 2, [4])])
        self.assertEqual([event for event in self.events if event[0] == 'update'],
                         [('update', 1, [2]), ('update', 1, [3]),
                          ('update', 2, [4]), ('update', 2, [5])])
        self.assertEqual([event[0] for event in self.events],
                         ['sample', 'collect', 'replay', 'update'] * 4)
        self.assertEqual(len(self.training_calls), 4)
        self.assertNotIn('--finetune', self.training_calls[1][1])
        self.assertEqual(self.argument(self.training_calls[3][1], '--finetune'), first / 'agent' / 'latest.pt')
        self.assertNotEqual(self.argument(self.training_calls[0][1], '--output'),
                            self.argument(self.training_calls[2][1], '--output'))
        self.assertIsNot(self.classifiers[0], self.classifiers[1])
        self.assertEqual(self.cycle.policy_checkpoint, second / 'agent' / 'latest.pt')
        self.assertTrue(all(env.closed for env in self.environments))
        self.assertTrue(self.agent.training)
        lines = output.getvalue().splitlines()
        summaries = [line for line in lines if line.startswith('[return round ')]
        self.assertEqual(summaries, [
            f'[return round {round_number}] episode {episode}/2 return=0.00 '
            'steps=2 reached=True takeover=True'
            for round_number in (1, 2) for episode in (1, 2)])
        events = [json.loads(line) for line in lines if line.startswith('{')]
        self.assertEqual(len(events), 4)
        self.assertTrue(all(event['env_return'] == 0. for event in events))
        for directory in (first, second):
            rollouts = json.loads((directory / 'rollouts.json').read_text())
            self.assertEqual([row['env_return'] for row in rollouts], [0., 0.])

    def test_default_configuration_uses_fixed_100_episodes(self):
        config = OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs' / 'configs.yaml')
        self.assertEqual(config.return_exploration.episodes, 100)
        self.assertEqual(config.return_exploration.interval, 100000)


if __name__ == '__main__':
    unittest.main()
