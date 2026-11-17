"""In-process return rounds share source arrays and retain live optimizer state."""

from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from omegaconf import OmegaConf
import torch

from episode_archive import EpisodeRecorder
from return_anchors import AnchorBank
from return_exploration import (ReturnExploration, auxiliary_training_context,
                                observation_batch, variant_modules)


class FakeClassifier:
    def __init__(self, round_number):
        self.round_number = round_number

    def pair_logits(self, left, right):
        return torch.where((left == right).all(-1), 10., -10.)


class FakePolicy(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(num_actions))


class ReturnCycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = OmegaConf.create(dict(interval=100, episodes=2, initial_policy=None,
                                            classifier=dict(window=20), policy=dict(updates=10000)))
        self.env_config = OmegaConf.create(dict(env_num=1, gray=False, lives='unused', seed=7,
                                                task='atari_montezuma_revenge', action_repeat=4))
        self.cycle = ReturnExploration(self.config, self.env_config, self.root, num_actions=3, device='cpu')
        self.events, self.training_calls, self.classifiers, self.environments = [], [], [], []
        self.sources, self.source_refs, self.policy_states, self.optimizer_states = {}, {}, [], []
        self.preparation_rounds = []
        self.replay = SimpleNamespace(add_transition=lambda row: self.events.append(('replay', row)))
        self.agent = torch.nn.Linear(1, 1)
        self.classifier_module, self.supervised_module = variant_modules()

    @staticmethod
    def frame(value):
        return np.full((64, 64, 3), value, dtype=np.uint8)

    def ordinary(self, value, interactions, reset=False, done=False):
        observation = dict(image=self.frame(value), is_first=reset, is_terminal=done)
        transition = observation_batch(observation, float(value), 'cpu')
        action = torch.nn.functional.one_hot(torch.tensor([1]), 3).float()
        self.cycle.record_ordinary(transition, action, torch.tensor([reset]),
                                   torch.tensor([done]), interactions)

    @staticmethod
    def file_sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def train_classifier(self, args, episodes, metadata, source_sha256):
        self.training_calls.append(('classifier', args))
        self.assertIsNone(args.archive)
        self.assertTrue(args.all_data)
        self.assertIs(metadata, self.cycle.archive.metadata)
        self.assertEqual(source_sha256, self.cycle.archive.fingerprint())
        self.assertTrue(all(episode is self.cycle.archive.load(episode['id']) for episode in episodes))
        self.assertTrue(all('payload_npz' not in episode for episode in episodes))
        self.current_episodes = episodes
        self.sources[self.cycle.round] = [dict(id=e['id'], source=e['source']) for e in episodes]
        self.source_refs[self.cycle.round] = [weakref.ref(e['frames']) for e in episodes]
        classifier = FakeClassifier(self.cycle.round)
        self.classifiers.append(classifier)
        checkpoint = dict(round_number=self.cycle.round, threshold=.5,
                          archive_sha256=source_sha256,
                          training_episode_ids=[episode['id'] for episode in episodes])
        torch.save(checkpoint, args.output)
        torch.save(checkpoint, args.output.with_name('classifier.visual.pt'))
        return classifier, checkpoint

    def demonstrations(self, episodes, ids, classifier, threshold, device, skip_empty):
        self.preparation_rounds.append(self.cycle.round)
        self.assertIs(episodes, self.current_episodes)
        self.assertTrue(skip_empty)
        self.assertEqual(ids, [episode['id'] for episode in episodes])
        self.current_demos = [dict(id=episode['id'], frames=episode['frames'], actions=episode['actions'],
                                  references=np.array([[0.]], dtype=np.float32),
                                  goals=episode['frames'][1:2].copy(), anchor_indices=np.array([1]))
                             for episode in episodes]
        return self.current_demos

    def train_policy(self, args, classifier, classifier_ckpt, metadata, demos, classifier_sha256, state=None):
        self.training_calls.append(('policy', args))
        self.assertIsNone(args.archive)
        self.assertIsNone(args.classifier)
        self.assertTrue(args.skip_empty_anchors)
        self.assertEqual(args.updates, 10000)
        self.assertIs(demos, self.current_demos)
        self.assertIs(metadata, self.cycle.archive.metadata)
        self.assertEqual(classifier_sha256, self.file_sha256(args.output.parent / 'classifier.pt'))
        for demo, episode in zip(demos, self.current_episodes):
            self.assertIs(demo['frames'], episode['frames'])
            self.assertIs(demo['actions'], episode['actions'])
        del self.current_demos, self.current_episodes
        if state is None:
            policy = FakePolicy(3)
            if args.finetune is not None:
                previous = torch.load(args.finetune, weights_only=True)
                policy.load_state_dict(previous['model_state'])
            state = self.supervised_module.PolicyTrainingState(
                policy, torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=0.),
                {}, 0, Path(), '')
        else:
            self.assertIsNone(args.finetune)
            self.assertIs(state, self.policy_states[-1])
            self.assertIs(state.optimizer, self.optimizer_states[-1])
        self.policy_states.append(state)
        self.optimizer_states.append(state.optimizer)
        previous_updates = state.total_updates
        state.policy.train().requires_grad_(True)
        state.optimizer.zero_grad(set_to_none=True)
        state.policy.weight.grad = torch.zeros_like(state.policy.weight)
        state.optimizer.step()
        with torch.no_grad():
            state.policy.weight.add_(1.)
        state.metadata = dict(archive_sha256=classifier_ckpt['archive_sha256'],
                              previous_training_updates=previous_updates)
        state.update = args.updates
        args.output.mkdir()
        for name in ('best.pt', 'latest.pt'):
            torch.save(dict(model_config=dict(num_actions=3), model_state=state.policy.state_dict(),
                            optimizer_state=state.optimizer.state_dict(),
                            metadata=state.metadata, update=state.update), args.output / name)
        state.checkpoint_path = args.output / 'latest.pt'
        state.checkpoint_sha256 = self.file_sha256(state.checkpoint_path)
        rollout = FakePolicy(3)
        rollout.load_state_dict(state.policy.state_dict())
        rollout.eval().requires_grad_(False)
        return state, rollout

    def make_env(self, config, index):
        env = SimpleNamespace(closed=False)
        env.close = lambda: setattr(env, 'closed', True)
        self.environments.append(env)
        return env

    def collect(self, agent, policy, classifier, env, goals, target, threshold, segment, device):
        self.assertFalse(agent.training)
        self.assertFalse(policy.training)
        self.assertIsNot(policy, self.cycle.policy_state.policy)
        self.assertEqual(classifier.round_number, self.cycle.round)
        self.assertEqual(goals.shape, (1, 84, 84, 3))
        self.assertEqual(self.cycle.next_milestone, self.cycle.round * 100)
        self.assertTrue(self.cycle.due(self.cycle.round * 100))
        self.assertFalse(self.cycle.due(self.cycle.round * 100 - 1))
        # Fitting/prefix preparation shared the source arrays and now released
        # them entirely. Only copied goal images remain during collection.
        self.assertTrue(all(reference() is None for reference in self.source_refs[self.cycle.round]))
        for source in self.sources[self.cycle.round]:
            self.assertNotIn(source['id'], self.cycle.archive.pending_ids)
            with self.assertRaises(KeyError):
                self.cycle.archive.load(source['id'])
        if self.cycle.round > 1:
            self.assertTrue((self.cycle.root / f'round_{self.cycle.round - 1:06d}' / 'agent/latest.pt').is_file())
        self.events.append(('collect', self.cycle.round))
        recorder = EpisodeRecorder(self.frame(50), source='assisted')
        recorder.step(0, self.frame(51), 0.)
        recorder.step(1, self.frame(52), 0., terminated=True)
        result = dict(env_return=0., interactions=2, return_actions=1, dreamer_actions=1,
                      reached=True, takeover=True, reached_at=1,
                      anchor_indices=[1], rollout_anchor_count=1)
        return recorder.finish(), [self.cycle.round], np.array([[0.]], dtype=np.float32), result

    @contextmanager
    def runtime(self, train_policy=None, collect=None):
        classifier_module = SimpleNamespace(parse_args=self.classifier_module.parse_args,
                                             train_from_episodes=self.train_classifier)
        supervised_module = SimpleNamespace(parse_args=self.supervised_module.parse_args,
                                              prepare_demonstrations=self.demonstrations,
                                              train_from_demonstrations=train_policy or self.train_policy)
        with patch('return_exploration.variant_modules', return_value=(classifier_module, supervised_module)), \
                patch('return_exploration.collect_assisted_episode', side_effect=collect or self.collect), \
                patch('envs.make_env', side_effect=self.make_env), \
                patch('return_exploration.torch.cuda.is_available', return_value=False), \
                patch('subprocess.run', side_effect=AssertionError('Auxiliary fitting must stay in process')):
            yield

    def assert_no_trajectory_files(self):
        for pattern in ('*.pkl', '*.npz', 'episode_archive', 'anchors.initial.json', 'counts.json', 'dreamer.pt'):
            self.assertFalse(list(self.root.rglob(pattern)))

    def test_two_cycles_share_sources_preserve_live_policy_and_update_counts_before_sampling(self):
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

        output = io.StringIO()
        with self.runtime(), patch.object(AnchorBank, 'sample', new=sample), \
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
            self.assert_no_trajectory_files()
            first = self.cycle.root / 'round_000001'
            self.assertTrue((first / 'agent/latest.pt').is_file())
            self.assertTrue((self.root / 'latest.pt').is_file())
            first_report = json.loads((first / 'report.json').read_text())
            first_optimizer = self.cycle.policy_state.optimizer

            self.ordinary(12, 110, done=True)
            completed = self.cycle.archive.load(3)
            np.testing.assert_array_equal(completed['frames'][:, 0, 0, 0], [10, 11, 12])
            del completed
            self.ordinary(20, 111, reset=True)
            paused_second = self.cycle.recorder
            self.assertEqual(self.cycle.archive.pending_ids, (1, 2, 3))
            self.cycle.run(self.agent, self.replay, ordinary_interactions=200)
            self.assertIs(self.cycle.recorder, paused_second)
            self.assertEqual(len(paused_second.frames), 1)
            self.assertEqual(paused_second.actions, [])
            self.assertIs(self.cycle.policy_state.optimizer, first_optimizer)

        second = self.cycle.root / 'round_000002'
        self.assertFalse(first.exists())
        self.assertEqual(sorted(path.name for path in self.cycle.root.glob('round_*')), ['round_000002'])
        self.assertEqual(self.sources[1], [dict(id=0, source='ordinary')])
        self.assertEqual(self.sources[2], [dict(id=1, source='assisted'), dict(id=2, source='assisted'),
                                         dict(id=3, source='ordinary')])
        self.assertEqual(self.preparation_rounds, [1, 2])
        self.assertEqual(self.cycle.archive.pending_ids, (4, 5))
        self.assertEqual(self.cycle.next_milestone, 300)
        self.assertEqual(self.cycle.assisted_interactions, 8)
        self.assertFalse(self.cycle.due(200))
        self.assertTrue(self.cycle.due(300))
        for episode_id in range(4):
            with self.assertRaises(KeyError):
                self.cycle.archive.load(episode_id)
        self.assert_no_trajectory_files()
        second_report = json.loads((second / 'report.json').read_text())
        history = [json.loads(line) for line in (self.cycle.root / 'cycles.jsonl').read_text().splitlines()]
        self.assertEqual(history, [first_report, second_report])
        for report, source_count, ordinary, assisted, sources, generated in (
                (first_report, 1, 100, 4, [0], [1, 2]), (second_report, 3, 200, 8, [1, 2, 3], [4, 5])):
            self.assertEqual(report['generated_episodes'], 2)
            self.assertEqual(report['source_episodes'], source_count)
            self.assertEqual(report['source_episode_ids'], sources)
            self.assertEqual(report['generated_episode_ids'], generated)
            self.assertEqual(report['ordinary_interactions'], ordinary)
            self.assertEqual(report['assisted_interactions'], assisted)
            self.assertEqual(report['total_interactions'], ordinary + assisted)
            self.assertEqual(report['next_milestone'], ordinary + 100)
            self.assertEqual(report['training_execution'], 'in_process')
        self.assertEqual([event for event in self.events if event[0] == 'sample'],
                         [('sample', 1, [1]), ('sample', 1, [2]), ('sample', 2, [3]), ('sample', 2, [4])])
        self.assertEqual([event for event in self.events if event[0] == 'update'],
                         [('update', 1, [2]), ('update', 1, [3]), ('update', 2, [4]), ('update', 2, [5])])
        self.assertEqual([event[0] for event in self.events], ['sample', 'collect', 'replay', 'update'] * 4)
        self.assertEqual([call[0] for call in self.training_calls], ['classifier', 'policy'] * 2)
        self.assertIsNone(self.training_calls[1][1].finetune)
        self.assertIsNone(self.training_calls[3][1].finetune)
        self.assertIsNot(self.classifiers[0], self.classifiers[1])
        self.assertIs(self.policy_states[0], self.policy_states[1])
        self.assertEqual(self.cycle.policy_checkpoint, second / 'agent/latest.pt')
        self.assertTrue(all(env.closed for env in self.environments))
        self.assertTrue(self.agent.training)
        final_policy = torch.load(self.cycle.policy_checkpoint, weights_only=True)
        torch.testing.assert_close(final_policy['model_state']['weight'], torch.full((3,), 2.))
        self.assertTrue(all(int(state['step']) == 2 for state in first_optimizer.state.values()))
        self.assertEqual(self.cycle.policy_state.total_updates, 20000)
        self.assertEqual(self.cycle.policy_state.policy.weight.device.type, 'cpu')
        self.assertIsNone(self.cycle.policy_state.policy.weight.grad)
        latest = torch.load(self.root / 'latest.pt', weights_only=True)
        self.assertEqual(latest['ordinary_interactions'], 200)
        self.assertEqual(latest['assisted_interactions'], 8)
        lines = output.getvalue().splitlines()
        self.assertEqual([line for line in lines if line.startswith('[return round ')], [
            f'[return round {round_number}] episode {episode}/2 return=0.00 steps=2 reached=True takeover=True'
            for round_number in (1, 2) for episode in (1, 2)])
        events = [json.loads(line) for line in lines if line.startswith('{')]
        self.assertEqual(len(events), 4)
        self.assertTrue(all(event['env_return'] == 0. for event in events))
        self.assertTrue(all('anchor_indices' not in event for event in events))
        rollouts = json.loads((second / 'rollouts.json').read_text())
        self.assertEqual([row['env_return'] for row in rollouts], [0., 0.])
        self.assertTrue(all('anchor_indices' not in row for row in rollouts))

    def test_failed_next_fit_preserves_previous_models_and_sources(self):
        self.ordinary(0, 0, reset=True)
        self.ordinary(1, 1, done=True)
        with self.runtime(), redirect_stdout(io.StringIO()):
            self.cycle.run(self.agent, self.replay, ordinary_interactions=100)
        first = self.cycle.root / 'round_000001'
        old_checkpoint = first / 'agent/latest.pt'
        previous_bytes = old_checkpoint.read_bytes()
        previous_dreamer = (self.root / 'latest.pt').read_bytes()
        pending = self.cycle.archive.pending_ids
        policy_state = self.cycle.policy_state

        def failing_policy(*args, **kwargs):
            del self.current_demos, self.current_episodes
            raise RuntimeError('Injected auxiliary fit failure')

        with self.runtime(train_policy=failing_policy), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'return policy failed') as raised:
                self.cycle.run(self.agent, self.replay, ordinary_interactions=200)
        self.assertIn('Injected auxiliary fit failure', str(raised.exception.__cause__))
        self.assertTrue(first.is_dir())
        self.assertEqual(old_checkpoint.read_bytes(), previous_bytes)
        self.assertEqual((self.root / 'latest.pt').read_bytes(), previous_dreamer)
        self.assertEqual(self.cycle.policy_checkpoint, old_checkpoint)
        self.assertIs(self.cycle.policy_state, policy_state)
        self.assertEqual(self.cycle.archive.pending_ids, pending)
        self.assertTrue(all(self.cycle.archive.load(i)['complete'] for i in pending))
        self.assertEqual(len((self.cycle.root / 'cycles.jsonl').read_text().splitlines()), 1)
        self.assert_no_trajectory_files()
        self.assertIn('Injected auxiliary fit failure', (self.cycle.root / 'round_000002/policy.log').read_text())

    def test_external_initial_policy_is_loaded_once_and_never_retired(self):
        external = self.root / 'external' / 'agent' / 'latest.pt'
        external.parent.mkdir(parents=True)
        model = FakePolicy(3)
        with torch.no_grad():
            model.weight.fill_(7.)
        torch.save(dict(model_config=dict(num_actions=3), model_state=model.state_dict()), external)
        external_bytes = external.read_bytes()
        self.config.initial_policy = str(external)
        self.cycle = ReturnExploration(self.config, self.env_config, self.root / 'fresh', num_actions=3, device='cpu')
        self.ordinary(0, 0, reset=True)
        self.ordinary(1, 1, done=True)
        with self.runtime(), redirect_stdout(io.StringIO()):
            self.cycle.run(self.agent, self.replay, ordinary_interactions=100)
            self.cycle.run(self.agent, self.replay, ordinary_interactions=200)
        self.assertEqual(self.training_calls[1][1].finetune, external)
        self.assertIsNone(self.training_calls[3][1].finetune)
        self.assertIs(self.policy_states[0], self.policy_states[1])
        self.assertEqual(external.read_bytes(), external_bytes)
        self.assertTrue(external.parent.is_dir())
        self.assertFalse((self.cycle.root / 'round_000001').exists())
        final_policy = torch.load(self.cycle.policy_checkpoint, weights_only=True)
        torch.testing.assert_close(final_policy['model_state']['weight'], torch.full((3,), 9.))

    def test_default_configuration_uses_fixed_100_episodes(self):
        config = OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs/configs.yaml')
        self.assertEqual(config.return_exploration.episodes, 100)
        self.assertEqual(config.return_exploration.interval, 100000)

    def test_auxiliary_context_restores_rng_and_threads_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(failure=fail):
                before_python, before_numpy = random.getstate(), np.random.get_state()
                before_torch, before_threads = torch.get_rng_state(), torch.get_num_threads()
                try:
                    with auxiliary_training_context('cpu'):
                        random.seed(123)
                        np.random.seed(123)
                        torch.manual_seed(123)
                        random.random(), np.random.random(), torch.rand(2)
                        torch.set_num_threads(1 if before_threads != 1 else 2)
                        if fail:
                            raise RuntimeError('injected')
                except RuntimeError as error:
                    self.assertTrue(fail)
                    self.assertEqual(str(error), 'injected')
                self.assertEqual(random.getstate(), before_python)
                after_numpy = np.random.get_state()
                self.assertEqual(after_numpy[0], before_numpy[0])
                np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
                self.assertEqual(after_numpy[2:], before_numpy[2:])
                self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))
                self.assertEqual(torch.get_num_threads(), before_threads)


if __name__ == '__main__':
    unittest.main()
