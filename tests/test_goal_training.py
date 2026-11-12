"""Behavioral tests for online milestones and recurrent on-policy goal learning."""

from copy import deepcopy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from goal_policy import GoalPolicy
from goal_a2c import GoalA2C, compute_returns
from online_milestones import CompletedTrajectoryStore, OnlineClassifier, Trajectory
from goal_training import GoalTraining, IsolatedRandomState, select_recent_evaluation

torch.set_num_threads(2)


def rgb(value, size=84):
    return np.full((size, size, 3), value, dtype=np.uint8)


class CounterEnv:
    """An action-independent tiny episode with separately tracked transitions."""

    def __init__(self, length=3, terminal=True, reward=2.0):
        self.length = length
        self.terminal = terminal
        self.reward = reward
        self.action_space = SimpleNamespace(n=3)
        self.steps = 0
        self.resets = 0
        self.closed = False
        self.elapsed = 0

    def _observation(self, first=False, last=False):
        return {"image": rgb(self.elapsed), "is_first": first,
                "is_last": last, "is_terminal": last and self.terminal}

    def reset(self, seed=None):
        self.resets += 1
        self.elapsed = 0
        return self._observation(first=True)

    def step(self, action):
        self.steps += 1
        self.elapsed += 1
        done = self.elapsed >= self.length
        return self._observation(last=done), self.reward, done, {}

    def close(self):
        self.closed = True


class ExactImageClassifier(torch.nn.Module):
    def encode(self, frames):
        return frames[:, 0, 0, 0:1]

    def pair_logits(self, left, right):
        return torch.where((left - right).abs().squeeze(-1) < 0.5 / 255, 10.0, -10.0)


class RecordingPolicy(GoalPolicy):
    def __init__(self):
        super().__init__(3, hidden_dim=16, attention_heads=4)
        self.training_forwards = []

    def forward(self, observations, memory, lengths, progress, hidden):
        if torch.is_grad_enabled():
            self.training_forwards.append((hidden.requires_grad, self.actor.weight._version))
        return super().forward(observations, memory, lengths, progress, hidden)


class A2CTests(unittest.TestCase):
    def test_rollout_bootstraps_real_time_limits_and_never_terminal_states(self):
        cases = [(2, True, [0., 0.]), (2, False, [0.5, 1.]), (5, True, [0.5, 1.])]
        for length, terminal, expected in cases:
            with self.subTest(length=length, terminal=terminal):
                policy = GoalPolicy(3, hidden_dim=16)
                with torch.no_grad():
                    policy.critic.weight.zero_()
                    policy.critic.bias.fill_(2.)
                agent = GoalA2C(policy, ExactImageClassifier(), gamma=0.5)
                recorded = []

                def record_returns(*args, **kwargs):
                    result = compute_returns(*args, **kwargs)
                    recorded.append(result.clone())
                    return result

                with patch("goal_a2c.compute_returns", side_effect=record_returns):
                    agent.run_batch([np.stack([rgb(99)])],
                                    lambda index: CounterEnv(length=length, terminal=terminal),
                                    [0], train=True, max_steps=2, rollout_steps=4)
                self.assertEqual(len(recorded), 1)
                torch.testing.assert_close(recorded[0][:, 0], torch.tensor(expected))

    def test_rewards_are_sequential_and_at_most_one_goal_per_environment_step(self):
        agent = GoalA2C(GoalPolicy(3, hidden_dim=16), ExactImageClassifier())
        environments = []

        def factory(index):
            environment = CounterEnv()
            environments.append(environment)
            return environment

        sequences = [np.stack([rgb(2), rgb(1)]), np.stack([rgb(1), rgb(1)]),
                     np.empty((0, 84, 84, 3), np.uint8)]
        result = agent.run_batch(sequences, factory, [0, 1, 2], train=False,
                                 max_steps=3, rollout_steps=2)
        self.assertEqual(len(environments), 2)
        first, second, empty = result["episodes"]
        self.assertEqual(first["hit_steps"], [2])
        self.assertEqual(second["hit_steps"], [1])
        for record in (first, second):
            self.assertEqual(record["goals_hit"], 1)
            self.assertEqual(record["goal_return"], 1)
            self.assertEqual(record["env_return"], 6)
            self.assertFalse(record["success"])
        self.assertTrue(empty["skipped_empty"])
        self.assertFalse(empty["success"])
        self.assertEqual(empty["env_steps"], 0)
        self.assertEqual(agent.updates, 0)

    def test_policy_gradients_are_on_policy_and_recurrence_is_truncated_between_chunks(self):
        torch.manual_seed(2)
        policy = RecordingPolicy()
        agent = GoalA2C(policy, ExactImageClassifier())
        before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
        result = agent.run_batch([np.stack([rgb(99)])], lambda index: CounterEnv(length=5),
                                 [0], train=True, max_steps=5, rollout_steps=2)
        self.assertEqual(result["summary"]["env_steps"], 5)
        self.assertEqual(agent.updates, 3)
        self.assertEqual([record["transitions"] for record in result["updates"]], [2, 2, 1])
        self.assertEqual([entry[0] for entry in policy.training_forwards], [False, True, False, True, False])
        versions = [entry[1] for entry in policy.training_forwards]
        self.assertEqual(versions[0], versions[1])
        self.assertEqual(versions[2], versions[3])
        self.assertLess(versions[1], versions[2])
        self.assertLess(versions[3], versions[4])
        for prefix in ("actor.", "critic.", "anchor_gru.", "controller."):
            self.assertTrue(any(not torch.equal(parameter, before[name])
                                for name, parameter in policy.named_parameters() if name.startswith(prefix)), prefix)
        self.assertTrue(all(np.isfinite(record["gradient_norm"]) and record["gradient_norm"] > 0
                            for record in result["updates"]))

    def test_environment_reward_is_logged_but_does_not_enter_policy_loss(self):
        torch.manual_seed(8)
        first_policy = GoalPolicy(3, hidden_dim=16)
        second_policy = deepcopy(first_policy)
        agents = [GoalA2C(first_policy, ExactImageClassifier()),
                  GoalA2C(second_policy, ExactImageClassifier())]
        results = []
        for agent, reward in zip(agents, [0.0, 1000.0]):
            torch.manual_seed(4)
            results.append(agent.run_batch([np.stack([rgb(99)])],
                                            lambda index: CounterEnv(length=2, reward=reward),
                                            [0], train=True, max_steps=2, rollout_steps=2))
        self.assertEqual(results[0]["episodes"][0]["env_return"], 0)
        self.assertEqual(results[1]["episodes"][0]["env_return"], 2000)
        for first, second in zip(first_policy.parameters(), second_policy.parameters()):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_checkpoint_restores_trained_policy_and_counters(self):
        agent = GoalA2C(GoalPolicy(3, hidden_dim=16), ExactImageClassifier())
        agent.run_batch([np.stack([rgb(99)])], lambda index: CounterEnv(length=2),
                        [0], train=True, max_steps=2, rollout_steps=2)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "goal.pt"
            agent.save(checkpoint, {"classifier_version": 4})
            restored = GoalA2C.load(checkpoint, ExactImageClassifier())
            self.assertEqual(restored.updates, agent.updates)
            self.assertEqual(restored.training_env_steps, agent.training_env_steps)
            self.assertEqual(restored.checkpoint_metadata["classifier_version"], 4)
            for first, second in zip(agent.policy.parameters(), restored.policy.parameters()):
                torch.testing.assert_close(first, second, rtol=0, atol=0)


class ReturnTests(unittest.TestCase):
    def test_terminal_truncation_and_continuation_bootstrap_differ(self):
        rewards = torch.tensor([[1., 1., 1.], [2., 2., 2.], [3., 0., 0.]])
        dones = torch.tensor([[False, False, False], [False, True, True], [False, False, False]])
        terminals = torch.zeros_like(dones)
        terminals[1, 1] = True
        valid = torch.tensor([[True, True, True], [True, True, True], [True, False, False]])
        bootstrap = torch.full_like(rewards, 6)
        returns = compute_returns(rewards, dones, terminals, valid, bootstrap,
                                  torch.tensor([10., 99., 99.]), gamma=0.5)
        torch.testing.assert_close(returns[:, 0], torch.tensor([4., 6., 8.]))
        torch.testing.assert_close(returns[:2, 1], torch.tensor([2., 2.]))
        torch.testing.assert_close(returns[:2, 2], torch.tensor([3.5, 5.]))
        self.assertTrue(torch.isfinite(returns).all())


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.policy = GoalPolicy(3, hidden_dim=16, attention_heads=4).eval()
        self.observations = torch.rand(2, 3, 84, 84)
        self.hidden = torch.zeros(2, 16)

    def test_padding_and_completed_anchors_do_not_influence_outputs(self):
        memory = torch.randn(2, 5, 16)
        lengths, progress = torch.tensor([3, 4]), torch.tensor([1, 2])
        changed = memory.clone()
        changed[0, 0] = 10000
        changed[0, 3:] = -10000
        changed[1, :2] = 10000
        changed[1, 4:] = -10000
        with torch.no_grad():
            first = self.policy(self.observations, memory, lengths, progress, self.hidden)
            second = self.policy(self.observations, changed, lengths, progress, self.hidden)
        for key in ("logits", "value", "hidden"):
            torch.testing.assert_close(first[key], second[key], atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.all(first["attention"][0, [0, 3, 4]] == 0))
        self.assertTrue(torch.all(first["attention"][1, [0, 1, 4]] == 0))

    def test_empty_and_all_completed_sequences_are_finite(self):
        with torch.no_grad():
            empty = torch.empty(2, 0, 3, 84, 84)
            memory = self.policy.encode_anchors(empty, torch.tensor([0, 0]))
            for lengths, progress in ((torch.tensor([0, 0]), torch.tensor([0, 0])),
                                      (torch.tensor([1, 1]), torch.tensor([1, 1]))):
                output = self.policy(self.observations, memory, lengths, progress, self.hidden)
                for key in ("logits", "value", "hidden", "attention"):
                    self.assertTrue(torch.isfinite(output[key]).all(), key)

    def test_padded_goal_images_and_batch_size_do_not_change_valid_memory(self):
        anchors = torch.rand(2, 4, 3, 84, 84)
        lengths = torch.tensor([1, 3])
        changed = anchors.clone()
        changed[0, 1:] = 1000
        changed[1, 3:] = -1000
        with torch.no_grad():
            first = self.policy.encode_anchors(anchors, lengths)
            second = self.policy.encode_anchors(changed, lengths)
            single = self.policy.encode_anchors(anchors[:1, :1], lengths[:1])
        torch.testing.assert_close(first[0, :1], second[0, :1])
        torch.testing.assert_close(first[1, :3], second[1, :3])
        torch.testing.assert_close(first[0, :1], single[0, :1], atol=1e-6, rtol=1e-6)

    def test_long_uint8_sequences_match_float_and_backpropagate_through_visual_chunks(self):
        anchors = torch.randint(0, 256, (2, 144, 84, 84, 3), dtype=torch.uint8)
        lengths = torch.tensor([1, 144])  # 145 valid images crosses the visual chunk boundary.
        floats = anchors.float().permute(0, 1, 4, 2, 3).contiguous() / 255
        with torch.no_grad():
            expected = self.policy.encode_anchors(floats, lengths)
        visual_batch_sizes = []
        hook = self.policy.vision.register_forward_pre_hook(
            lambda module, inputs: visual_batch_sizes.append(len(inputs[0])))
        try:
            actual = self.policy.encode_anchors(anchors, lengths)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            self.assertEqual(visual_batch_sizes, [128, 17])
            self.assertTrue(torch.all(actual[0, 1:] == 0))
            loss = actual[0, 0].square().sum() + actual[1, -1].square().sum()
            loss.backward()
        finally:
            hook.remove()
        self.assertGreater(len(visual_batch_sizes), 2)  # Backward recomputes checkpointed CNN chunks.
        self.assertLessEqual(max(visual_batch_sizes), 128)
        for module in (self.policy.vision, self.policy.anchor_gru):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)


class OnlineMilestoneTests(unittest.TestCase):
    def test_only_complete_trajectories_are_retained_and_workers_do_not_mix(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = CompletedTrajectoryStore(Path(temporary), max_recent=2)
            self.assertIsNone(store.add_observation(0, rgb(0), True, False, 0))
            self.assertIsNone(store.add_observation(1, rgb(10), True, False, 0))
            self.assertIsNone(store.add_observation(0, rgb(1), False, False, 4, reward=2))
            self.assertEqual(len(store.episodes), 0)
            first = store.add_observation(1, rgb(11), False, True, 8, reward=5)
            self.assertTrue(first.complete)
            np.testing.assert_array_equal(first.frames[:, 0, 0, 0], [10, 11])
            self.assertEqual(first.reward, 5)
            second = store.add_observation(0, rgb(2), False, True, 12, reward=3)
            np.testing.assert_array_equal(second.frames[:, 0, 0, 0], [0, 1, 2])
            self.assertEqual(second.reward, 5)
            self.assertEqual(second.end_step, 12)
            store.add_observation(0, rgb(20), True, False, 16)
            store.discard_partial()
            self.assertEqual([episode.id for episode in store.latest(1)], [second.id])
            self.assertEqual(len(store.episodes), 2)

    def test_classifier_snapshot_parameters_are_immutable_after_live_update(self):
        live = OnlineClassifier(device="cpu", embedding_dim=16, window=1, batch_size=4, seed=0)
        snapshot, version = live.snapshot()
        before = {key: value.clone() for key, value in snapshot.state_dict().items()}
        self.assertFalse(snapshot.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in snapshot.parameters()))
        episodes = [Trajectory(i, i + 1, np.stack([rgb(j + i) for j in range(5)]), 0)
                    for i in range(2)]
        live.update(episodes, steps=1)
        self.assertGreater(live.version, version)
        for key, value in snapshot.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        self.assertTrue(any(not torch.equal(live.model.state_dict()[key], before[key]) for key in before))


class IntegrationTests(unittest.TestCase):
    def test_final_evaluation_never_uses_partial_or_old_fallback_episodes(self):
        episodes = [Trajectory(0, 10, np.stack([rgb(0), rgb(1)]), 0),
                    Trajectory(1, 79, np.stack([rgb(0), rgb(1)]), 0),
                    Trajectory(2, 80, np.stack([rgb(0), rgb(1)]), 0),
                    Trajectory(3, 90, np.stack([rgb(0), rgb(1)]), 0),
                    Trajectory(4, 95, np.stack([rgb(0), rgb(1)]), 0, complete=False),
                    Trajectory(5, 104, np.stack([rgb(0), rgb(1)]), 0)]
        selected = select_recent_evaluation(episodes, budget=100, fraction=0.2, count=1)
        self.assertEqual([episode.id for episode in selected], [3])
        selected = select_recent_evaluation(episodes[:2], budget=100, fraction=0.2, count=5)
        self.assertEqual(selected, [])
        selected = select_recent_evaluation(episodes, budget=100, fraction=0.2, count=5)
        self.assertEqual([episode.id for episode in selected], [2, 3])

    def test_companion_random_stream_does_not_advance_dreamer_stream(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        original = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        state = IsolatedRandomState(123, "cpu")
        with state.use():
            first = random.random(), float(np.random.rand()), float(torch.rand(()))
        with state.use():
            second = random.random(), float(np.random.rand()), float(torch.rand(()))
        self.assertNotEqual(first, second)
        self.assertEqual(random.getstate(), original[0])
        self.assertEqual(np.random.get_state()[0], original[1][0])
        np.testing.assert_array_equal(np.random.get_state()[1], original[1][1])
        torch.testing.assert_close(torch.get_rng_state(), original[2], rtol=0, atol=0)

    def test_goal_training_interactions_are_separate_from_source_budget_and_corpus(self):
        from omegaconf import OmegaConf

        config = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/goal_conditioning/default.yaml")
        config.device = "cpu"
        config.policy.hidden_dim = 16
        config.policy.rollout_steps = 2
        config.policy.max_steps = 3
        config.classifier.embedding_dim = 16
        config.evaluation.episodes = 1
        config.evaluation.max_steps = 3
        config.evaluation.save_video = False
        environments = []

        def factory(index):
            environment = CounterEnv(length=3)
            environments.append(environment)
            return environment

        with tempfile.TemporaryDirectory() as temporary:
            manager = GoalTraining(config, SimpleNamespace(action_repeat=4), None,
                                   temporary, env_factory=factory)
            for base in (80, 88):
                for index in range(3):
                    transition = {"image": torch.from_numpy(rgb(index)[None]),
                                  "is_first": torch.tensor([[index == 0]]),
                                  "is_last": torch.tensor([[index == 2]]),
                                  "reward": torch.tensor([[0. if index == 0 else 2.]])}
                    manager.observe(transition, base + index * 4)
            source_ids = [episode.id for episode in manager.store.episodes]
            source_frames = [episode.frames.copy() for episode in manager.store.episodes]
            manager.classifier.updates = 1
            with patch.object(manager.classifier, "update", return_value={"status": "updated"}), \
                 patch.object(manager.classifier, "snapshot", return_value=(ExactImageClassifier(), 7)), \
                 patch.object(manager, "_save"):
                training = manager.maybe_train(96)
                report = manager.finish(100, 100)
            self.assertIsNotNone(training)
            self.assertGreater(manager.train_env_steps, 0)
            self.assertGreater(manager.eval_env_steps, 0)
            self.assertEqual(manager.train_env_steps + manager.eval_env_steps,
                             sum(environment.steps for environment in environments))
            self.assertEqual(report["dreamer_step"], 100)
            self.assertEqual(report["dreamer_budget"], 100)
            self.assertEqual(report["source_ids"], [source_ids[-1]])
            self.assertEqual([episode.id for episode in manager.store.episodes], source_ids)
            self.assertEqual(manager.completed, 2)
            for original, episode in zip(source_frames, manager.store.episodes):
                np.testing.assert_array_equal(original, episode.frames)
            self.assertTrue(all(environment.closed for environment in environments))


if __name__ == "__main__":
    unittest.main()
