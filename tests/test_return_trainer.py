"""Exercise periodic orchestration in the real OnlineTrainer loop."""

from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict

from trainer import OnlineTrainer


def observation(index, kind=0, last=False):
    return TensorDict(dict(image=torch.full((1, 2, 2, 3), index, dtype=torch.int64),
                           kind=torch.tensor([[kind]]),
                           reward=torch.tensor([[float(index)]]),
                           is_first=torch.tensor([[index == 0]]),
                           is_last=torch.tensor([[last]]),
                           is_terminal=torch.tensor([[last]])), batch_size=(1,))


class FakeParallelEnvironment:
    env_num = 1

    def __init__(self, events, kind=0, length=20):
        self.events = events
        self.kind, self.length = kind, length
        self.index = None
        self.actions = []
        self.resets = 0

    def step(self, action, done):
        if bool(done[0]):
            self.resets += 1
            self.index = 0
            self.events.append(('reset', self.kind))
        else:
            self.index += 1
            self.actions.append(action.clone())
            self.events.append(('step', self.kind, self.index, int(action.argmax(-1))))
        terminal = self.index == self.length
        return observation(self.index, self.kind, terminal), torch.tensor([terminal])


class FakeAgent:
    device = torch.device('cpu')

    def __init__(self, events):
        self.events = events
        self.training = True
        self.history = []
        self.updates = 0
        self.inside_phase = False

    def get_initial_state(self, batch):
        return TensorDict(dict(prev_action=torch.zeros(batch, 3),
                               stoch=torch.zeros(batch, 1),
                               deter=torch.zeros(batch, 1)), batch_size=(batch,))

    def act(self, obs, state, eval=False):
        index = int(obs['image'][0, 0, 0, 0])
        kind = int(obs['kind'][0])
        self.events.append(('act', kind, index))
        self.history.append(dict(kind=kind, index=index, previous_action=state['prev_action'].clone(),
                                 previous_deter=float(state['deter'][0]), eval=eval))
        action = torch.nn.functional.one_hot(torch.tensor([(index + 1) % 3]), 3).float()
        deter = torch.ones(1, 1) if bool(obs['is_first'][0]) else state['deter'] + 1
        return action, TensorDict(dict(prev_action=action.clone(), stoch=torch.full((1, 1), float(index)),
                                       deter=deter), batch_size=(1,))

    def update(self, replay):
        if self.inside_phase:
            raise AssertionError('Dreamer updated during the assisted phase')
        self.updates += 1
        self.events.append(('update', self.updates))
        return {'loss': torch.tensor(1.)}

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode


class FakeReplay:
    def __init__(self):
        self.rows = []

    def count(self):
        return len(self.rows)

    def add_transition(self, transition):
        self.rows.append(transition.clone())


class FakeLogger:
    def __init__(self):
        self.values = []
        self.writes = []

    def scalar(self, name, value):
        self.values.append((name, float(value)))

    def write(self, step, **kwargs):
        self.writes.append((step, kwargs))


class FakeReturnPhase:
    def __init__(self, events, no_op_first=False):
        self.events = events
        self.records = []
        self.runs = []
        self.next_milestone = 3
        self.assisted_interactions = 0
        self.no_op_first = no_op_first

    def record_ordinary(self, transition, action, previous_done, done, interactions):
        self.events.append(('record', int(transition['kind'][0]), interactions))
        self.records.append(dict(transition=transition.clone(), action=action.clone(),
                                 previous_done=previous_done.clone(), done=done.clone(),
                                 interactions=interactions))

    def due(self, interactions):
        return interactions >= self.next_milestone

    def run(self, agent, replay, interactions):
        self.runs.append(interactions)
        self.events.append(('phase', interactions))
        if self.no_op_first and len(self.runs) == 1:
            return False
        before = agent.updates
        agent.inside_phase = True
        try:
            # Independent recurrent work must leave the ordinary live state alone.
            agent.act(observation(0, kind=2), agent.get_initial_state(1))
            source = replay.rows[-1].clone()
            source['kind'][:] = 2
            source['episode'][:] = -1
            for _ in range(100):
                replay.add_transition(source)
            self.assisted_interactions += 10000
        finally:
            agent.inside_phase = False
        if agent.updates != before:
            raise AssertionError('Unexpected Dreamer update in assisted phase')
        self.next_milestone += 3
        return True


class ReturnTrainerTest(unittest.TestCase):
    def run_trainer(self, no_op_first=False, use_phase=True):
        events = []
        config = SimpleNamespace(steps=8 * 4, pretrain=1, eval_every=2 * 4,
                                 eval_episode_num=1, batch_size=1, batch_length=1,
                                 train_ratio=1, update_log_every=1000, action_repeat=4)
        replay, logger, agent = FakeReplay(), FakeLogger(), FakeAgent(events)
        ordinary = FakeParallelEnvironment(events)
        evaluation = FakeParallelEnvironment(events, kind=1, length=2)
        trainer = OnlineTrainer(config, replay, logger, ordinary, evaluation)
        phase = FakeReturnPhase(events, no_op_first) if use_phase else None
        trainer.return_exploration = phase
        trainer.begin(agent)
        return events, ordinary, evaluation, replay, logger, agent, phase

    def test_ordinary_clock_recording_and_state_survive_assisted_rounds(self):
        events, ordinary, evaluation, replay, logger, agent, phase = self.run_trainer()
        self.assertEqual(phase.runs, [3, 6])
        self.assertEqual(phase.assisted_interactions, 20000)
        self.assertEqual(len(ordinary.actions), 8)
        self.assertEqual(ordinary.resets, 1)
        self.assertGreater(len(evaluation.actions), 0)
        self.assertEqual(len(phase.records), 9)  # Initial frame plus eight actual actions.
        self.assertEqual([r['interactions'] for r in phase.records], list(range(9)))
        self.assertTrue(bool(phase.records[0]['previous_done'][0]))
        self.assertFalse(any(bool(r['previous_done'][0]) for r in phase.records[1:]))
        for index, recorded in enumerate(phase.records):
            self.assertEqual(int(recorded['transition']['kind'][0]), 0)
            if index:
                torch.testing.assert_close(recorded['action'], ordinary.actions[index - 1])
            event_position = events.index(('record', 0, index))
            self.assertEqual(events[event_position + 1], ('act', 0, index))
        history = [row for row in agent.history if row['kind'] == 0]
        self.assertEqual([row['index'] for row in history], list(range(9)))
        self.assertEqual([row['previous_deter'] for row in history], list(map(float, range(9))))
        for index, row in enumerate(history[1:]):
            torch.testing.assert_close(row['previous_action'], ordinary.actions[index])
        ordinary_rows = [row for row in replay.rows if int(row['kind'][0]) == 0]
        self.assertEqual([int(row['episode'][0]) for row in ordinary_rows], [0, 0, 0, 0, 1, 1, 1, 2, 2])
        self.assertEqual(len(replay.rows), 209)
        self.assertFalse(any(int(row['kind'][0]) == 1 for row in replay.rows))
        totals = [value for name, value in logger.values if name == 'return/total_interactions']
        self.assertEqual(totals, [10003., 20006.])
        baseline = self.run_trainer(use_phase=False)
        self.assertEqual(agent.updates, baseline[5].updates)
        self.assertEqual(agent.updates, 6)

    def test_deferred_empty_round_does_not_change_replay_segment(self):
        events, ordinary, evaluation, replay, logger, agent, phase = self.run_trainer(no_op_first=True)
        self.assertEqual(phase.runs, [3, 4, 6])
        self.assertEqual(phase.assisted_interactions, 20000)
        ordinary_rows = [row for row in replay.rows if int(row['kind'][0]) == 0]
        self.assertEqual([int(row['episode'][0]) for row in ordinary_rows], [0, 0, 0, 0, 0, 1, 1, 2, 2])
        ordinary_logged = [value for name, value in logger.values if name == 'return/ordinary_interactions']
        self.assertEqual(ordinary_logged, [4., 6.])
        self.assertEqual(len(ordinary.actions), 8)
        self.assertEqual(ordinary.resets, 1)
        self.assertEqual(agent.updates, 6)


if __name__ == '__main__':
    unittest.main()
