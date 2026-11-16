"""Exercise auxiliary-episode insertion with the real TorchRL replay sampler."""

from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict

from buffer import Buffer, OnlineQueue


def transition(segment, marker, first=False):
    # Marker identifies an observation. Its action is the outgoing action, so
    # training on marker t must receive action (t - 1) + 0.25 after sampling.
    latent = segment * 1000 + marker
    return TensorDict({
        'observation': torch.tensor([[marker]]),
        'action': torch.tensor([[marker + 0.25]]),
        'episode': torch.tensor([segment]),
        'is_first': torch.tensor([[first]]),
        'stoch': torch.full((1, 2, 2), float(latent)),
        'deter': torch.full((1, 2), float(latent)),
    }, batch_size=(1,))


class ReturnReplayTests(unittest.TestCase):
    def test_online_windows_discard_partial_segments_at_interruption(self):
        queue = OnlineQueue(window=4, max_windows=10)
        row = 0
        for segment, markers in ((0, range(2)), (1, range(100, 106)),
                                 (2, range(200, 206)), (3, range(2, 6))):
            for position, marker in enumerate(markers):
                queue.push(transition(segment, marker, first=position == 0 and segment != 3),
                           torch.tensor([[row, 0]]))
                row += 1

        data, envs, rows = queue.take(3)
        torch.testing.assert_close(data['episode'],
                                   torch.tensor([[1] * 4, [2] * 4, [3] * 4]))
        torch.testing.assert_close(data['observation'].squeeze(-1),
                                   torch.tensor([[100, 101, 102, 103],
                                                 [200, 201, 202, 203], [2, 3, 4, 5]]))
        torch.testing.assert_close(rows, torch.tensor([[2, 3, 4, 5],
                                                      [8, 9, 10, 11], [14, 15, 16, 17]]))
        torch.testing.assert_close(envs, torch.zeros(3, dtype=torch.long))
        self.assertFalse(bool(data['is_first'][2].any()),
                         'Resuming ordinary collection does not reset the environment')
        self.assertIsNone(queue.take(1))

    def make_buffer(self, online, max_size=100):
        replay = Buffer(SimpleNamespace(device='cpu', storage_device='cpu', batch_size=1,
                                        batch_length=3, max_size=max_size, online=online,
                                        update_latents=False))
        # Ordinary collection pauses after two observations, two complete assisted
        # episodes are inserted, and the same ordinary episode resumes at marker 2.
        for segment, markers in ((0, range(2)), (1, range(100, 107)),
                                 (2, range(200, 207)), (3, range(2, 9))):
            for position, marker in enumerate(markers):
                replay.add_transition(transition(segment, marker,
                                                  first=position == 0 and segment != 3))
        return replay

    def assert_contiguous_sample(self, sample):
        data, index, (stoch, deter) = sample
        segments = data['episode'][0]
        self.assertTrue(bool((segments == segments[0]).all()))
        segment = int(segments[0])
        self.assertIn(segment, (1, 2, 3))
        markers = data['observation'][0, :, 0]
        torch.testing.assert_close(markers[1:] - markers[:-1], torch.ones(2, dtype=torch.long))
        if segment == 1:
            self.assertTrue(bool(((markers >= 101) & (markers <= 106)).all()))
        elif segment == 2:
            self.assertTrue(bool(((markers >= 201) & (markers <= 206)).all()))
        else:
            self.assertTrue(bool(((markers >= 3) & (markers <= 8)).all()))
            self.assertFalse(bool(data['is_first'].any()))

        torch.testing.assert_close(data['action'][0, :, 0], markers.float() - 0.75)
        # The context observation must also belong to the same uninterrupted
        # segment; homogeneous target observations alone would not establish this.
        context = segment * 1000 + int(markers[0]) - 1
        torch.testing.assert_close(stoch, torch.full((1, 2, 2), float(context)))
        torch.testing.assert_close(deter, torch.full((1, 2), float(context)))
        torch.testing.assert_close(index[1], torch.zeros((1, 3), dtype=torch.long))
        return segment

    def test_real_buffer_sampling_preserves_segments_and_action_alignment(self):
        for online in (False, True):
            with self.subTest(online=online):
                torch.manual_seed(12)
                replay = self.make_buffer(online)
                if online:
                    # The bounded queue retains the newest two complete windows.
                    self.assertEqual(self.assert_contiguous_sample(replay.sample()), 2)
                    self.assertEqual(self.assert_contiguous_sample(replay.sample()), 3)
                seen = {self.assert_contiguous_sample(replay.sample()) for _ in range(64)}
                self.assertEqual(seen, {1, 2, 3})

    def test_real_buffer_sampling_after_storage_wrap(self):
        torch.manual_seed(7)
        replay = self.make_buffer(online=False, max_size=17)
        seen = {self.assert_contiguous_sample(replay.sample()) for _ in range(64)}
        self.assertEqual(seen, {2, 3})


if __name__ == '__main__':
    unittest.main()
