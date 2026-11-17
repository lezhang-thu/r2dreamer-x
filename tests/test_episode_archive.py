import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from PIL import Image

from episode_archive import EpisodeArchive, EpisodeRecorder, resize_frame


class EpisodeArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = {"num_actions": 4, "env_config": {"task": "atari_montezuma_revenge"}}
        self.archive = EpisodeArchive(self.metadata)

    def episode(self, truncated=False):
        recorder = EpisodeRecorder(np.zeros((64, 64, 3), np.uint8), source="ordinary")
        recorder.step(2, np.ones((64, 64, 3), np.uint8), 1.5)
        recorder.step(3, np.full((64, 64, 3), 2, np.uint8), -1.,
                      terminated=not truncated, truncated=truncated)
        return recorder.finish(end_step=10)

    def test_materialize_shares_decoded_arrays_and_releases_compressed_source(self):
        episode_id = self.archive.append(self.episode())
        self.assertIn('payload_npz', self.archive._records[episode_id])
        self.assertNotIn('frames', self.archive._records[episode_id])
        before = self.archive.fingerprint()
        with patch('subprocess.run', side_effect=AssertionError('No IPC is needed')):
            records = self.archive.materialize()
        decoded = records[0]
        self.assertIs(decoded, self.archive._records[episode_id])
        self.assertNotIn('payload_npz', decoded)
        self.assertIs(self.archive.load(episode_id), decoded)
        self.assertIs(self.archive.materialize()[0]['frames'], decoded['frames'])
        self.assertEqual(self.archive.fingerprint(), before)
        self.assertEqual(decoded['frames'].shape, (3, 84, 84, 3))
        self.assertEqual(decoded['frame_count'], 3)
        self.assertEqual(decoded['source'], 'ordinary')
        np.testing.assert_array_equal(decoded['actions'], [2, 3])
        np.testing.assert_array_equal(decoded['frames'][:, 0, 0, 0], [0, 1, 2])
        np.testing.assert_array_equal(decoded['terminated'], [False, True])
        for key in ('frames', 'actions', 'rewards', 'terminated', 'truncated'):
            self.assertFalse(decoded[key].flags.writeable)
        with self.assertRaises(ValueError):
            decoded['frames'][0, 0, 0, 0] = 255
        self.assertEqual(list(self.root.iterdir()), [])

    def test_incomplete_episodes_excluded(self):
        recorder = EpisodeRecorder(np.zeros((84, 84, 3), np.uint8))
        recorder.step(0, np.ones((84, 84, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, 'complete'):
            recorder.finish()
        self.assertEqual(self.archive.pending_ids, ())
        with self.assertRaises(ValueError):
            self.archive.materialize()
        with self.assertRaises(ValueError):
            self.archive.fingerprint()
        recorder.step(1, np.ones((84, 84, 3), np.uint8), truncated=True)
        self.archive.append(recorder.finish())
        with self.assertRaisesRegex(ValueError, 'ended'):
            recorder.step(1, np.ones((84, 84, 3), np.uint8))

    def test_consume_releases_source_arrays_but_keeps_new_episodes(self):
        first = self.archive.append(self.episode())
        second = self.archive.append(self.episode())
        source_ids = self.archive.pending_ids
        records = self.archive.materialize(source_ids)
        frames = weakref.ref(records[0]['frames'])
        generated = self.archive.append(self.episode(truncated=True))
        self.archive.consume(source_ids)
        self.assertEqual(self.archive.pending_ids, (generated,))
        self.assertEqual(set(self.archive._records), {generated})
        self.assertEqual(set(self.archive._digests), {generated})
        with self.assertRaises(KeyError):
            self.archive.load(first)
        with self.assertRaises(KeyError):
            self.archive.load(second)
        # Consumers can finish using shared records after releasing pool ownership.
        self.assertIsNotNone(frames())
        del records
        gc.collect()
        self.assertIsNone(frames())
        self.assertEqual(self.archive.load(generated)['id'], generated)
        self.assertEqual(self.archive.append(self.episode()), generated + 1)
        with self.assertRaises(ValueError):
            self.archive.consume(source_ids)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failed_decode_preserves_sources_for_retry(self):
        first = self.archive.append(self.episode())
        second = self.archive.append(self.episode(truncated=True))
        original_load = self.archive.load

        def fail_second(episode_id):
            if episode_id == second:
                with patch('episode_archive.np.load', side_effect=RuntimeError('simulated decode failure')):
                    return original_load(episode_id)
            return original_load(episode_id)

        before = self.archive.fingerprint()
        with patch.object(self.archive, 'load', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'decode failure'):
                self.archive.materialize()
        self.assertEqual(self.archive.pending_ids, (first, second))
        self.assertNotIn('payload_npz', self.archive._records[first])
        self.assertIn('payload_npz', self.archive._records[second])
        first_frames = self.archive._records[first]['frames']
        retried = self.archive.materialize()
        self.assertIs(retried[0]['frames'], first_frames)
        self.assertTrue(retried[1]['complete'])
        self.assertEqual(self.archive.fingerprint(), before)

    def test_fingerprint_tracks_contents_order_metadata_and_subset(self):
        first = self.archive.append(self.episode())
        second = self.archive.append(self.episode(truncated=True))
        copied = EpisodeArchive(self.metadata)
        copied.append(self.episode())
        copied.append(self.episode(truncated=True))
        expected = self.archive.fingerprint()
        self.assertEqual(len(expected), 64)
        self.assertEqual(copied.fingerprint(), expected)
        self.assertNotEqual(self.archive.fingerprint((first,)), expected)
        self.assertNotEqual(self.archive.fingerprint((second, first)), expected)
        self.archive.materialize((first,))
        self.assertEqual(self.archive.fingerprint(), expected)
        different = self.episode()
        different['frames'][0, 0, 0, 0] = 17
        changed = EpisodeArchive(self.metadata)
        changed.append(different)
        self.assertNotEqual(changed.fingerprint(), self.archive.fingerprint((first,)))
        other_metadata = EpisodeArchive(self.metadata | {'note': 'changed'})
        other_metadata.append(self.episode())
        self.assertNotEqual(other_metadata.fingerprint(), self.archive.fingerprint((first,)))

    def test_rejects_invalid_subset_before_materializing_any_sources(self):
        first = self.archive.append(self.episode())
        second = self.archive.append(self.episode())
        for invalid in ((first, first), (second + 1,), (first, second + 1), (True,)):
            with self.assertRaises(ValueError):
                self.archive.materialize(invalid)
            with self.assertRaises(ValueError):
                self.archive.fingerprint(invalid)
            with self.assertRaises(ValueError):
                self.archive.consume(invalid)
        self.assertEqual(self.archive.pending_ids, (first, second))
        self.assertTrue(all('payload_npz' in record for record in self.archive._records.values()))

    def test_rejects_incorrect_alignment_and_termination(self):
        episode = self.episode()
        episode['actions'] = np.asarray([1])
        with self.assertRaisesRegex(ValueError, 'align'):
            self.archive.append(episode)
        episode = self.episode()
        episode['terminated'][0] = True
        with self.assertRaisesRegex(ValueError, 'final transition'):
            self.archive.append(episode)
        episode = self.episode()
        episode['actions'][0] = self.metadata['num_actions']
        with self.assertRaisesRegex(ValueError, 'action space'):
            self.archive.append(episode)

    def test_resize_uses_bilinear_and_copies_source(self):
        frame = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        expected = np.asarray(Image.fromarray(frame).resize((84, 84), Image.Resampling.BILINEAR))
        np.testing.assert_array_equal(resize_frame(frame), expected)
        frame84 = np.zeros((84, 84, 3), np.uint8)
        copied = resize_frame(frame84)
        frame84[:] = 10
        self.assertFalse(copied.any())


if __name__ == '__main__':
    unittest.main()
