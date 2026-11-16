import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from PIL import Image

from episode_archive import EpisodeArchive, EpisodeRecorder, resize_frame


class EpisodeArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata = {"num_actions": 4, "env_config": {"task": "atari_montezuma_revenge"}}
        self.archive = EpisodeArchive(self.root / "archive", self.metadata)

    def episode(self, truncated=False):
        recorder = EpisodeRecorder(np.zeros((64, 64, 3), np.uint8), source="ordinary")
        recorder.step(2, np.ones((64, 64, 3), np.uint8), 1.5)
        recorder.step(3, np.full((64, 64, 3), 2, np.uint8), -1.,
                      terminated=not truncated, truncated=truncated)
        return recorder.finish(end_step=10)

    def test_completed_alignment_and_compressed_snapshot(self):
        episode_id = self.archive.append(self.episode())
        path = self.root / "round.pkl"
        self.assertEqual(self.archive.snapshot(path), (episode_id,))
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        self.assertEqual(payload["kind"], "atari_episode_archive")
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(payload["metadata"], self.metadata)
        record = payload["episodes"][0]
        self.assertNotIn("frames", record)
        with np.load(io.BytesIO(record["payload_npz"]), allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["actions"], [2, 3])
            np.testing.assert_array_equal(arrays["frames"][:, 0, 0, 0], [0, 1, 2])
            np.testing.assert_array_equal(arrays["terminated"], [False, True])
        decoded = self.archive.load(episode_id)
        self.assertEqual(decoded["frames"].shape, (3, 84, 84, 3))
        self.assertEqual(decoded["frame_count"], 3)
        self.assertEqual(decoded["source"], "ordinary")

    def test_incomplete_episodes_excluded(self):
        recorder = EpisodeRecorder(np.zeros((84, 84, 3), np.uint8))
        recorder.step(0, np.ones((84, 84, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "complete"):
            recorder.finish()
        self.assertEqual(self.archive.pending_ids, ())
        with self.assertRaises(ValueError):
            self.archive.snapshot(self.root / "empty.pkl")
        recorder.step(1, np.ones((84, 84, 3), np.uint8), truncated=True)
        self.archive.append(recorder.finish())
        with self.assertRaisesRegex(ValueError, "ended"):
            recorder.step(1, np.ones((84, 84, 3), np.uint8))

    def test_generated_episodes_survive_consuming_source_snapshot(self):
        first = self.archive.append(self.episode())
        second = self.archive.append(self.episode())
        source_ids = self.archive.snapshot(self.root / "source.pkl")
        generated = self.archive.append(self.episode(truncated=True))
        self.archive.consume(source_ids)
        self.assertEqual(self.archive.pending_ids, (generated,))
        reloaded = EpisodeArchive(self.root / "archive", self.metadata)
        self.assertEqual(reloaded.pending_ids, (generated,))
        self.assertEqual(reloaded.load(first)["id"], first)
        self.assertEqual(reloaded.load(second)["id"], second)
        self.assertEqual(reloaded.append(self.episode()), generated + 1)
        with self.assertRaises(ValueError):
            reloaded.consume(source_ids)

    def test_recovers_record_committed_before_manifest(self):
        old_manifest = self.archive.manifest_path.read_text()
        episode_id = self.archive.append(self.episode())
        self.archive.manifest_path.write_text(old_manifest)
        recovered = EpisodeArchive(self.root / "archive", self.metadata)
        self.assertEqual(recovered.pending_ids, (episode_id,))
        self.assertEqual(json.loads(recovered.manifest_path.read_text())["next_id"], episode_id + 1)

    def test_rejects_incorrect_alignment_and_termination(self):
        episode = self.episode()
        episode["actions"] = np.asarray([1])
        with self.assertRaisesRegex(ValueError, "align"):
            self.archive.append(episode)
        episode = self.episode()
        episode["terminated"][0] = True
        with self.assertRaisesRegex(ValueError, "final transition"):
            self.archive.append(episode)
        episode = self.episode()
        episode["actions"][0] = self.metadata["num_actions"]
        with self.assertRaisesRegex(ValueError, "action space"):
            self.archive.append(episode)

    def test_resize_uses_bilinear_and_copies_source(self):
        frame = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        expected = np.asarray(Image.fromarray(frame).resize((84, 84), Image.Resampling.BILINEAR))
        np.testing.assert_array_equal(resize_frame(frame), expected)
        frame84 = np.zeros((84, 84, 3), np.uint8)
        copied = resize_frame(frame84)
        frame84[:] = 10
        self.assertFalse(copied.any())


if __name__ == "__main__":
    unittest.main()
