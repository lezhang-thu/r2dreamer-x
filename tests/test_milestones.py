"""Behavioral checks for temporal sampling, replay, and greedy extraction."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch

from extract_milestones import extract_anchors, load_frames, save_video
from milestone_model import PairClassifier, frames_to_tensor, load_classifier
from prepare_milestone_data import replay_episode
from train_milestone_classifier import EpisodeFrames, SkipGramSampler, pair_metrics


torch.set_num_threads(2)


class LookupModel(torch.nn.Module):
    """Prescribed pair scores, including a return to an earlier visual state."""

    def encode(self, frames):
        return (frames[:, 0, 0, 0] * 255).round().unsqueeze(1)

    def pair_logits(self, left, right):
        probabilities = {(0, 1): 0.5, (0, 3): 0.2, (0, 4): 0.9,
                         (3, 4): 0.5, (3, 5): 0.4, (5, 6): 0.9}
        result = [probabilities.get(tuple(sorted((int(a), int(b)))), 0.8)
                  for a, b in zip(left[:, 0], right[:, 0])]
        return torch.logit(torch.tensor(result, device=left.device))


def numbered_frames(count):
    return np.broadcast_to(np.arange(count, dtype=np.uint8)[:, None, None, None],
                           (count, 84, 84, 3)).copy()


class ExtractionTests(unittest.TestCase):
    def test_first_crossing_is_exact_and_batch_independent(self):
        expected_pairs = [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (5, 6)]
        for batch_size in (1, 2, 4, 64):
            with self.subTest(batch_size=batch_size):
                anchors, comparisons = extract_anchors(
                    LookupModel(), numbered_frames(7), batch_size=batch_size)
                self.assertEqual(anchors, [0, 3, 5])
                self.assertEqual([(x["anchor_index"], x["frame_index"])
                                  for x in comparisons], expected_pairs)

    def test_initial_frame_only_and_terminal_is_not_forced(self):
        anchors, comparisons = extract_anchors(LookupModel(), numbered_frames(1))
        self.assertEqual((anchors, comparisons), ([0], []))
        anchors, comparisons = extract_anchors(LookupModel(), numbered_frames(3))
        self.assertEqual(anchors, [0])
        self.assertEqual(len(comparisons), 2)

    def test_invalid_inputs_are_rejected(self):
        for threshold in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                extract_anchors(LookupModel(), numbered_frames(3), threshold=threshold)
        with self.assertRaises(ValueError):
            extract_anchors(LookupModel(), numbered_frames(0))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gray.npy"
            np.save(path, np.zeros((2, 84, 84, 1), np.uint8))
            with self.assertRaises(ValueError):
                load_frames(path)

    def test_mp4_preserves_count_resolution_timing_and_rgb_order(self):
        frames = np.zeros((6, 84, 84, 3), np.uint8)
        frames[:3, :, :, 0] = 240
        frames[3:, :, :, 2] = 240
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.mp4"
            save_video(frames, path, 15)
            cap = cv2.VideoCapture(str(path))
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 15)
            decoded = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            self.assertEqual(np.asarray(decoded).shape, frames.shape)
            self.assertGreater(decoded[0][:, :, 0].mean(), 200)
            self.assertLess(decoded[0][:, :, 2].mean(), 20)
            self.assertGreater(decoded[-1][:, :, 2].mean(), 200)


class FakeAtari:
    def __init__(self, *args, **kwargs):
        self.index = 0
        self.closed = False
        self.kwargs = kwargs

    def reset(self, seed):
        self.seed = seed
        return {"image": np.full((84, 84, 3), 10, np.uint8), "is_terminal": False}

    def step(self, action):
        self.index += 1
        terminal = self.index == 3
        return {"image": np.full((84, 84, 3), 10 + self.index, np.uint8),
                "is_terminal": terminal}, float(action), terminal, {}

    def close(self):
        self.closed = True


class ReplayTests(unittest.TestCase):
    def test_reset_and_successors_are_aligned_and_rewards_are_unclipped(self):
        environment = FakeAtari()
        with patch("envs.atari.Atari", return_value=environment) as constructor:
            frames, rewards, complete, terminal = replay_episode([1, 2, 3], 6)
        constructor.assert_called_once_with("MontezumaRevenge", sticky=False, gray=False, size=(84, 84))
        self.assertEqual(environment.seed, 0)
        np.testing.assert_array_equal(frames[:, 0, 0, 0], [10, 11, 12, 13])
        np.testing.assert_array_equal(rewards, [1, 2, 3])
        self.assertTrue(complete and terminal and environment.closed)

    def test_partial_episode_and_failure_cleanup(self):
        environment = FakeAtari()
        with patch("envs.atari.Atari", return_value=environment):
            _, _, complete, terminal = replay_episode([1], 1)
        self.assertFalse(complete or terminal)
        self.assertTrue(environment.closed)
        for actions, score in (([1], 2), ([1, 2, 3, 4], 10)):
            environment = FakeAtari()
            with patch("envs.atari.Atari", return_value=environment):
                with self.assertRaises(ValueError):
                    replay_episode(actions, score)
            self.assertTrue(environment.closed)


class SamplingTests(unittest.TestCase):
    def test_temporal_labels_never_cross_boundaries_and_are_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for i, length in enumerate((2, 9, 17)):
                path = Path(temporary) / f"episode_{i:04d}.npz"
                np.savez(path, frames=numbered_frames(length))
                paths.append(path)
            corpus = EpisodeFrames(paths)
            first = SkipGramSampler(corpus, window=3, negative_ratio=2, seed=7)
            second = SkipGramSampler(corpus, window=3, negative_ratio=2, seed=7)
            pairs = first.sample(3000)
            for a, b in zip(pairs, second.sample(3000)):
                np.testing.assert_array_equal(a, b)
            left, right, labels = pairs
            le, lt = corpus.locate(left)
            re, rt = corpus.locate(right)
            positive = labels == 1
            self.assertEqual(int(positive.sum()), 1000)
            self.assertTrue(np.all(le[positive] == re[positive]))
            self.assertTrue(np.all(np.abs(lt[positive] - rt[positive]) <= 3))
            self.assertTrue(np.all(left[positive] != right[positive]))
            known_positive = (le == re) & (np.abs(lt - rt) <= 3)
            self.assertFalse(np.any(known_positive[~positive]))
            self.assertTrue(np.any(lt[positive] == 0))
            self.assertTrue(np.any(lt[positive] == corpus.lengths[le[positive]] - 1))
            np.testing.assert_array_equal(corpus.get(np.array([0, 1, 2, 10, 11]))[:, 0, 0, 0],
                                          [0, 1, 0, 8, 0])

    def test_no_negative_possible_is_rejected_and_auc_handles_ties(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode_0000.npz"
            np.savez(path, frames=numbered_frames(7))
            with self.assertRaises(ValueError):
                SkipGramSampler(EpisodeFrames([path]), window=3)
        result = pair_metrics(np.zeros(4, np.float32), np.array([1, 0, 1, 0]))
        self.assertEqual(result["auroc"], 0.5)
        self.assertEqual(result["balanced_accuracy"], 0.5)


class ModelTests(unittest.TestCase):
    def test_pair_order_symmetry_and_checkpoint_roundtrip(self):
        torch.manual_seed(0)
        model = PairClassifier(embedding_dim=16).eval()
        frames = np.random.default_rng(0).integers(0, 256, (4, 84, 84, 3), dtype=np.uint8)
        tensors = frames_to_tensor(frames)
        self.assertEqual(tuple(tensors.shape), (4, 3, 84, 84))
        with torch.inference_mode():
            expected = model(tensors[:2], tensors[2:])
            torch.testing.assert_close(expected, model(tensors[2:], tensors[:2]), rtol=0, atol=0)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "classifier.pt"
                torch.save({"format_version": 1, "model_config": model.model_config,
                            "model_state": model.state_dict(), "threshold": 0.5}, path)
                loaded, metadata = load_classifier(path)
                self.assertFalse(loaded.training)
                self.assertEqual(metadata["threshold"], 0.5)
                torch.testing.assert_close(expected, loaded(tensors[:2], tensors[2:]), rtol=0, atol=0)

    def test_cli_exports_exact_rgb_anchor_and_metadata(self):
        model = PairClassifier(embedding_dim=16)
        for parameter in model.parameters():
            parameter.data.zero_()  # All pairs have exactly p=0.5: only f0 is selected.
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            checkpoint = temporary / "classifier.pt"
            torch.save({"format_version": 1, "model_config": model.model_config,
                        "model_state": model.state_dict(), "threshold": 0.5}, checkpoint)
            frames = np.zeros((4, 84, 84, 3), np.uint8)
            frames[:, :, :42, 0] = 230
            frames[:, :, 42:, 2] = 120
            trajectory = temporary / "trajectory.npz"
            np.savez(trajectory, frames=frames)
            output = temporary / "demo"
            result = subprocess.run([
                sys.executable, str(Path(__file__).resolve().parents[1] / "extract_milestones.py"),
                "--trajectory", str(trajectory), "--checkpoint", str(checkpoint),
                "--output", str(output), "--threads", "1",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "anchors.json").read_text())
            self.assertEqual(report["frame_count"], 4)
            self.assertEqual(report["subgoal_count"], 0)
            self.assertEqual(report["anchors"][0]["frame_index"], 0)
            with Image.open(output / report["anchors"][0]["file"]) as anchor:
                self.assertEqual(anchor.mode, "RGB")
                self.assertEqual(anchor.size, (84, 84))
                np.testing.assert_array_equal(np.asarray(anchor), frames[0])
            self.assertTrue((output / "trajectory.mp4").is_file())
            self.assertTrue((output / report["contact_sheets"][0]).is_file())


if __name__ == "__main__":
    unittest.main()
