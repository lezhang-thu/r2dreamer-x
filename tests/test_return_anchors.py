import json
import unittest

import numpy as np

from return_anchors import AnchorBank


class AnchorBankTests(unittest.TestCase):
    def setUp(self):
        # 0~1 and 1~2, but 0!~2. Later 3 matches both representatives 0 and 2.
        self.table = np.eye(5)
        for first, second in ((0, 1), (1, 2), (0, 3), (2, 3)):
            self.table[first, second] = self.table[second, first] = 0.5
        self.calls = []

        def similarity(left, right):
            self.calls.append((len(left), len(right)))
            return self.table[left[:, 0].astype(int)[:, None], right[:, 0].astype(int)[None]]

        self.similarity = similarity
        self.demos = [dict(id=10, references=np.array([[0.], [1.]])),
                      dict(id=20, references=np.array([[2.], [3.]]))]

    def bank(self):
        return AnchorBank.build(self.demos, self.similarity, chunk_size=2)

    def test_streaming_counts_increment_all_matches_without_merging(self):
        bank = self.bank()
        np.testing.assert_array_equal(bank.references[:, 0], [0, 2])
        np.testing.assert_array_equal(bank.counts, [3, 2])
        self.assertEqual(bank.origins, ((10, 0), (20, 0)))

    def test_second_pass_indexes_earlier_occurrences_for_later_representatives(self):
        bank = self.bank()
        self.assertEqual(bank.prefixes[0], ((10, 0), (10, 1), (20, 1)))
        self.assertEqual(bank.prefixes[1], ((10, 1), (20, 0), (20, 1)))
        self.assertTrue(all(n <= 2 and m <= 2 for n, m in self.calls))

    def test_generated_rollouts_only_change_counts(self):
        bank = self.bank()
        references = bank.references.copy()
        prefixes = bank.prefixes
        increments = bank.update(np.array([[1.], [4.], [3.]]))
        np.testing.assert_array_equal(increments, [2, 2])
        np.testing.assert_array_equal(bank.counts, [5, 4])
        np.testing.assert_array_equal(bank.references, references)
        self.assertEqual(bank.prefixes, prefixes)
        self.assertEqual(len(bank), 2)
        np.testing.assert_array_equal(bank.update(np.empty((0, 1))), [0, 0])

    def test_sampling_uses_current_counts_then_uniform_prefix(self):
        bank = self.bank()
        test = self

        class RNG:
            def choice(self, count, p):
                test.assertEqual(count, 2)
                np.testing.assert_allclose(p, [0.4, 0.6])
                return 1

            def integers(self, count):
                test.assertEqual(count, 3)
                return 0

        self.assertEqual(bank.sample(RNG()), (1, (10, 1)))
        bank.update(np.array([[0.]]))
        np.testing.assert_allclose(bank.probabilities(), [1 / 3, 2 / 3])

    def test_exact_origin_is_available_when_self_similarity_fails(self):
        bank = AnchorBank.build([dict(id=5, references=np.array([[7.]]))],
                                lambda left, right: np.zeros((len(left), len(right))))
        self.assertEqual(bank.prefixes, (((5, 0),),))
        np.testing.assert_array_equal(bank.counts, [1])
        self.assertEqual(bank.sample(np.random.default_rng(0)), (0, (5, 0)))

    def test_state_survives_json_roundtrip(self):
        for demo in self.demos:
            demo['references'] = demo['references'].astype(np.float32)
        bank = self.bank()
        bank.update(np.array([[1.]]))
        saved = json.loads(json.dumps(bank.state_dict()))
        restored = AnchorBank.from_state_dict(saved, self.similarity)
        np.testing.assert_array_equal(restored.references, bank.references)
        self.assertEqual(restored.references.dtype, bank.references.dtype)
        np.testing.assert_array_equal(restored.counts, bank.counts)
        self.assertEqual(restored.prefixes, bank.prefixes)
        self.assertEqual(restored.origins, bank.origins)
        self.assertEqual(restored.sample(np.random.default_rng(2)),
                         bank.sample(np.random.default_rng(2)))

    def test_empty_invalid_and_duplicate_sources_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'No anchors'):
            AnchorBank.build([], self.similarity)
        with self.assertRaisesRegex(ValueError, 'unique'):
            AnchorBank.build(self.demos + self.demos, self.similarity)
        with self.assertRaisesRegex(ValueError, 'finite'):
            AnchorBank.build([dict(id=0, references=np.array([[np.nan]]))], self.similarity)
        with self.assertRaisesRegex(ValueError, 'probabilities'):
            AnchorBank.build(self.demos, lambda left, right: np.ones((1, 1)) * 2)

    def test_representatives_own_their_storage_and_are_read_only(self):
        bank = self.bank()
        self.demos[0]['references'][0, 0] = 4
        self.assertEqual(bank.references[0, 0], 0)
        with self.assertRaises(ValueError):
            bank.references[0, 0] = 4


if __name__ == '__main__':
    unittest.main()
