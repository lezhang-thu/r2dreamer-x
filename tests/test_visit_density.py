"""Neighborhood counts and retrieved source frames agree with a direct oracle."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "variants-supervised-agent"))
import visit_density
from visit_density import retrieve_least_visited


def test_runs_count_reentry_and_episode_start_at_equal_frames():
    # Repeated 1 across episode boundaries must start a new visit. The last
    # trajectory ends inside 1's neighborhood; the visit counts at cutoff.
    x = np.array([[1], [1], [0], [1], [1], [1], [3], [1]], np.float32)
    y = torch.ones((len(x), 2))
    result = retrieve_least_visited(x, y, [0, 4, 8], radius=0, limit=100, batch_size=2)
    np.testing.assert_array_equal(result.frame_counts, [6, 6, 1, 6, 6, 6, 1, 6])
    np.testing.assert_array_equal(result.visit_counts, [4, 4, 1, 4, 4, 4, 1, 4])
    assert set(result.indices[:2]) == {2, 6}
    assert x[result.indices[2], 0] == 1
    assert result.neighbors is None and result.radius_samples == 0
    assert result.dino_scale == 1
    positive = retrieve_least_visited(x, y, [0, 4, 8], radius=0.01, batch_size=3)
    for key in ("frame_counts", "visit_counts"):
        np.testing.assert_array_equal(getattr(result, key), getattr(positive, key))
    assert set(positive.indices[:2]) == {2, 6}
    assert x[positive.indices[2], 0] == 1


def test_radius_calibration_excludes_self_and_uses_median_of_distances():
    x = np.array([[0], [3], [7], [30]], np.float32)
    y = np.ones_like(x)
    result = retrieve_least_visited(x, y, [0, 2, 4], neighbors=1, batch_size=3)
    assert result.radius_samples == 4 and result.neighbors == 1
    assert result.radius == pytest.approx(3.5 / x.astype(np.float64).std())
    result = retrieve_least_visited(x, y, [0, 2, 4], neighbors=20, radius_samples=2, seed=3)
    sampled = np.random.default_rng(3).choice(4, 2, replace=False)
    distances = np.abs(x.astype(float) - x.T) / x.astype(float).std()
    assert result.radius == pytest.approx(np.median(distances[sampled].max(axis=1)))
    assert result.radius_samples == 2 and result.neighbors == 3


def test_closed_ball_includes_members_exactly_at_its_radius():
    x = np.array([[-1.0], [1.0]], np.float32)
    y = np.zeros_like(x)
    for radius, counts, n_selected in [(2.0, [2, 2], 1), (np.nextafter(2.0, 0), [1, 1], 2)]:
        result = retrieve_least_visited(x, y, [0, 2], radius=radius, batch_size=1)
        np.testing.assert_array_equal(result.frame_counts, counts)
        assert len(set(result.indices)) == n_selected
        np.testing.assert_allclose(result.neighbor_distances, 2.0)


def test_all_frames_counted_before_suppression_and_scaling_matches_oracle():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(29, 31)).astype(np.float32) * 1000
    y = rng.normal(size=(29, 7)).astype(np.float32)
    offsets = [0, 5, 16, 29]
    result = retrieve_least_visited(x, y, offsets, radius=1.9, limit=5, batch_size=4)
    points = np.column_stack(((x - result.latent_mean) / result.latent_scale,
                              (y - result.dino_mean) / result.dino_scale))
    for block in [points[:, :31], points[:, 31:]]:
        np.testing.assert_allclose(np.square(block).sum(axis=1).mean(), 1.0)
    distances = np.linalg.norm(points[:, None] - points, axis=-1)
    inside = distances <= result.radius
    visits = np.zeros(len(x), dtype='int64')
    for begin, end in zip(offsets[:-1], offsets[1:]):
        for q, row in enumerate(inside[:, begin:end]):
            visits[q] += sum(member and (t == 0 or not row[t-1]) for t, member in enumerate(row))
    np.testing.assert_array_equal(result.visit_counts, visits)
    np.testing.assert_array_equal(result.frame_counts, inside.sum(axis=1))
    np.fill_diagonal(distances, np.inf)
    assert result.isolation_neighbors == 20
    isolation = np.sort(distances, axis=1)[:, 19]
    np.testing.assert_allclose(result.neighbor_distances, isolation[result.indices])
    episode = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    chosen_per_episode = np.zeros(len(offsets) - 1, dtype=int)
    available = np.ones(len(x), dtype=bool)
    frames = inside.sum(axis=1)
    for i in result.indices:
        eligible = np.flatnonzero(available)
        for criterion in (visits, frames, chosen_per_episode[episode]):
            eligible = eligible[criterion[eligible] == criterion[eligible].min()]
        assert i in eligible and isolation[i] == pytest.approx(isolation[eligible].max())
        available[inside[i]] = False
        chosen_per_episode[episode[i]] += 1
    assert len(result.indices) == 5 or not available.any()


def test_degenerate_radius_and_small_archives():
    x = np.ones((30, 8), dtype=np.float32)
    result = retrieve_least_visited(x, x, [0, 2, 26, 30], neighbors=20, radius_samples=4)
    assert result.radius == 0 and len(result.indices) == 1
    assert result.neighbor_distances[0] == 0
    assert np.all(result.visit_counts == 3) and np.all(result.frame_counts == 30)
    singleton = retrieve_least_visited(x[:1], x[:1], [0, 1])
    assert singleton.radius == 0 and singleton.visit_counts.tolist() == [1]
    with pytest.raises(ValueError, match="offsets"):
        retrieve_least_visited(x, x, [0, 30, 30])
    with pytest.raises(ValueError, match="Radius"):
        retrieve_least_visited(x, x, [0, 30], radius=np.nan)


def test_singletons_spread_across_eligible_episodes_and_reuse_counts(monkeypatch):
    # Episode 0 alone supplies more singleton candidates than the output limit.
    x = np.arange(40., dtype=np.float64)[:, None] ** 2
    y = np.zeros_like(x)
    offsets = np.array([0, 20, 25, 32, 40])
    args = dict(radius=0.00001, limit=8, neighbors=2, seed=7, batch_size=5)
    fresh = retrieve_least_visited(x, y, offsets, **args)
    assert np.all(fresh.visit_counts == 1) and np.all(fresh.frame_counts == 1)
    selected_episodes = np.searchsorted(offsets[1:], fresh.indices, side="right")
    for round_of_picks in selected_episodes.reshape(2, 4):
        assert sorted(round_of_picks) == [0, 1, 2, 3]
    assert fresh.isolation_neighbors == 2
    distances = abs(x - x.T) / x.std()
    np.fill_diagonal(distances, np.inf)
    expected = np.sort(distances, axis=1)[:, 1]
    np.testing.assert_allclose(fresh.neighbor_distances, expected[fresh.indices])
    assert fresh.indices[0] == np.argmax(expected)

    def fail_recount(*args, **kwargs):
        pytest.fail("Reranking recomputed visit counts")

    monkeypatch.setattr(visit_density, "_count_visits", fail_recount)
    reranked = retrieve_least_visited(x, y, offsets, previous_counts=(fresh.visit_counts, fresh.frame_counts), **args)
    np.testing.assert_array_equal(reranked.indices, fresh.indices)
    np.testing.assert_allclose(reranked.neighbor_distances, fresh.neighbor_distances)


def test_episode_coverage_only_breaks_exact_count_ties():
    x = np.array([[1], [5], [11], [20], [20], [30], [30], [20], [20]])
    # Three singletons only occur in episode 0. Neither other episode can
    # displace them. In the next tier, episode 1's two-frame stay goes next.
    out = retrieve_least_visited(x, x, [0, 5, 7, 9], radius=0.001, neighbors=2)
    assert set(out.indices[:3]) == {0, 1, 2}
    assert out.indices[3] in (5, 6)
    assert list(zip(out.visit_counts[out.indices], out.frame_counts[out.indices])) == [
        (1, 1), (1, 1), (1, 1), (1, 2), (2, 4),
    ]
    # Requiring more than the query frame drops the three singletons and
    # keeps the one-visit/two-frame stay ahead of the two-visit neighborhood.
    supported = retrieve_least_visited(x, x, [0, 5, 7, 9], radius=0.001, neighbors=2,
                                       min_neighborhood_frames=2)
    np.testing.assert_array_equal(supported.visit_counts, out.visit_counts)
    np.testing.assert_array_equal(supported.frame_counts, out.frame_counts)
    assert supported.indices[0] in (5, 6)
    assert supported.visit_counts[supported.indices].tolist() == [1, 2]
    empty = retrieve_least_visited(x, x, [0, 5, 7, 9], radius=0.001, min_neighborhood_frames=5)
    assert len(empty.indices) == len(empty.neighbor_distances) == 0


def test_true_ties_use_seed_without_always_favoring_the_first_frame():
    x = np.zeros((12, 1))

    def representative(seed):
        out = retrieve_least_visited(x, x, [0, 12], seed=seed)
        assert out.indices.shape == (1,)
        return out.indices[0]

    assert representative(23) == representative(23)
    assert len({representative(seed) for seed in range(8)}) > 1
