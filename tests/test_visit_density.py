"""Neighborhood counts and retrieved source frames agree with a direct oracle."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "variants-supervised-agent"))
from visit_density import retrieve_least_visited


def test_runs_count_reentry_and_episode_start_at_equal_frames():
    # Repeated 1 across episode boundaries must start a new visit. The last
    # trajectory ends inside 1's neighborhood; the visit counts at cutoff.
    x = np.array([[1], [1], [0], [1], [1], [1], [3], [1]], np.float32)
    y = torch.ones((len(x), 2))
    result = retrieve_least_visited(x, y, [0, 4, 8], radius=0, limit=100, batch_size=2)
    np.testing.assert_array_equal(result.frame_counts, [6, 6, 1, 6, 6, 6, 1, 6])
    np.testing.assert_array_equal(result.visit_counts, [4, 4, 1, 4, 4, 4, 1, 4])
    np.testing.assert_array_equal(result.indices, [2, 6, 0])
    assert result.neighbors is None and result.radius_samples == 0
    assert result.dino_scale == 1
    positive = retrieve_least_visited(x, y, [0, 4, 8], radius=0.01, batch_size=3)
    for key in ("indices", "frame_counts", "visit_counts"):
        np.testing.assert_array_equal(getattr(result, key), getattr(positive, key))


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
    for radius, counts, selected in [(2.0, [2, 2], [0]), (np.nextafter(2.0, 0), [1, 1], [0, 1])]:
        result = retrieve_least_visited(x, y, [0, 2], radius=radius, batch_size=1)
        np.testing.assert_array_equal(result.frame_counts, counts)
        np.testing.assert_array_equal(result.indices, selected)


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
    inside = np.linalg.norm(points[:, None] - points, axis=-1) <= result.radius
    visits = np.zeros(len(x), dtype='int64')
    for begin, end in zip(offsets[:-1], offsets[1:]):
        for q, row in enumerate(inside[:, begin:end]):
            visits[q] += sum(member and (t == 0 or not row[t-1]) for t, member in enumerate(row))
    np.testing.assert_array_equal(result.visit_counts, visits)
    np.testing.assert_array_equal(result.frame_counts, inside.sum(axis=1))
    selected = []
    for i in sorted(range(len(x)), key=lambda j: (visits[j], inside[j].sum(), j)):
        if all(not inside[i, j] for j in selected):
            selected.append(i)
        if len(selected) == 5:
            break
    np.testing.assert_array_equal(result.indices, selected)


def test_degenerate_radius_and_small_archives():
    x = np.ones((30, 8), dtype=np.float32)
    result = retrieve_least_visited(x, x, [0, 2, 26, 30], neighbors=20, radius_samples=4)
    assert result.radius == 0 and result.indices.tolist() == [0]
    assert np.all(result.visit_counts == 3) and np.all(result.frame_counts == 30)
    singleton = retrieve_least_visited(x[:1], x[:1], [0, 1])
    assert singleton.radius == 0 and singleton.visit_counts.tolist() == [1]
    with pytest.raises(ValueError, match="offsets"):
        retrieve_least_visited(x, x, [0, 30, 30])
    with pytest.raises(ValueError, match="Radius"):
        retrieve_least_visited(x, x, [0, 30], radius=np.nan)
