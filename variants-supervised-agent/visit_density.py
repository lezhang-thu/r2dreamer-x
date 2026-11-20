"""Retrieve separated archived states with the fewest neighborhood entries."""

from dataclasses import dataclass
import hashlib

import numpy as np
import torch


@dataclass
class VisitDensityResult:
    indices: np.ndarray
    visit_counts: np.ndarray
    frame_counts: np.ndarray
    radius: float
    neighbors: int | None
    radius_samples: int
    latent_mean: np.ndarray
    latent_scale: float
    dino_mean: np.ndarray
    dino_scale: float
    neighbor_distances: np.ndarray
    isolation_neighbors: int


def _normalize(latents, dino, device, batch_size):
    """Scale each modality to unit RMS distance from its own mean."""
    width = latents.shape[1] + dino.shape[1]
    points = torch.empty((len(latents), width), dtype=torch.float64, device=device)
    offset = 0
    stats = []
    for source in (latents, dino):
        block = points[:, offset:offset + source.shape[1]]
        for start in range(0, len(points), batch_size):
            block[start:start + batch_size].copy_(torch.as_tensor(source[start:start + batch_size]))
        variance, mean = torch.var_mean(block, dim=0, correction=0)
        scale = float(variance.sum().sqrt())
        if not np.isfinite(scale) or not torch.isfinite(mean).all():
            raise ValueError("State features must be finite")
        if scale == 0:
            scale = 1.0
        block.sub_(mean).div_(scale).add_(0.0)  # Canonicalize signed zero for row fingerprints.
        stats.extend((mean.cpu().numpy(), scale))
        offset += source.shape[1]
    return points, stats


def _estimate_radius(points, neighbors, samples, seed, batch_size):
    ids = np.random.default_rng(seed).choice(len(points), samples, replace=False)
    radii = []
    for start in range(0, samples, batch_size):
        batch = torch.as_tensor(ids[start:start + batch_size], device=points.device)
        # Coordinate differences avoid cancellation for repeated or nearly identical states.
        dist = torch.cdist(points[batch], points, compute_mode="donot_use_mm_for_euclid_dist")
        dist[torch.arange(len(batch), device=points.device), batch] = torch.inf
        radii.append(dist.kthvalue(neighbors, dim=1).values.cpu().numpy())
    return float(np.median(np.concatenate(radii)))


def _inside_radius(points, norms, indices, radius):
    queries = points[indices]
    sum_norms = norms[indices, None] + norms
    squared = torch.addmm(sum_norms, queries, points.T, alpha=-2.0)
    # The dot-product formula can lose precision at a radius boundary. Explicit
    # coordinate differences settle only those comparisons whose rounding error
    # could flip membership. This includes a calibrated neighbor exactly at r.
    radius_squared = radius * radius
    rounding = (8.0 * points.shape[1] * torch.finfo(points.dtype).eps) * sum_norms
    inside = squared <= radius_squared
    rows, cols = torch.where((squared - radius_squared).abs() <= rounding)
    for start in range(0, len(rows), 512):
        r, c = rows[start:start + 512], cols[start:start + 512]
        dist = torch.cdist(queries[r, None], points[c, None], compute_mode="donot_use_mm_for_euclid_dist")
        inside[r, c] = dist.flatten() <= radius
    inside[torch.arange(len(queries), device=points.device), indices] = True
    return inside


def _exact_match_groups(points):
    # A zero radius is a valid outcome with repeated frames. Group once to avoid
    # N² near-boundary comparisons. Row comparisons resolve hash collisions.
    points = points.cpu().numpy()
    buckets = {}
    exemplars = []
    groups = np.empty(len(points), dtype=np.int64)
    for i, row in enumerate(points):
        bucket = buckets.setdefault(hashlib.sha256(row).digest(), [])
        for group in bucket:
            if np.array_equal(row, points[exemplars[group]]):
                break
        else:
            group = len(exemplars)
            exemplars.append(i)
            bucket.append(group)
        groups[i] = group
    return groups


def _count_visits(points, norms, groups, offsets, radius, batch_size):
    if groups is not None:
        entries = np.r_[True, groups[1:] != groups[:-1]]
        entries[offsets[:-1]] = True
        visit_counts = np.bincount(groups[entries], minlength=int(groups.max()) + 1)[groups]
        frame_counts = np.bincount(groups)[groups]
        return visit_counts, frame_counts
    starts = torch.as_tensor(offsets[:-1], device=points.device)
    visits = torch.empty(len(points), device=points.device, dtype=torch.long)
    frames = torch.empty_like(visits)
    for start in range(0, len(points), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(points)), device=points.device)
        inside = _inside_radius(points, norms, indices, radius)
        frames[indices] = inside.sum(dim=1)
        entries = torch.empty_like(inside)
        entries[:, 1:] = inside[:, 1:] & ~inside[:, :-1]
        entries[:, starts] = inside[:, starts]
        visits[indices] = entries.sum(dim=1)
    return visits.cpu().numpy(), frames.cpu().numpy()


def _isolation_distances(points, norms, candidates, neighbors, batch_size, groups, frame_counts):
    """Distance to the kth other archived frame. Only the queried count tier needs scoring."""
    scores = np.zeros(len(candidates), dtype=np.float64)
    if neighbors == 0:
        return scores
    if groups is not None:
        # At radius zero, rows with k identical peers have isolation distance
        # zero. For remaining duplicates, query just one copy of each row.
        _, first, inverse = np.unique(groups[candidates], return_index=True, return_inverse=True)
        candidates = candidates[first]
        to_score = np.flatnonzero(frame_counts[candidates] <= neighbors)
        scores = np.zeros(len(first), dtype=np.float64)
    else:
        to_score = np.arange(len(candidates))
    max_norm = norms.max()
    for start in range(0, len(to_score), batch_size):
        positions = to_score[start:start + batch_size]
        ids = torch.as_tensor(candidates[positions], device=points.device)
        rows = torch.arange(len(ids), device=points.device)
        queries = points[ids]
        squared = torch.addmm(norms[ids, None] + norms, queries, points.T, alpha=-2.0)
        squared[rows, ids] = torch.inf
        kth, which = squared.kthvalue(neighbors, dim=1)
        distances = torch.linalg.vector_norm(queries - points[which], dim=1)
        # Dot-product rounding can change the kth index for nearly coincident
        # distances. Where the boundary is ambiguous, use coordinate differences.
        margin = 16.0 * points.shape[1] * torch.finfo(points.dtype).eps * (norms[ids] + max_norm)
        ambiguous = ((squared - kth[:, None]).abs() <= margin[:, None]).sum(1) > 1
        if ambiguous.any():
            dist = torch.cdist(queries[ambiguous], points, compute_mode="donot_use_mm_for_euclid_dist")
            dist[torch.arange(len(dist), device=points.device), ids[ambiguous]] = torch.inf
            distances[ambiguous] = dist.kthvalue(neighbors, dim=1).values
        scores[positions] = distances.cpu().numpy()
    return scores[inverse] if groups is not None else scores


def _select(points, norms, groups, offsets, radius, visits, frames, limit, neighbors, seed,
            batch_size, min_neighborhood_frames):
    n = len(points)
    eligible = np.flatnonzero(frames >= min_neighborhood_frames)
    order = eligible[np.lexsort((frames[eligible], visits[eligible]))]
    boundaries = np.flatnonzero((visits[order[1:]] != visits[order[:-1]])
                                | (frames[order[1:]] != frames[order[:-1]])) + 1
    episodes = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    picks_per_episode = np.zeros(len(offsets) - 1, dtype=np.int64)
    random_priority = np.random.default_rng(seed).permutation(n)
    scores = np.empty(n, dtype=np.float64)
    selected = []
    available = np.ones(n, dtype=bool)
    for tier in np.split(order, boundaries):
        tier = tier[available[tier]]
        if not len(tier):
            continue
        scores[tier] = _isolation_distances(points, norms, tier, neighbors, batch_size, groups, frames)
        candidates = tier[np.lexsort((random_priority[tier], -scores[tier]))]
        while len(candidates):
            # The first minimum preserves isolation order among equally represented episodes.
            i = candidates[np.argmin(picks_per_episode[episodes[candidates]])]
            selected.append(i)
            if len(selected) == limit:
                break
            picks_per_episode[episodes[i]] += 1
            available[i] = False
            if frames[i] > 1:
                if groups is not None:
                    available[groups == groups[i]] = False
                else:
                    query = torch.as_tensor([i], device=points.device)
                    available[_inside_radius(points, norms, query, radius)[0].cpu().numpy()] = False
            candidates = candidates[available[candidates]]
        if len(selected) == limit:
            break
    selected = np.asarray(selected, dtype=np.int64)
    return selected, scores[selected]


@torch.inference_mode()
def retrieve_least_visited(latents, dino, episode_offsets, *, limit=100, radius=None, neighbors=20,
                           radius_samples=1024, seed=0, device="cpu", batch_size=256,
                           previous_counts=None, min_neighborhood_frames=1):
    """Count entries into each state's closed Euclidean ball, then pick separated states.

    Each input row is a frame in chronological episode order. Offsets begin at
    zero and include the exclusive end of the last episode. A neighborhood can
    be entered on the first frame or on a transition from outside to inside.
    The occupancy floor limits candidacy. Every frame still contributes to the
    counts and nearest-neighbor distances. Within each tied pair of counts,
    prefer underrepresented episodes, then greater isolation, then seeded ties.
    A one-frame archive has neighbor rank and reported distance zero. Pass
    verified (visit_counts, frame_counts) and their original radius to rerank a saved run.
    """
    if latents.ndim != 2 or dino.ndim != 2 or min(latents.shape[1], dino.shape[1]) < 1:
        raise ValueError("Expected two nonempty-width feature matrices")
    n = len(latents)
    if n < 1 or len(dino) != n:
        raise ValueError("Need equally many nonempty latent and DINO rows")
    offsets = np.asarray(episode_offsets)
    if (offsets.ndim != 1 or len(offsets) < 2 or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0 or offsets[-1] != n or np.any(offsets[1:] <= offsets[:-1])):
        raise ValueError("Episode offsets must strictly increase from zero to the frame count")
    offsets = offsets.astype(np.int64, copy=False)
    if min(limit, neighbors, radius_samples, batch_size, min_neighborhood_frames) < 1:
        raise ValueError("limit, neighbors, radius_samples, batch_size and min_neighborhood_frames must be positive")
    if radius is not None and (not np.isfinite(radius) or radius < 0):
        raise ValueError("Radius must be finite and nonnegative")
    isolation_neighbors = min(neighbors, n - 1)
    if previous_counts is not None:
        if radius is None:
            raise ValueError("Reusing counts requires their original radius")
        visits, frames = map(np.asarray, previous_counts)
        if (any(a.shape != (n,) or not np.issubdtype(a.dtype, np.integer) for a in (visits, frames))
                or np.any(visits < 1) or np.any(frames < visits) or np.any(frames > n)):
            raise ValueError("Expected one valid visit and frame count per archive row")
    points, stats = _normalize(latents, dino, device, batch_size)
    if radius is None:
        neighbors = min(neighbors, n - 1)
        radius_samples = min(radius_samples, n)
        radius = _estimate_radius(points, neighbors, radius_samples, seed, batch_size) if neighbors else 0.0
    else:
        neighbors, radius_samples = None, 0
    norms = torch.linalg.vector_norm(points, dim=1).square_()
    groups = _exact_match_groups(points) if radius == 0 else None
    if previous_counts is None:
        visits, frames = _count_visits(points, norms, groups, offsets, radius, batch_size)
    selected, distances = _select(points, norms, groups, offsets, radius, visits, frames,
                                  limit, isolation_neighbors, seed, batch_size, min_neighborhood_frames)
    return VisitDensityResult(selected, visits, frames, float(radius), neighbors, radius_samples, *stats,
                              distances, isolation_neighbors)
