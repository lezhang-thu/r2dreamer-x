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


def _retrieve(points, offsets, radius, limit, batch_size):
    n = len(points)
    if radius == 0:
        groups = _exact_match_groups(points)
        entries = np.r_[True, groups[1:] != groups[:-1]]
        entries[offsets[:-1]] = True
        visit_counts = np.bincount(groups[entries], minlength=int(groups.max()) + 1)[groups]
        frame_counts = np.bincount(groups)[groups]
    else:
        norms = torch.linalg.vector_norm(points, dim=1).square_()
        starts = torch.as_tensor(offsets[:-1], device=points.device)
        visits = torch.empty(n, device=points.device, dtype=torch.long)
        frames = torch.empty_like(visits)
        for start in range(0, n, batch_size):
            indices = torch.arange(start, min(start + batch_size, n), device=points.device)
            inside = _inside_radius(points, norms, indices, radius)
            frames[indices] = inside.sum(dim=1)
            entries = torch.empty_like(inside)
            entries[:, 1:] = inside[:, 1:] & ~inside[:, :-1]
            entries[:, starts] = inside[:, starts]
            visits[indices] = entries.sum(dim=1)
        visit_counts, frame_counts = visits.cpu().numpy(), frames.cpu().numpy()

    # Rank against the full archive before suppressing any candidate. Stable ties
    # prefer lower frame occupancy, then the earlier original frame index.
    order = np.lexsort((np.arange(n), frame_counts, visit_counts))
    selected = []
    available = np.ones(n, dtype=bool)
    for i in order:
        if not available[i]:
            continue
        selected.append(i)
        if len(selected) == limit:
            break
        if radius == 0:
            available[groups == groups[i]] = False
        else:
            query = torch.tensor([i], device=points.device)
            available[_inside_radius(points, norms, query, radius)[0].cpu().numpy()] = False
    return np.asarray(selected, dtype=np.int64), visit_counts, frame_counts


@torch.inference_mode()
def retrieve_least_visited(latents, dino, episode_offsets, *, limit=100, radius=None, neighbors=20,
                           radius_samples=1024, seed=0, device="cpu", batch_size=256):
    """Count entries into each state's closed Euclidean ball, then pick separated states.

    Each input row is a frame in chronological episode order. Offsets begin at
    zero and include the exclusive end of the last episode. A neighborhood can
    be entered on the first frame or on a transition from outside to inside.
    Returned count arrays include every input frame in the original order.
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
    if min(limit, neighbors, radius_samples, batch_size) < 1:
        raise ValueError("limit, neighbors, radius_samples and batch_size must be positive")
    if radius is not None and (not np.isfinite(radius) or radius < 0):
        raise ValueError("Radius must be finite and nonnegative")
    points, stats = _normalize(latents, dino, device, batch_size)
    if radius is None:
        neighbors = min(neighbors, n - 1)
        radius_samples = min(radius_samples, n)
        radius = _estimate_radius(points, neighbors, radius_samples, seed, batch_size) if neighbors else 0.0
    else:
        neighbors, radius_samples = None, 0
    selected, visits, frames = _retrieve(points, offsets, radius, limit, batch_size)
    return VisitDensityResult(selected, visits, frames, float(radius), neighbors, radius_samples, *stats)
