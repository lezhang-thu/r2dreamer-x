"""Read and validate per-frame counts from a completed least-visited export."""

import json
import math
from pathlib import Path

import numpy as np


DISTANCE = "Euclidean on concatenated blocks, each centered and divided by its RMS radius"
VISIT_DEFINITION = "One contiguous run of in-radius frames within one episode"


def load_saved_counts(directory, offsets, episode_ids, dino_provenance, dreamer_provenance):
    """Return the source manifest and per-frame visit/occupancy arrays in archive order."""
    directory = Path(directory)
    manifest_path, arrays_path = directory / "manifest.json", directory / "frame_counts.npz"
    for path in (manifest_path, arrays_path):
        if not path.is_file():
            raise FileNotFoundError(f"Reusing counts requires the original export directory with both "
                                    f"manifest.json and frame_counts.npz. Missing {path}")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a least-visited manifest in {manifest_path}")
    n, episode_count = int(offsets[-1]), len(episode_ids)
    expected = dict(distance=DISTANCE, visit_definition=VISIT_DEFINITION, frame_count=n,
                    episode_count=episode_count, action_count=n - episode_count,
                    archive_sha256=dino_provenance["archive_sha256"])
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Saved count manifest has a different {key}: {manifest_path}")
    for key, current, fields in (
        ("dino_provenance", dino_provenance, ("archive_sha256", "backbone_sha256", "preprocessing")),
        ("dreamer_provenance", dreamer_provenance, ("archive_sha256", "checkpoint_sha256",
                                                   "config_sha256", "representation")),
    ):
        saved = manifest.get(key)
        for field in fields:
            if not isinstance(saved, dict) or not current.get(field) or saved.get(field) != current[field]:
                raise ValueError(f"Saved counts were computed with a different {key}.{field}: {manifest_path}")

    radius = manifest.get("radius")
    if type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0:
        raise ValueError(f"Invalid saved neighborhood radius in {manifest_path}")
    method = manifest.get("radius_method")
    rank, samples = manifest.get("effective_neighbors"), manifest.get("radius_sample_size")
    if method == "fixed":
        valid_calibration = rank is None and samples == 0
    elif method == "median_kth_neighbor":
        valid_calibration = (type(rank) is int and (rank == 0 if n == 1 else 1 <= rank < n)
                             and type(samples) is int and 1 <= samples <= n)
    else:
        valid_calibration = False
    if not valid_calibration:
        raise ValueError(f"Unrecognized saved radius calibration in {manifest_path}")

    with np.load(arrays_path, allow_pickle=False) as arrays:
        required = {"episode_offsets", "episode_ids", "visit_counts", "neighborhood_frame_counts", "selected_indices"}
        if not required.issubset(arrays.files):
            raise ValueError(f"Missing count or episode arrays in {arrays_path}")
        if (not np.array_equal(arrays["episode_offsets"], offsets)
                or not np.array_equal(arrays["episode_ids"], episode_ids)):
            raise ValueError(f"The saved counts have a different episode order or episode lengths: {arrays_path}")
        visits, frames = arrays["visit_counts"], arrays["neighborhood_frame_counts"]
        selected = arrays["selected_indices"]
    if (any(a.shape != (n,) or a.dtype.kind not in "iu" for a in (visits, frames))
            or np.any(visits < 1) or np.any(visits > frames) or np.any(frames > n)):
        raise ValueError(f"Invalid per-frame neighborhood counts in {arrays_path}")
    # Existing frame_counts.npz files have no manifest fingerprint. Cross-check
    # their counts and representative indices before using the manifest's radius.
    rows = manifest.get("representatives", [])
    if (manifest.get("frames_with_one_visit") != np.count_nonzero(visits == 1)
            or manifest.get("frames_with_only_self_in_radius") != np.count_nonzero(frames == 1)
            or not isinstance(rows, list) or len(rows) != manifest.get("representative_count")
            or selected.shape != (len(rows),) or selected.dtype.kind not in "iu"
            or np.any(selected < 0) or np.any(selected >= n)):
        raise ValueError(f"Saved count arrays do not match the manifest in {directory}")
    for i, row in zip(selected, rows):
        if (not isinstance(row, dict) or (int(i), int(visits[i]), int(frames[i]))
                != (row.get("archive_frame_index"), row.get("visit_count"), row.get("neighborhood_frame_count"))):
            raise ValueError(f"Saved representative counts do not match the manifest in {directory}")
    return manifest, (visits, frames)
