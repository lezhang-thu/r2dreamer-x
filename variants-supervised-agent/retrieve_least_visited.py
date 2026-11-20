"""Export native Atari frames whose Dreamer–DINOv3 neighborhoods were visited least often."""

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import torch

from atari_data import file_sha256, load_archive
from dreamer_latent_cache import load_or_replay_latents
from extract_dino_features import load_features
from saved_visit_counts import DISTANCE, VISIT_DEFINITION, load_saved_counts
from visit_density import retrieve_least_visited


def aligned_dino(episodes, archive_hash, path):
    records, _, provenance = load_features(path)
    if provenance["archive_sha256"] != archive_hash:
        raise ValueError("The DINOv3 embeddings belong to a different episodes.pkl")
    by_id = {record["id"]: record for record in records}
    if set(by_id) != {ep["id"] for ep in episodes}:
        raise ValueError("Episode IDs differ between the archive and the DINOv3 embeddings")
    vectors = []
    for ep in episodes:
        record = by_id[ep["id"]]
        if (len(record["features"]) != len(ep["frames"])
                or any(bool(record.get(key)) != bool(ep.get(key)) for key in ("complete", "collection_cutoff"))):
            raise ValueError(f"DINOv3 features do not align with episode {ep['id']}")
        vectors.append(record["features"])
    return torch.cat(vectors), provenance


def save_results(directory, episodes, offsets, result, manifest):
    rows = []
    for rank, index in enumerate(result.indices, start=1):
        ep_index = int(np.searchsorted(offsets, index, side="right") - 1)
        ep = episodes[ep_index]
        t = int(index - offsets[ep_index])
        visits = int(result.visit_counts[index])
        frames = int(result.frame_counts[index])
        filename = f"rank_{rank:03d}_visits_{visits}_frames_{frames}_ep_{ep['id']}_t_{t:06d}.png"
        Image.fromarray(ep["frames"][t]).save(directory / filename)
        rows.append(dict(rank=rank, visit_count=visits, neighborhood_frame_count=frames,
                         kth_neighbor_distance=float(result.neighbor_distances[rank - 1]),
                         episode_id=ep["id"], frame_index=t, archive_frame_index=int(index), file=filename))
    np.savez(directory / "frame_counts.npz", visit_counts=result.visit_counts,
             neighborhood_frame_counts=result.frame_counts, selected_indices=result.indices,
             episode_ids=np.asarray([ep["id"] for ep in episodes]), episode_offsets=offsets,
             latent_mean=result.latent_mean, latent_scale=result.latent_scale,
             dino_mean=result.dino_mean, dino_scale=result.dino_scale)
    eligible = result.frame_counts >= manifest["min_neighborhood_frames"]
    visit_counts, frequencies = np.unique(result.visit_counts[eligible], return_counts=True)
    manifest.update(
        radius=float(result.radius), isolation_neighbors=result.isolation_neighbors, representative_count=len(rows),
        eligible_query_frames=int(eligible.sum()),
        eligible_episode_count=sum(bool(eligible[a:b].any()) for a, b in zip(offsets[:-1], offsets[1:])),
        selected_episode_count=len({row["episode_id"] for row in rows}),
        eligible_visit_count_histogram={int(v): int(n) for v, n in zip(visit_counts, frequencies)},
        frames_with_one_visit=int(np.count_nonzero(result.visit_counts == 1)),
        frames_with_only_self_in_radius=int(np.count_nonzero(result.frame_counts == 1)),
        representatives=rows,
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=Path("logdir/montezuma_100k_0"),
                        help="Collection directory with episodes.pkl, embeddings.pt, latest.pt and config.yaml")
    parser.add_argument("--archive", type=Path, help="Override episodes.pkl")
    parser.add_argument("--features", type=Path, help="Override the existing DINOv3 embeddings.pt")
    parser.add_argument("--checkpoint", type=Path, help="Override latest.pt")
    parser.add_argument("--config", type=Path, help="Override the collected run's config.yaml")
    parser.add_argument("--latent-cache", type=Path, help="Reusable cache directory (default RUN_DIR/dreamer_latents)")
    parser.add_argument("--output", type=Path, help="Fresh output directory")
    parser.add_argument("--reuse-counts", type=Path, help="Rerank using a previous export directory and its radius")
    parser.add_argument("--min-neighborhood-frames", type=int, default=1,
                        help="Require at least this many archived frames, including the query, in each exported neighborhood")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of distinct frames to export")
    parser.add_argument("--radius", type=float, help="Shared Euclidean radius in normalized joint feature space")
    parser.add_argument("--neighbors", type=int, default=20, help="Neighbor rank for isolation and for new automatic radius estimates")
    parser.add_argument("--radius-samples", type=int, default=1024, help="Number of calibration queries")
    parser.add_argument("--seed", type=int, default=0, help="Seed for radius sampling and breaking residual ties")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256, help="Distance queries per batch")
    parser.add_argument("--replay-batch-size", type=int, default=64, help="Observation encoder batch size during replay")
    args = parser.parse_args(argv)
    if min(args.min_neighborhood_frames, args.limit, args.neighbors, args.radius_samples, args.batch_size, args.replay_batch_size) < 1:
        parser.error("Minimum neighborhood size, limit, neighbor rank, sample size and batch sizes must be positive")
    if args.reuse_counts and args.radius is not None:
        parser.error("--reuse-counts uses the original radius. To change --radius, run a fresh count.")
    if args.seed < 0 or (args.radius is not None and (not math.isfinite(args.radius) or args.radius < 0)):
        parser.error("Seed and radius must be nonnegative, and radius must be finite")

    archive = args.archive or args.run_dir / "episodes.pkl"
    features = args.features or args.run_dir / "embeddings.pt"
    checkpoint = args.checkpoint or args.run_dir / "latest.pt"
    config = args.config or args.run_dir / "config.yaml"
    cache = args.latent_cache or args.run_dir / "dreamer_latents"
    output = args.output or args.run_dir / ("least_visited_reranked" if args.reuse_counts else "least_visited")
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    if cache.resolve().is_relative_to(output.resolve()):
        parser.error("The latent cache must be outside the output directory")
    for path in (archive, features, checkpoint, config):
        if not path.is_file():
            raise FileNotFoundError(path)
    episodes, metadata = load_archive(archive)
    if any(ep["frames"].shape[1:] != (210, 160, 3) for ep in episodes):
        raise ValueError("Use the native 210x160 RGB trajectory archive")
    offsets = np.cumsum([0] + [len(ep["frames"]) for ep in episodes])
    print(f"Checking cached DINOv3 features for {offsets[-1]} frames", flush=True)
    archive_hash = file_sha256(archive)
    dino, dino_provenance = aligned_dino(episodes, archive_hash, features)
    latents, dreamer_provenance = load_or_replay_latents(
        cache, episodes, metadata, archive_hash, checkpoint, config, device=args.device,
        batch_size=args.replay_batch_size,
    )
    radius = args.radius
    previous_counts = None
    if args.reuse_counts:
        previous, previous_counts = load_saved_counts(
            args.reuse_counts, offsets, [ep["id"] for ep in episodes], dino_provenance, dreamer_provenance)
        radius = previous["radius"]
        print(f"Reranking saved counts from {args.reuse_counts} at radius {radius:.6g}", flush=True)
    else:
        print(f"Counting neighborhood visits across {len(episodes)} trajectories", flush=True)
    result = retrieve_least_visited(
        latents, dino, offsets, limit=args.limit, radius=radius, neighbors=args.neighbors,
        radius_samples=args.radius_samples, seed=args.seed, device=args.device, batch_size=args.batch_size,
        previous_counts=previous_counts, min_neighborhood_frames=args.min_neighborhood_frames,
    )
    manifest = dict(
        frame_count=int(offsets[-1]), action_count=int(offsets[-1] - len(episodes)), episode_count=len(episodes),
        limit=args.limit, radius_method="fixed" if args.radius is not None else "median_kth_neighbor",
        effective_neighbors=result.neighbors, radius_sample_size=result.radius_samples,
        sampling_seed=args.seed, selection_seed=args.seed, min_neighborhood_frames=args.min_neighborhood_frames,
        archive_sha256=archive_hash, dreamer_provenance=dreamer_provenance,
        dino_provenance=dino_provenance,
        distance=DISTANCE, visit_definition=VISIT_DEFINITION,
        ranking=["visit_count", "neighborhood_frame_count", "already_selected_from_episode",
                 "descending_kth_neighbor_distance", "seeded_tie_break"],
    )
    if args.reuse_counts:
        for field in ("radius_method", "effective_neighbors", "radius_sample_size", "sampling_seed"):
            manifest[field] = previous[field]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        directory = Path(tmp)
        save_results(directory, episodes, offsets, result, manifest)
        directory.rename(output)
    print(f"Saved {len(result.indices)} native PNGs from {manifest['selected_episode_count']} episodes "
          f"to {output} (radius {result.radius:.6g}). "
          f"{manifest['eligible_query_frames']}/{offsets[-1]} frames met the minimum neighborhood size.", flush=True)
    return manifest


if __name__ == "__main__":
    main()
