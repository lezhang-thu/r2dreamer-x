"""Export the native RGB frame of each return-exploration anchor representative."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from PIL import Image
import torch

from extract_dino_features import load_features
from train_gru_classifier import encode_feature_trajectory, encode_trajectory, file_sha256
from train_supervised_agent import extract_anchor_indices, load_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from return_anchors import AnchorBank, classifier_similarity


def load_matching_features(path, episodes, checkpoint):
    if checkpoint.get("features_sha256") and file_sha256(path) != checkpoint["features_sha256"]:
        raise ValueError("Feature file differs from the one used to train this classifier")
    records, _, provenance = load_features(path)
    if (provenance["archive_sha256"] != checkpoint["archive_sha256"]
            or provenance["backbone_sha256"] != checkpoint["backbone_sha256"]):
        raise ValueError("Feature file belongs to a different archive or DINOv3 backbone")
    by_id = {record["id"]: record["features"] for record in records}
    if (set(by_id) != {ep["id"] for ep in episodes}
            or any(len(by_id[ep["id"]]) != len(ep["frames"]) for ep in episodes)):
        raise ValueError("Feature sequences do not align with the source episodes")
    return by_id


@torch.no_grad()
def export_anchor_set(archive, classifier_path, output, device, feature_batch_size, complete_only, features_path=None):
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    if feature_batch_size < 1:
        raise ValueError("--feature-batch-size must be positive")

    classifier, checkpoint, episodes, _ = load_inputs(
        archive, classifier_path, device, load_encoder=features_path is None)
    features = load_matching_features(features_path, episodes, checkpoint) if features_path is not None else None
    if complete_only:
        episodes = [ep for ep in episodes if ep["complete"]]
    if any(ep["frames"].shape[1:] != (210, 160, 3) for ep in episodes):
        raise ValueError("Export requires the native 210x160 RGB trajectory archive")
    threshold = checkpoint["threshold"]
    sources = {ep["id"]: ep for ep in episodes}
    candidates = []
    # Keep only small context vectors and indices. The original pixels stay in the archive.
    for ep in episodes:
        if features is None:
            contexts = encode_trajectory(classifier, ep["frames"], device, batch_size=feature_batch_size)
        else:
            contexts = encode_feature_trajectory(classifier, features[ep["id"]], device)
        contexts = contexts.to(device)
        indices = extract_anchor_indices(classifier, contexts, threshold)
        candidates.append(dict(id=ep["id"], references=contexts[indices].cpu().numpy(), indices=indices))
        print(f"Episode {ep['id']}: {len(indices)} anchors in {len(contexts)} frames", flush=True)

    _, counts, origins = AnchorBank.select_representatives(
        candidates, classifier_similarity(classifier, device), threshold)
    positions = {candidate["id"]: candidate["indices"] for candidate in candidates}
    manifest = dict(
        archive_sha256=checkpoint["archive_sha256"], classifier_fingerprint=checkpoint["model_fingerprint"],
        threshold=threshold, complete_only=complete_only, episode_count=len(episodes),
        candidate_count=sum(map(len, positions.values())), representative_count=len(origins),
        representatives=[],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        directory = Path(tmp)
        for index, ((episode_id, position), count) in enumerate(zip(origins, counts)):
            frame_index = positions[episode_id][position]
            name = f"anchor_{index:06d}_ep_{episode_id}_frame_{frame_index:06d}.png"
            Image.fromarray(sources[episode_id]["frames"][frame_index]).save(directory / name)
            manifest["representatives"].append(dict(
                id=index, file=name, episode_id=episode_id, anchor_position=position,
                frame_index=frame_index, count=int(count),
            ))
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        directory.rename(output)
    print(f"Saved {len(origins)} representatives from {manifest['candidate_count']} anchors to {output}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Native RGB episodes.pkl")
    parser.add_argument("--features", type=Path, help="Reuse the saved DINOv3 embeddings for anchor selection")
    parser.add_argument("--classifier", required=True, type=Path, help="GRU checkpoint fitted on this archive")
    parser.add_argument("--output", required=True, type=Path, help="New directory for PNGs and manifest.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-batch-size", type=int, default=16, help="Frames per encoder forward pass")
    parser.add_argument("--complete-only", action="store_true", help="Use the online cycle's completed-episode rule")
    args = parser.parse_args(argv)
    return export_anchor_set(args.archive, args.classifier, args.output, args.device,
                             args.feature_batch_size, args.complete_only, args.features)


if __name__ == "__main__":
    main()
