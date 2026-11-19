"""Extract one frozen DINOv3 ViT-L/16 CLS vector per archived Atari frame."""

import argparse
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import functional as F

from atari_data import file_sha256, frames_to_tensor, load_archive, state_digest


DEFAULT_BACKBONE = "facebook/dinov3-vitl16-pretrain-lvd1689m"
FEATURE_DIM = 1024
PREPROCESSING = "symmetric_black_pad16_imagenet"


def load_timm_checkpoint(path):
    import timm

    weights = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    model = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=0, global_pool="token")
    model.load_state_dict(weights, strict=True)
    return model


def load_dino_backbone(source, revision=None):
    from transformers import AutoModel

    return AutoModel.from_pretrained(source, revision=revision)


class DinoFrameEncoder(nn.Module):
    """Frozen ViT-L/16 with the same patch-aligned preprocessing for offline and live RGB."""

    def __init__(self, backbone=DEFAULT_BACKBONE, revision=None):
        super().__init__()
        path = Path(backbone).expanduser()
        self.timm_checkpoint = path.suffix.lower() == ".pth" and not path.is_dir()
        if self.timm_checkpoint:
            if not path.is_file():
                raise FileNotFoundError(path)
            self.backbone = str(path.resolve())
            self.model = load_timm_checkpoint(self.backbone)
            self.revision = None
        else:
            self.backbone = str(path.resolve()) if path.is_dir() else str(backbone)
            self.model = load_dino_backbone(self.backbone, revision=revision)
            config = self.model.config
            if (config.model_type, config.patch_size, config.hidden_size, config.num_hidden_layers) != (
                    "dinov3_vit", 16, FEATURE_DIM, 24):
                raise ValueError("--backbone must be a pretrained DINOv3 ViT-L/16 model")
            self.revision = getattr(config, "_commit_hash", None) or revision
        self.backbone_sha256 = state_digest(hashlib.sha256(), self.model.state_dict()).hexdigest()
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1), persistent=False)
        self.requires_grad_(False).eval()

    def provenance(self, archive_sha256):
        return dict(archive_sha256=archive_sha256, backbone=self.backbone, revision=self.revision,
                    backbone_sha256=self.backbone_sha256, preprocessing=PREPROCESSING)

    @torch.no_grad()
    def forward(self, images):
        height_pad, width_pad = -images.shape[-2] % 16, -images.shape[-1] % 16
        padded = F.pad(images, (width_pad // 2, width_pad - width_pad // 2,
                                height_pad // 2, height_pad - height_pad // 2))
        pixels = (padded - self.rgb_mean) / self.rgb_std
        use_bfloat16 = pixels.is_cuda and torch.cuda.get_device_capability(pixels.device)[0] >= 8
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            cls = self.model(pixels) if self.timm_checkpoint else self.model(pixel_values=pixels).pooler_output
        return cls.float()


@torch.no_grad()
def extract_episodes(episodes, encoder, device, batch_size=16):
    """Preserve trajectory and frame order, including resets and final cutoff screens."""
    if batch_size < 1:
        raise ValueError("--feature-batch-size must be positive")
    encoder.eval()
    records = []
    for episode in episodes:
        frames = episode["frames"]
        features = torch.empty((len(frames), FEATURE_DIM), dtype=torch.float32, device="cpu")
        for start in range(0, len(frames), batch_size):
            stop = min(start + batch_size, len(frames))
            images = frames_to_tensor(frames[start:stop], device)
            vectors = encoder(images).cpu()
            if vectors.shape != (stop - start, FEATURE_DIM) or not torch.isfinite(vectors).all():
                raise ValueError(f"Invalid DINOv3 output in episode {episode['id']} at frame {start}")
            features[start:stop].copy_(vectors)
        print(f"Episode {episode['id']}: extracted {len(features)} frame embeddings", flush=True)
        records.append(dict(id=episode["id"], complete=bool(episode.get("complete", True)),
                            collection_cutoff=bool(episode.get("collection_cutoff", False)), features=features))
    return records


def load_features(path):
    """Memory-map a feature file as CPU tensors, without opening RGB data or loading a ViT."""
    data = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    if (not isinstance(data, dict) or data.get("kind") != "dinov3_atari_features"
            or data.get("format_version") != 1):
        raise ValueError("Expected a feature file created by extract_dino_features.py")
    episodes, metadata, provenance = data["episodes"], data["metadata"], data["provenance"]
    if (not isinstance(episodes, list) or not episodes or not isinstance(metadata, dict)
            or not isinstance(provenance, dict) or provenance.get("preprocessing") != PREPROCESSING
            or not {"archive_sha256", "backbone", "revision", "backbone_sha256"} <= provenance.keys()):
        raise ValueError("Invalid feature metadata")
    ids = set()
    for episode in episodes:
        features = episode["features"]
        if (type(episode["id"]) is not int or episode["id"] in ids
                or not isinstance(features, torch.Tensor) or features.dtype != torch.float32
                or features.ndim != 2 or features.shape[1] != FEATURE_DIM or len(features) < 2):
            raise ValueError("Each episode needs a unique integer ID and float32 features of shape [T+1, 1024]")
        ids.add(episode["id"])
    return episodes, metadata, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Source episodes.pkl")
    parser.add_argument("--output", type=Path, required=True, help="New embeddings.pt file")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, help="Hugging Face ID, local snapshot, or timm ViT-L/16 .pth file")
    parser.add_argument("--feature-batch-size", type=int, default=16, help="Frames per frozen DINO forward pass")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh feature output path: {args.output}")
    if args.feature_batch_size < 1:
        raise ValueError("--feature-batch-size must be positive")

    episodes, archive_metadata = load_archive(args.archive)
    encoder = DinoFrameEncoder(args.backbone).to(args.device)
    provenance = encoder.provenance(file_sha256(args.archive))
    metadata = {k: [int(ep_id) for ep_id in archive_metadata[k]] for k in ("training_episode_ids", "evaluation_episode_ids")
                if k in archive_metadata}
    records = extract_episodes(episodes, encoder, args.device, args.feature_batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp) / "embeddings.pt"
        torch.save(dict(kind="dinov3_atari_features", format_version=1, episodes=records,
                        metadata=metadata, provenance=provenance), staged)
        # Publish the complete file atomically, even if another extraction uses the same output path.
        os.link(staged, args.output)
    print(f"Saved {sum(len(ep['features']) for ep in records)} frame embeddings "
          f"from {len(records)} trajectories to {args.output}", flush=True)


if __name__ == "__main__":
    main()
