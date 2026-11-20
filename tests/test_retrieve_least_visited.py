"""The export must preserve native RGB, archive coordinates, and the fixed visit counts."""

import json
from pathlib import Path
import pickle
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "variants-supervised-agent"))
from atari_data import file_sha256
from extract_dino_features import PREPROCESSING
import retrieve_least_visited as export


def test_counts_and_pngs_match_source_with_reordered_feature_records(tmp_path, monkeypatch):
    episodes, features = [], []
    # The first neighborhood occurs in two stretches of ep 14 and continues at the reset of ep 6.
    for ep_id, states, complete in [(14, [1, 1, 2, 1], True), (6, [1, 3], False)]:
        frames = np.stack([np.full((210, 160, 3), ep_id + t, dtype=np.uint8) for t in range(len(states))])
        ep = dict(id=ep_id, complete=complete, collection_cutoff=not complete)
        episodes.append(dict(ep, frames=frames))
        features.append(dict(ep, features=torch.tensor(states, dtype=torch.float32)[:, None].repeat(1, 1024)))
    archive = tmp_path / "episodes.pkl"
    with archive.open("wb") as stream:
        pickle.dump(episodes, stream)
    provenance = dict(archive_sha256=file_sha256(archive), preprocessing=PREPROCESSING,
                      backbone="fixture", revision=None, backbone_sha256="fixture")
    torch.save(dict(kind="dinov3_atari_features", format_version=1, metadata={}, provenance=provenance,
                    episodes=features[::-1]), tmp_path / "embeddings.pt")
    for name in ("latest.pt", "config.yaml"):
        (tmp_path / name).touch()
    monkeypatch.setattr(export, "load_or_replay_latents", lambda *a, **kw: (torch.zeros(6, 1), {}))
    args = ["--run-dir", str(tmp_path), "--device", "cpu", "--radius", "0.01"]
    export.main(args)
    directory = tmp_path / "least_visited"
    with np.load(directory / "frame_counts.npz") as data:
        np.testing.assert_array_equal(data["visit_counts"], [3, 3, 1, 3, 3, 1])
        np.testing.assert_array_equal(data["neighborhood_frame_counts"], [4, 4, 1, 4, 4, 1])
        np.testing.assert_array_equal(data["selected_indices"], [2, 5, 0])
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["representative_count"] == 3
    assert manifest["frame_count"] == 6 and manifest["frames_with_one_visit"] == 2
    for row in manifest["representatives"]:
        ep = next(ep for ep in episodes if ep["id"] == row["episode_id"])
        with Image.open(directory / row["file"]) as image:
            assert image.format == "PNG" and image.mode == "RGB" and image.size == (160, 210)
            np.testing.assert_array_equal(image, ep["frames"][row["frame_index"]])
    with pytest.raises(FileExistsError):
        export.main(args)
