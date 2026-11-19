"""Check feature-file alignment, shape and provenance without pretrained model downloads."""
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "variants-supervised-agent"))
import extract_dino_features as extractor
from atari_data import file_sha256


def test_feature_file_matches_all_source_frames_and_splits(tmp_path, stub_dino):
    archive = tmp_path / "episodes.pkl"
    output = tmp_path / "embeddings.pt"
    source = []
    for ep_id, shades, complete in [(17, [0, 4, 16], True), (3, [200, 255], False)]:
        frames = np.stack([np.full((210, 160, 3), shade, np.uint8) for shade in shades])
        source.append(dict(id=ep_id, frames=frames, complete=complete, collection_cutoff=not complete))
    split = dict(training_episode_ids=[17], evaluation_episode_ids=[3])
    with archive.open("wb") as stream:
        pickle.dump(dict(episodes=source, metadata={k: np.asarray(ids) for k, ids in split.items()}), stream)
    args = ["--archive", str(archive), "--output", str(output), "--device", "cpu", "--feature-batch-size", "2"]
    before = file_sha256(archive)
    extractor.main(args)
    records, metadata, provenance = extractor.load_features(output)
    assert metadata == split
    assert provenance["archive_sha256"] == before == file_sha256(archive)
    assert provenance["backbone"] == extractor.DEFAULT_BACKBONE and provenance["revision"] == "test-revision"
    assert len(provenance["backbone_sha256"]) == 64
    assert [r["id"] for r in records] == [17, 3]
    assert [r["complete"] for r in records] == [True, False]
    assert [r["collection_cutoff"] for r in records] == [False, True]
    assert [tuple(batch.shape) for batch in stub_dino] == [(2, 3, 224, 160), (1, 3, 224, 160), (2, 3, 224, 160)]
    expected = torch.cat(stub_dino).mean(dim=(1, 2, 3))[:, None].expand(-1, 1024)
    torch.testing.assert_close(torch.cat([ep["features"] for ep in records]), expected)
    assert all(ep["features"].dtype == torch.float32 and "frames" not in ep for ep in records)
    with pytest.raises(FileExistsError):
        extractor.main(args)
    assert len(stub_dino) == 3


def test_loader_rejects_wrong_width_and_duplicate_ids(tmp_path):
    path = tmp_path / "bad.pt"
    data = dict(kind="dinov3_atari_features", format_version=1, metadata={},
                provenance=dict(preprocessing=extractor.PREPROCESSING, revision=None,
                                backbone="dino", backbone_sha256="a" * 64, archive_sha256="b" * 64))
    for records in [[dict(id=9, features=torch.ones((3, 1000)))],
                    [dict(id=9, features=torch.ones((3, 1024)))] * 2]:
        torch.save(dict(data, episodes=records), path)
        with pytest.raises(ValueError, match="unique integer ID"):
            extractor.load_features(path)
