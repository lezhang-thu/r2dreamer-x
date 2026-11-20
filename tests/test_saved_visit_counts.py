"""Guard against applying counts to another run or to reordered features."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "variants-supervised-agent"))
from saved_visit_counts import DISTANCE, VISIT_DEFINITION, load_saved_counts


def test_previous_export_loads_only_for_corresponding_features_and_counts(tmp_path):
    dino = dict(archive_sha256="archive", backbone_sha256="weights", preprocessing="pixels")
    dreamer = dict(archive_sha256="archive", checkpoint_sha256="checkpoint", config_sha256="config",
                   representation="posterior")
    manifest = dict(frame_count=4, episode_count=2, action_count=2, archive_sha256="archive",
                    distance=DISTANCE, visit_definition=VISIT_DEFINITION, dino_provenance=dino,
                    dreamer_provenance=dreamer, radius=0.25, radius_method="median_kth_neighbor",
                    radius_sample_size=4, effective_neighbors=2, frames_with_one_visit=2,
                    frames_with_only_self_in_radius=2, representative_count=1,
                    representatives=[dict(archive_frame_index=1, visit_count=1, neighborhood_frame_count=1)])
    arrays = dict(visit_counts=[2, 1, 2, 1], neighborhood_frame_counts=[2, 1, 2, 1],
                  selected_indices=[1], episode_offsets=[0, 2, 4], episode_ids=[4, 1])
    manifest_path, array_path = tmp_path / "manifest.json", tmp_path / "frame_counts.npz"
    manifest_path.write_text(json.dumps(manifest))

    def load(ep_ids=(4, 1), encoder=dino, model=dreamer):
        return load_saved_counts(tmp_path, [0, 2, 4], ep_ids, encoder, model)

    with pytest.raises(FileNotFoundError, match="frame_counts.npz"):
        load()
    np.savez(array_path, **arrays)
    original, (visits, frames) = load()
    assert original == manifest
    np.testing.assert_array_equal(visits, arrays["visit_counts"])
    np.testing.assert_array_equal(frames, arrays["neighborhood_frame_counts"])
    for bad_model, field in [(dict(dreamer, checkpoint_sha256="different"), "checkpoint"),
                             (dict(dreamer, config_sha256="different"), "config")]:
        with pytest.raises(ValueError, match=field):
            load(model=bad_model)
    with pytest.raises(ValueError, match="backbone_sha256"):
        load(encoder=dict(dino, backbone_sha256="different"))
    with pytest.raises(ValueError, match="episode order"):
        load(ep_ids=(1, 4))

    for patch, message in [(dict(distance="another metric"), "distance"), (dict(radius=-1), "radius"),
                           (dict(effective_neighbors=4), "calibration")]:
        manifest_path.write_text(json.dumps(dict(manifest, **patch)))
        with pytest.raises(ValueError, match=message):
            load()
    manifest_path.write_text(json.dumps(dict(manifest, radius_method="fixed", effective_neighbors=None,
                                             radius_sample_size=0)))
    assert load()[0]["radius_method"] == "fixed"
    for patch, message in [(dict(visit_counts=[0, 0, 0, 0]), "per-frame"),
                           (dict(neighborhood_frame_counts=[4, 3]), "per-frame"),
                           (dict(selected_indices=[3]), "representative")]:
        np.savez(array_path, **dict(arrays, **patch))
        with pytest.raises(ValueError, match=message):
            load()
