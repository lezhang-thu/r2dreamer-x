"""Exercise the native PNG export with a saved classifier and real GRU histories."""

import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'variants-supervised-agent'))
import train_gru_classifier as classifier


def make_episode(episode_id, intensities, *, complete=True):
    frames = np.stack([np.full((210, 160, 3), value, dtype=np.uint8) for value in intensities])
    # A colored sprite on the bottom edge identifies the exact source pixels.
    frames[-1, -1, -1] = [episode_id, 21, 205]
    return dict(id=episode_id, frames=frames, actions=np.zeros(len(frames) - 1, dtype=np.int64),
                complete=complete, collection_cutoff=not complete)


def write_inputs(tmp_path, episodes):
    archive = tmp_path / 'episodes.pkl'
    with archive.open('wb') as stream:
        pickle.dump(dict(episodes=episodes, metadata=dict(num_actions=1)), stream)

    # For constant gray images the stub's CLS vector contains the normalized mean.
    # This GRU retains half its previous state; the head matches |a - b| <= 0.2.
    model = classifier.RecurrentClassifier(hidden_dim=1)
    threshold = .7
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.gru.weight_ih_l0[2, 0] = 1
        model.head[0].weight[0, 0] = 1
        model.head[-1].weight[0, 0] = -1
        model.head[-1].bias.fill_(math.log(threshold / (1 - threshold)) + .2)
    checkpoint = tmp_path / 'classifier.pt'
    ids = [ep['id'] for ep in episodes]
    torch.save(dict(kind='standalone_gru_classifier', format_version=2,
                    model_config=model.model_config, model_state=model.classifier_state_dict(),
                    model_fingerprint=classifier.model_fingerprint(model),
                    archive_sha256=classifier.file_sha256(archive), threshold=threshold,
                    training_episode_ids=ids[:1], evaluation_episode_ids=ids[1:]), checkpoint)
    return ['--archive', str(archive), '--classifier', str(checkpoint), '--device', 'cpu']


@pytest.mark.parametrize('complete_only,expected_counts', [(False, [3, 2]), (True, [2, 1])])
def test_exports_first_origins_in_archive_order(tmp_path, stub_dino, complete_only, expected_counts):
    from export_anchor_set import main

    # The final frame of each episode is nearly identical. The different GRU
    # histories produce contexts A, B, C, B with A~B, B~C, A!~C.
    episodes = [make_episode(9, [90, 90, 255]), make_episode(6, [122, 255]),
                make_episode(4, [174, 255]), make_episode(2, [122, 255], complete=False)]
    args = write_inputs(tmp_path, episodes)
    output = tmp_path / 'anchors'
    main([*args, '--output', str(output), '--feature-batch-size', '1',
          *(['--complete-only'] if complete_only else [])])

    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['episode_count'] == manifest['candidate_count'] == (3 if complete_only else 4)
    assert manifest['representative_count'] == 2
    reps = manifest['representatives']
    assert [(r['id'], r['episode_id'], r['anchor_position'], r['frame_index']) for r in reps] == [
        (0, 9, 0, 2), (1, 4, 0, 1)]
    assert [r['count'] for r in reps] == expected_counts
    assert {output / r['file'] for r in reps} == set(output.rglob('*.png'))
    by_id = {ep['id']: ep for ep in episodes}
    for row in reps:
        original = by_id[row['episode_id']]['frames'][row['frame_index']]
        with Image.open(output / row['file']) as image:
            assert image.format == 'PNG' and image.mode == 'RGB' and image.size == (160, 210)
            np.testing.assert_array_equal(np.asarray(image), original)


def test_empty_set_writes_manifest_without_reset_or_terminal_png(tmp_path, stub_dino):
    from export_anchor_set import main

    args = write_inputs(tmp_path, [make_episode(5, [122, 122, 122])])
    output = tmp_path / 'empty'
    main([*args, '--output', str(output)])
    assert json.loads((output / 'manifest.json').read_text())['representatives'] == []
    assert not list(output.rglob('*.png'))
