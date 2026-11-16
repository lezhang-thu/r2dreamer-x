"""Small CPU checks for the periodic classifier/policy training interfaces."""

import copy
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'variants-supervised-agent'))
import train_gru_classifier as classifier
import train_supervised_agent as supervised


def write_archive(path, value=0):
    frames = np.full((3, 84, 84, 3), value, dtype=np.uint8)
    episodes = [dict(id=7, frames=frames, actions=np.array([0, 1]), complete=True)]
    with path.open('wb') as stream:
        pickle.dump(dict(episodes=episodes, metadata=dict(num_actions=2,
                    action_meanings=['NOOP', 'FIRE'], env_config=dict(task='atari_test'))), stream)
    return episodes


def write_classifier(path, archive, bias=-10.):
    model = classifier.RecurrentClassifier('small', 8, 8)
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.fill_(bias)
    classifier.atomic_save(path, dict(kind='standalone_gru_classifier', format_version=1,
        model_config=model.model_config, model_state=model.state_dict(),
        model_fingerprint=classifier.model_fingerprint(model), threshold=.5,
        archive_sha256=classifier.file_sha256(archive), training_episode_ids=[7], evaluation_episode_ids=[]))
    return model


def policy_args(archive, checkpoint, output, extra=()):
    return ['train', '--archive', str(archive), '--classifier', str(checkpoint),
            '--output', str(output), '--device', 'cpu', '--threads', '1', '--seed', '13',
            '--hidden-dim', '8', '--attention-heads', '2', '--batch-size', '1',
            '--unroll-length', '2', '--updates', '1', '--score-every', '1', *extra]


def test_all_data_trains_one_episode_without_holdout(tmp_path):
    archive, output = tmp_path / 'episodes.pkl', tmp_path / 'classifier.pt'
    write_archive(archive)
    classifier.main(['--archive', str(archive), '--output', str(output), '--all-data',
                     '--architecture', 'small', '--embedding-dim', '8', '--hidden-dim', '8',
                     '--visual-steps', '1', '--gru-steps', '1', '--batch-size', '2',
                     '--eval-pairs', '2', '--device', 'cpu', '--threads', '1', '--seed', '3'])
    saved = torch.load(output, weights_only=True)
    assert saved['training_episode_ids'] == saved['internal_train_ids'] == [7]
    assert saved['evaluation_episode_ids'] == saved['internal_validation_ids'] == []
    assert saved['diagnostic_episode_ids'] == [7]
    assert saved['selection_scope'] == 'training_episode_diagnostics_not_heldout'
    report = json.loads(output.with_suffix('.metrics.json').read_text())
    assert len(report['history']) == 2


def test_standalone_split_stays_disjoint_and_requires_enough_episodes():
    episodes = [dict(id=i) for i in range(8)]
    train, test = classifier.split_episodes(episodes, {})
    assert train == list(range(7)) and test == [7]
    with pytest.raises(ValueError, match='at least one test episode'):
        classifier.split_episodes(episodes[:1], {})
    with pytest.raises(ValueError, match='cannot be combined'):
        classifier.split_episodes(episodes, {}, test_episodes=1, all_data=True)


def test_finetune_preserves_weights_and_optimizer_on_new_data(tmp_path, monkeypatch):
    archive, first_classifier = tmp_path / 'one.pkl', tmp_path / 'one.pt'
    write_archive(archive)
    write_classifier(first_classifier, archive)
    first_output = tmp_path / 'first'
    supervised.main(policy_args(archive, first_classifier, first_output))
    previous = torch.load(first_output / 'latest.pt', weights_only=True)

    new_archive, new_classifier = tmp_path / 'two.pkl', tmp_path / 'two.pt'
    write_archive(new_archive, 25)
    write_classifier(new_classifier, new_archive)
    initial_checkpoints = []
    original_save = supervised.save_checkpoint

    def capture_initial(path, payload):
        if payload['update'] == 0:
            initial_checkpoints.append(copy.deepcopy(payload))
        original_save(path, payload)

    monkeypatch.setattr(supervised, 'save_checkpoint', capture_initial)
    second_output = tmp_path / 'second'
    supervised.main(policy_args(new_archive, new_classifier, second_output,
                    ['--finetune', str(first_output / 'latest.pt')]))
    initial = initial_checkpoints[0]
    for name, value in previous['model_state'].items():
        assert torch.equal(value, initial['model_state'][name])
    for key, state in previous['optimizer_state']['state'].items():
        for name, value in state.items():
            assert torch.equal(value, initial['optimizer_state']['state'][key][name])
    final = torch.load(second_output / 'latest.pt', weights_only=True)
    assert final['update'] == 1
    assert final['metadata']['previous_training_updates'] == 1
    assert final['metadata']['archive_sha256'] != previous['metadata']['archive_sha256']
    assert all(int(state['step']) == 2 for state in final['optimizer_state']['state'].values())


def test_empty_anchors_are_explicit_and_never_fabricated(tmp_path, capsys):
    archive, path = tmp_path / 'episodes.pkl', tmp_path / 'classifier.pt'
    episodes = write_archive(archive)
    model = write_classifier(path, archive, bias=10.)
    with pytest.raises(ValueError, match='has no anchors'):
        supervised.prepare_demonstrations(episodes, [7], model, .5)
    with pytest.raises(supervised.NoUsableAnchorsError, match='No usable anchor tasks'):
        supervised.prepare_demonstrations(episodes, [7], model, .5, skip_empty=True)
    assert 'skipped_empty_anchors' in capsys.readouterr().out


def test_finetune_and_resume_cannot_be_combined():
    with pytest.raises(SystemExit):
        supervised.parse_args(['train', '--archive', 'a', '--classifier', 'c', '--output', 'o',
                               '--resume', 'old', '--finetune', 'old'])
