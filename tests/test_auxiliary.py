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


def test_in_process_classifier_shares_frames_and_never_reloads_files(tmp_path, monkeypatch):
    frames = np.full((3, 84, 84, 3), 10, dtype=np.uint8)
    frames.setflags(write=False)
    episodes = [dict(id=7, frames=frames, actions=np.array([0, 1]), complete=True)]
    args = classifier.parse_args(['--output', str(tmp_path / 'classifier.pt'), '--all-data',
        '--architecture', 'small', '--embedding-dim', '8', '--hidden-dim', '8',
        '--visual-steps', '1', '--gru-steps', '1', '--batch-size', '2', '--eval-pairs', '2',
        '--device', 'cpu', '--threads', '1', '--seed', '3'], require_archive=False)
    cache = classifier.cache_features
    observed = []

    def check_shared(model, sequences, device, **kwargs):
        observed.extend(sequences)
        return cache(model, sequences, device, **kwargs)

    def unexpected_read(*args, **kwargs):
        raise AssertionError('In-process classifier must use its inputs and live best weights')

    monkeypatch.setattr(classifier, 'cache_features', check_shared)
    monkeypatch.setattr(classifier, 'load_archive', unexpected_read)
    monkeypatch.setattr(torch, 'load', unexpected_read)
    model, checkpoint = classifier.train_from_episodes(args, episodes, {}, 'a' * 64)
    assert len(observed) == 1
    assert observed[0] is frames
    assert checkpoint['archive_sha256'] == 'a' * 64
    assert checkpoint['model_fingerprint'] == classifier.model_fingerprint(model)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert not list(tmp_path.rglob('*.pkl')) and not list(tmp_path.rglob('*.npz'))


def test_live_policy_reuses_optimizer_and_latest_weights_without_checkpoint_reads(tmp_path, monkeypatch):
    model = classifier.RecurrentClassifier('small', 8, 8)
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.fill_(-10.)
    model.eval().requires_grad_(False)
    metadata = dict(num_actions=2, action_meanings=['NOOP', 'FIRE'], env_config=dict(task='atari_test'))
    checkpoint = dict(archive_sha256='a' * 64, training_episode_ids=[7], evaluation_episode_ids=[], threshold=.5)
    frames = np.full((3, 84, 84, 3), 20, dtype=np.uint8)
    actions = np.array([0, 1])
    frames.setflags(write=False)
    actions.setflags(write=False)
    demos = supervised.prepare_demonstrations(
        [dict(id=7, frames=frames, actions=actions, complete=True)], [7], model, .5)
    assert demos[0]['frames'] is frames

    def arguments(output):
        return supervised.parse_args(['train', '--output', str(output), '--device', 'cpu',
            '--threads', '1', '--seed', '13', '--hidden-dim', '8', '--attention-heads', '2',
            '--batch-size', '1', '--unroll-length', '2', '--updates', '1', '--score-every', '1'],
            require_inputs=False)

    def unexpected_read(*args, **kwargs):
        raise AssertionError('Live policy continuation must not read a checkpoint or archive')

    initial = []
    original_save = supervised.save_checkpoint

    def capture_initial(path, payload):
        if payload['update'] == 0 and path.name == 'latest.pt':
            initial.append(copy.deepcopy(payload))
        original_save(path, payload)

    # Select the initial policy as best, ensuring next-round fitting must use
    # the updated latest policy rather than accidentally continuing the rollout.
    scores = iter([1., 0., 1., 0.])
    monkeypatch.setattr(supervised, 'score_training', lambda *args: dict(action_accuracy=next(scores), loss=0.))
    monkeypatch.setattr(supervised, 'save_checkpoint', capture_initial)
    monkeypatch.setattr(supervised, 'load_archive', unexpected_read)
    monkeypatch.setattr(torch, 'load', unexpected_read)
    state, rollout = supervised.train_from_demonstrations(
        arguments(tmp_path / 'first'), model, checkpoint, metadata, demos, 'b' * 64)
    policy, optimizer = state.policy, state.optimizer
    previous_weights = classifier.copy_model_state(policy)
    previous_optimizer = copy.deepcopy(optimizer.state_dict())
    assert any(not torch.equal(previous_weights[key], initial[0]['model_state'][key]) for key in previous_weights)
    for key, value in rollout.state_dict().items():
        assert torch.equal(value, initial[0]['model_state'][key])
    assert rollout is not policy and not rollout.training
    assert all(not p.requires_grad for p in rollout.parameters())
    assert state.metadata['optimizer_continuation'] == 'new'
    previous_path, previous_hash = state.checkpoint_path, state.checkpoint_sha256
    state.to('cpu')
    # An in-memory continuation must not even instantiate a replacement policy.
    monkeypatch.setattr(supervised.GoalPolicy, '__init__', unexpected_read)
    checkpoint = checkpoint | dict(archive_sha256='c' * 64)
    continued, _ = supervised.train_from_demonstrations(
        arguments(tmp_path / 'second'), model, checkpoint, metadata, demos, 'd' * 64, state=state)
    assert continued is state and state.policy is policy and state.optimizer is optimizer
    assert state.total_updates == 2
    assert state.metadata['previous_training_updates'] == 1
    assert state.metadata['optimizer_continuation'] == 'memory'
    assert state.metadata['finetuned_from'] == str(previous_path)
    assert state.metadata['finetuned_from_sha256'] == previous_hash
    for key, value in previous_weights.items():
        assert torch.equal(value, initial[1]['model_state'][key])
    for key, values in previous_optimizer['state'].items():
        for name, value in values.items():
            assert torch.equal(value, initial[1]['optimizer_state']['state'][key][name])
    assert all(int(values['step']) == 2 for values in optimizer.state.values())
    assert optimizer.param_groups[0]['lr'] == 3e-4
    assert not list(tmp_path.rglob('*.pkl')) and not list(tmp_path.rglob('*.npz'))
