"""Verify a completed full-model GPU integration run from its saved artifacts.

Run with the r2dreamer interpreter after train.py exits successfully. This tool
does not need a GPU and never loads the large Dreamer checkpoint into memory.
Only use it on trusted run directories: episode archives contain pickle files.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import re
import zipfile

import numpy as np
from omegaconf import OmegaConf


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    require(path.is_file(), f'Missing artifact: {path}')
    return json.loads(path.read_text())


def read_jsonl(path):
    require(path.is_file(), f'Missing log: {path}')
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_exists(path):
    require(path.is_file() and path.stat().st_size > 0,
            f'Missing completed checkpoint (run may still be pending): {path}')
    require(zipfile.is_zipfile(path), f'Checkpoint is not a complete PyTorch ZIP file: {path}')
    with zipfile.ZipFile(path) as stream:
        require(any(name.endswith('/data.pkl') for name in stream.namelist()),
                f'Checkpoint is missing serialized data: {path}')


def load_pickle(path):
    require(path.is_file(), f'Missing archive: {path}')
    with path.open('rb') as stream:
        return pickle.load(stream)


def verify(run_dir, expected_rounds=2):
    import torch

    root = Path(run_dir).expanduser().resolve()
    require(expected_rounds >= 2, 'At least two rounds are required to test fine-tuning')
    require((root / 'config.yaml').is_file(), f'Missing run configuration: {root}')
    config = OmegaConf.to_container(OmegaConf.load(root / 'config.yaml'), resolve=True)
    require(str(config['device']).startswith('cuda'), 'Run did not select CUDA')
    require(config['env']['task'] == 'atari_hero', 'Expected the HERO integration run')
    require(config['model']['compile'] is True, 'Dreamer compilation was disabled')
    require(config['model']['rep_loss'] == 'r2dreamer', 'Expected R2-Dreamer')
    for name, value in dict(deter=6144, hidden=768, discrete=48, depth=48, units=768).items():
        require(config['model'][name] == value, f'Model is not size100M: {name}')
    require(config['batch_size'] == 16 and config['batch_length'] == 64,
            'Dreamer training batch settings differ from production')
    require(config['buffer']['storage_device'] == 'cpu', 'Expected CPU replay storage')
    require(config['buffer']['max_size'] == 1000000, 'Expected one-million-row replay capacity')
    auxiliary = config['return_exploration']
    require(auxiliary['enabled'] and auxiliary['episodes'] == 100, 'Expected 100 assisted episodes per cycle')
    require(auxiliary['initial_policy'] is None, 'First cycle must train the default new return policy')
    for name, value in dict(architecture='resnet18', visual_steps=5000, gru_steps=5000,
                            window=20, batch_size=64).items():
        require(auxiliary['classifier'][name] == value, f'Classifier override is not production: {name}')
    for name, value in dict(updates=10000, batch_size=4, unroll_length=2048).items():
        require(auxiliary['policy'][name] == value, f'Policy override is not production: {name}')
    checkpoint_exists(root / 'latest.pt')
    directories = sorted((root / 'return_exploration').glob('round_*'))
    require(len(directories) == expected_rounds,
            f'Expected {expected_rounds} completed rounds, found {len(directories)}')
    console = (root / 'console.log').read_text()
    require('Traceback (most recent call last)' not in console, 'Run console contains a traceback')
    require('Compiling update function with torch.compile' in console, 'No compiled-model startup evidence')
    events = [json.loads(line) for line in console.splitlines() if line.startswith('{')]
    events = [event for event in events if event.get('event') == 'return_episode']
    require(len(events) == 100 * expected_rounds, 'Missing or duplicated per-episode JSON console records')
    summary_pattern = re.compile(r'^\[return round (\d+)\] episode (\d+)/(\d+) return=([-+\d.eE]+) ', re.M)
    printed = summary_pattern.findall(console)
    require(len(printed) == 100 * expected_rounds, 'Missing or duplicated human-readable return summaries')
    metrics = read_jsonl(root / 'metrics.jsonl')
    updates = [row for row in metrics if 'train/opt/updates' in row]
    require(updates, 'No Dreamer optimizer updates were logged')
    for row in updates:
        require(all(math.isfinite(value) for value in row.values() if isinstance(value, (int, float))),
                f'Nonfinite Dreamer metric at frame step {row["step"]}')
        require(row.get('train/opt/updates_skipped', 0) == 0,
                f'Dreamer skipped nonfinite updates at frame step {row["step"]}')

    previous_sources, previous_generated, previous_directory = set(), set(), None
    classifier_hashes, round_summaries = [], []
    cumulative_assisted = 0
    for number, directory in enumerate(directories, 1):
        label = f'Round {number}'
        require(directory.name == f'round_{number:06d}', f'{label}: round numbering is not consecutive')
        report = read_json(directory / 'report.json')
        plan = read_json(directory / 'plan.json')
        rollouts = read_json(directory / 'rollouts.json')
        counts = read_json(directory / 'counts.json')
        initial_bank = read_json(directory / 'anchors.initial.json')
        require(report['generated_episodes'] == len(rollouts) == counts['completed_episodes'] == 100,
                f'{label}: not all 100 assisted episodes completed')
        require(len(counts['counts']) == len(initial_bank['counts']) == report['representatives'],
                f'{label}: anchor set changed during generation')
        require(all(end >= start for start, end in zip(initial_bank['counts'], counts['counts'])),
                f'{label}: an anchor counter decreased')
        sources = load_pickle(directory / 'episodes.pkl')['episodes']
        source_ids = {record['id'] for record in sources}
        require(len(source_ids) == len(sources) == report['source_episodes'], f'{label}: duplicate source IDs')
        require(source_ids == set(plan['source_ids']), f'{label}: snapshot and plan disagree')
        require(all(record.get('complete') for record in sources), f'{label}: incomplete source episode')
        if number == 1:
            require(all(record['source'] == 'ordinary' for record in sources), f'{label}: initial S is not ordinary data')
        else:
            assisted = {record['id'] for record in sources if record['source'] == 'assisted'}
            require(assisted == previous_generated, f'{label}: S does not contain exactly the previous 100 assisted episodes')
            require(not source_ids & previous_sources, f'{label}: older source episodes were reused')
            ordinary = [record for record in sources if record['source'] == 'ordinary']
            require(ordinary, f'{label}: no newly completed ordinary source episodes')
            require(all(record['end_step'] > (number - 1) * auxiliary['interval'] for record in ordinary),
                    f'{label}: old ordinary episodes entered the snapshot')

        classifier_metrics = read_json(directory / 'classifier.metrics.json')
        require(set(classifier_metrics['training_episode_ids']) == source_ids,
                f'{label}: classifier omitted source episodes')
        require(classifier_metrics['evaluation_episode_ids'] == [], f'{label}: classifier reserved a holdout')
        for stage in ('visual', 'gru'):
            history = [row for row in classifier_metrics['history'] if row['stage'] == stage]
            require(history and history[-1]['step'] == 5000, f'{label}: {stage} training did not reach 5000 updates')
        require(str(classifier_metrics['train_config']['resolved_device']).startswith('cuda'),
                f'{label}: classifier was not trained on CUDA')
        checkpoint_exists(directory / 'classifier.pt')
        checkpoint_exists(directory / 'classifier.visual.pt')
        classifier_hash = sha256(directory / 'classifier.pt')
        require(classifier_hash not in classifier_hashes, f'{label}: classifier checkpoint reused from a prior cycle')
        classifier_hashes.append(classifier_hash)

        policy_log = read_jsonl(directory / 'agent' / 'training.jsonl')
        require([row['update'] for row in policy_log] == list(range(1, 10001)),
                f'{label}: policy did not complete exactly 10000 additional updates')
        require(all(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']) for row in policy_log),
                f'{label}: nonfinite policy training metric')
        policy_report = read_json(directory / 'agent' / 'training_report.json')
        require(policy_report['update'] == 10000, f'{label}: incomplete policy report')
        for name in ('best.pt', 'latest.pt'):
            checkpoint_exists(directory / 'agent' / name)
        policy = torch.load(directory / 'agent' / 'latest.pt', map_location='cpu', weights_only=True)
        metadata = policy['metadata']
        require(policy['update'] == 10000, f'{label}: latest policy checkpoint is incomplete')
        require(policy.get('cuda_rng_state') is not None, f'{label}: policy checkpoint lacks CUDA training RNG state')
        require(metadata['classifier_sha256'] == classifier_hash, f'{label}: policy uses another classifier')
        require(metadata['archive_sha256'] == sha256(directory / 'episodes.pkl'), f'{label}: policy uses another archive')
        require(set(metadata['training_episode_ids']) == source_ids, f'{label}: policy source IDs differ')
        if previous_directory is None:
            require('finetuned_from' not in metadata, f'{label}: unexpected initial warm start')
        else:
            previous_policy = previous_directory / 'agent' / 'latest.pt'
            require(Path(metadata['finetuned_from']).resolve() == previous_policy,
                    f'{label}: fine-tuning did not use the preceding latest policy')
            require(metadata['finetuned_from_sha256'] == sha256(previous_policy),
                    f'{label}: fine-tuning input checkpoint hash differs')
            require(metadata['previous_training_updates'] == (number - 1) * 10000,
                    f'{label}: cumulative previous policy updates were reset')
        optimizer_steps = [int(value['step']) for value in policy['optimizer_state']['state'].values() if 'step' in value]
        require(optimizer_steps and max(optimizer_steps) == number * 10000,
                f'{label}: optimizer step history was not retained across cycles')
        del policy
        checkpoint_exists(directory / 'dreamer.pt')

        round_events = [event for event in events if event['round'] == number]
        require([event['episode'] for event in round_events] == list(range(1, 101)), f'{label}: printed episodes are incomplete')
        round_printed = [(int(index), int(total), float(value)) for rnd, index, total, value in printed if int(rnd) == number]
        require([(index, total) for index, total, _ in round_printed] == [(index, 100) for index in range(1, 101)],
                f'{label}: human-readable episode counters are incomplete')
        for rollout, event, (_, _, printed_return) in zip(rollouts, round_events, round_printed):
            value = rollout['env_return']
            require(math.isfinite(value) and event['env_return'] == value, f'{label}: missing/nonfinite/inconsistent return')
            require(math.isclose(printed_return, value, abs_tol=.0051), f'{label}: printed return differs from saved return')
            record = load_pickle(root / 'episode_archive' / 'episodes' / f'{rollout["episode_id"]:09d}.pkl')
            require(record.get('complete') and record['source'] == 'assisted' and record['return_round'] == number,
                    f'{label}: generated episode is missing or incomplete in archive')
            with np.load(io.BytesIO(record['payload_npz']), allow_pickle=False) as arrays:
                rewards = arrays['rewards']
                require(len(rewards) == rollout['interactions'], f'{label}: rollout interaction count differs from archive')
                require(math.isclose(float(rewards.sum(dtype=np.float64)), value, rel_tol=1e-6, abs_tol=1e-6),
                        f'{label}: return is not the sum of the complete episode rewards')
        generated = {row['episode_id'] for row in rollouts}
        require(len(generated) == 100 and not generated & source_ids, f'{label}: duplicate/reused generated episode IDs')
        cumulative_assisted += sum(row['interactions'] for row in rollouts)
        ordinary = number * auxiliary['interval']
        require(report['ordinary_interactions'] == ordinary, f'{label}: assisted interactions advanced the ordinary clock')
        require(report['assisted_interactions'] == cumulative_assisted, f'{label}: cumulative assisted count differs')
        require(report['total_interactions'] == ordinary + cumulative_assisted, f'{label}: total interaction count differs')
        boundary = ordinary * config['env']['action_repeat']
        before = [row for row in updates if row['step'] <= boundary]
        after = [row for row in updates if row['step'] > boundary]
        require(before and after and after[0]['train/opt/updates'] > before[-1]['train/opt/updates'],
                f'{label}: no logged Dreamer optimizer progress after the return phase')
        round_summaries.append(dict(round=number, source_episodes=len(source_ids), generated_episodes=100,
                                   reached=report['reached'], takeovers=report['takeovers'],
                                   episode_interactions_min=min(row['interactions'] for row in rollouts),
                                   episode_interactions_max=max(row['interactions'] for row in rollouts),
                                   return_min=min(row['env_return'] for row in rollouts),
                                   return_max=max(row['env_return'] for row in rollouts),
                                   return_mean=sum(row['env_return'] for row in rollouts) / 100,
                                   ordinary_interactions=ordinary, assisted_interactions=cumulative_assisted,
                                   classifier_sha256=classifier_hash,
                                   policy_optimizer_steps=max(optimizer_steps)))
        previous_sources, previous_generated, previous_directory = source_ids, generated, directory

    ordinary_budget = config['env']['steps'] // config['env']['action_repeat']
    require(sum(row['takeovers'] for row in round_summaries) > 0,
            'No live return-to-Dreamer handoff was exercised')
    require(ordinary_budget > expected_rounds * auxiliary['interval'], 'Run ended at the last cycle without resumed collection')
    return dict(status='passed', run_dir=str(root), device=config['device'], task=config['env']['task'],
                model='size100M', compile=True, cycle_interval=auxiliary['interval'],
                episode_time_limit_frames=config['env']['time_limit'],
                ordinary_interaction_budget=ordinary_budget, assisted_interactions=cumulative_assisted,
                completed_rounds=expected_rounds, completed_assisted_episodes=100 * expected_rounds,
                last_logged_dreamer_updates=updates[-1]['train/opt/updates'], rounds=round_summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--expected-rounds', type=int, default=2)
    args = parser.parse_args()
    try:
        result = verify(args.run_dir, args.expected_rounds)
    except (AssertionError, KeyError, ValueError, OSError) as error:
        parser.exit(1, f'GPU integration verification failed: {error}\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
