"""Verify a completed full-model GPU integration run from its saved artifacts.

Run with the r2dreamer interpreter after train.py exits successfully. This tool
does not need a GPU and never loads the large Dreamer checkpoint into memory.
Only use it on trusted run directories: legacy episode archives contain pickle
files. Memory-only runs are checked through retained cycle and episode logs;
in-process runs also record live optimizer continuation between cycles.
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


def verify(run_dir, expected_rounds=2, allow_short_training=False):
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
    classifier_settings = auxiliary['classifier']
    stages = ('visual', 'gru') if 'visual_steps' in classifier_settings else ('gru',)
    expected_encoder = (dict(architecture='resnet18') if 'visual' in stages else
                        dict(backbone='facebook/dinov3-vitl16-pretrain-lvd1689m'))
    for name, value in dict(**expected_encoder, window=20, batch_size=64).items():
        require(auxiliary['classifier'][name] == value, f'Classifier override is not production: {name}')
    for name, value in dict(batch_size=4, unroll_length=2048).items():
        require(auxiliary['policy'][name] == value, f'Policy override is not production: {name}')
    counts = [('classifier', f'{stage}_steps', 5000) for stage in stages]
    for section, name, value in [*counts, ('policy', 'updates', 10000)]:
        actual = auxiliary[section][name]
        require(isinstance(actual, int) and actual > 0, f'Invalid training count: {section}.{name}')
        require(allow_short_training or actual == value,
                f'{section}.{name} is not production; use --allow-short-training for a smoke run')
    checkpoint_exists(root / 'latest.pt')
    directories = sorted((root / 'return_exploration').glob('round_*'))
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

    if (root / 'return_exploration' / 'cycles.jsonl').is_file():
        return verify_memory_run(root, config, expected_rounds, events, printed, updates,
                                 console=console, metrics=metrics,
                                 allow_short_training=allow_short_training)

    require(len(directories) == expected_rounds,
            f'Expected {expected_rounds} completed rounds, found {len(directories)}')
    policy_updates = auxiliary['policy']['updates']
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
        for stage in stages:
            history = [row for row in classifier_metrics['history'] if row['stage'] == stage]
            target = auxiliary['classifier'][f'{stage}_steps']
            require(history and history[-1]['step'] == target,
                    f'{label}: {stage} training did not reach {target} updates')
        require(str(classifier_metrics['train_config']['resolved_device']).startswith('cuda'),
                f'{label}: classifier was not trained on CUDA')
        checkpoint_exists(directory / 'classifier.pt')
        if 'visual' in stages:
            checkpoint_exists(directory / 'classifier.visual.pt')
        classifier_hash = sha256(directory / 'classifier.pt')
        require(classifier_hash not in classifier_hashes, f'{label}: classifier checkpoint reused from a prior cycle')
        classifier_hashes.append(classifier_hash)

        policy_log = read_jsonl(directory / 'agent' / 'training.jsonl')
        require([row['update'] for row in policy_log] == list(range(1, policy_updates + 1)),
                f'{label}: policy did not complete exactly {policy_updates} additional updates')
        require(all(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']) for row in policy_log),
                f'{label}: nonfinite policy training metric')
        policy_report = read_json(directory / 'agent' / 'training_report.json')
        require(policy_report['update'] == policy_updates, f'{label}: incomplete policy report')
        for name in ('best.pt', 'latest.pt'):
            checkpoint_exists(directory / 'agent' / name)
        policy = torch.load(directory / 'agent' / 'latest.pt', map_location='cpu', weights_only=True)
        metadata = policy['metadata']
        require(policy['update'] == policy_updates, f'{label}: latest policy checkpoint is incomplete')
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
            require(metadata['previous_training_updates'] == (number - 1) * policy_updates,
                    f'{label}: cumulative previous policy updates were reset')
        optimizer_steps = [int(value['step']) for value in policy['optimizer_state']['state'].values() if 'step' in value]
        require(optimizer_steps and max(optimizer_steps) == number * policy_updates,
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


def verify_memory_run(root, config, expected_rounds, events, printed, updates, *,
                      console, metrics, allow_short_training=False):
    import torch

    auxiliary = config['return_exploration']
    exploration = root / 'return_exploration'
    reports = read_jsonl(exploration / 'cycles.jsonl')
    require(len(reports) == expected_rounds,
            f'Expected {expected_rounds} completed cycle summaries, found {len(reports)}')
    execution = reports[0].get('training_execution')
    require(execution in (None, 'in_process'), f'Unexpected auxiliary training execution: {execution}')
    directory = exploration / f'round_{expected_rounds:06d}'
    require(sorted(exploration.glob('round_*')) == [directory] and directory.is_dir(),
            'Completed run must retain exactly the latest round directory')
    plan = read_json(directory / 'plan.json')
    prefix_selection = plan.get('prefix_sampling', 'legacy')
    shortest_prefixes = prefix_selection == 'shortest_matching_anchor_prefix'
    if shortest_prefixes:
        require(plan.get('prefix_tiebreak') == 'lowest_episode_id',
                'Shortest-prefix selection has an unexpected tie-break rule')
    evaluation_disabled = None
    if execution == 'in_process':
        require(config['env']['eval_episode_num'] == 0,
                'In-process run did not disable evaluation environment configuration')
        require('Evaluating the policy' not in console,
                'In-process run launched Dreamer evaluation')
        require(not any(key.startswith('episode/eval_') for row in metrics for key in row),
                'In-process run logged Dreamer evaluation metrics')
        evaluation_disabled = True
    require(not (root / 'episode_archive').exists(), 'A persistent episode archive remains')
    trajectory_files = sorted(path.relative_to(root).as_posix() for path in root.rglob('*')
                              if path.is_file() and path.suffix in ('.pkl', '.npz'))
    require(not trajectory_files, f'Trajectory files remain on disk: {trajectory_files}')
    forbidden = {'anchors.initial.json', 'counts.json', 'dreamer.pt', 'classifier.visual.pt'}
    require(not any(path.name in forbidden for path in exploration.rglob('*')),
            'Historical anchor dumps or redundant checkpoints remain')
    expected_models = {root / 'latest.pt', directory / 'classifier.pt',
                       directory / 'agent' / 'best.pt', directory / 'agent' / 'latest.pt'}
    require(set(root.rglob('*.pt')) == expected_models,
            'Model checkpoint retention differs from one Dreamer and one completed auxiliary round')
    require(not list(root.rglob('*.pt.tmp')), 'An unfinished checkpoint replacement remains')
    for path in expected_models:
        checkpoint_exists(path)

    consumed_sources, previous_generated = set(), set()
    classifier_hashes, round_summaries = set(), []
    cumulative_assisted = 0
    latest_events = []
    for number, report in enumerate(reports, 1):
        label = f'Round {number}'
        require(report['trajectory_storage'] == 'memory', f'{label}: source trajectories were not memory-only')
        require(report.get('training_execution') == execution,
                f'{label}: auxiliary training execution changed between cycles')
        source_ids = set(report['source_episode_ids'])
        generated = set(report['generated_episode_ids'])
        require(len(source_ids) == len(report['source_episode_ids']) == report['source_episodes'] > 0,
                f'{label}: empty or duplicate source IDs')
        require(not source_ids & consumed_sources, f'{label}: older source episodes were reused')
        if number > 1:
            require(previous_generated <= source_ids,
                    f'{label}: S does not contain all previous 100 assisted episodes')
            require(source_ids - previous_generated, f'{label}: no newly completed ordinary source episodes')
        require(len(generated) == len(report['generated_episode_ids']) == report['generated_episodes'] == 100,
                f'{label}: not all 100 assisted episodes completed')
        require(not generated & (consumed_sources | source_ids), f'{label}: generated IDs were reused')
        require(len(report['pending_episode_ids']) == 100 and set(report['pending_episode_ids']) == generated,
                f'{label}: consumed source records remain in the pending pool')
        for key in ('source_sha256', 'classifier_sha256', 'policy_sha256'):
            require(re.fullmatch(r'[0-9a-f]{64}', report[key]), f'{label}: invalid {key}')
        require(report['classifier_sha256'] not in classifier_hashes,
                f'{label}: classifier checkpoint reused from a prior cycle')
        classifier_hashes.add(report['classifier_sha256'])
        require(Path(report['policy_checkpoint']).resolve() ==
                exploration / f'round_{number:06d}' / 'agent' / 'latest.pt',
                f'{label}: unexpected saved policy path')

        round_events = [event for event in events if event['round'] == number]
        require([event['episode'] for event in round_events] == list(range(1, 101)),
                f'{label}: printed episodes are incomplete')
        require([event['episode_id'] for event in round_events] == report['generated_episode_ids'],
                f'{label}: generated IDs differ between cycle and console records')
        round_printed = [(int(index), int(total), float(value)) for rnd, index, total, value in printed
                         if int(rnd) == number]
        require([(index, total) for index, total, _ in round_printed] ==
                [(index, 100) for index in range(1, 101)],
                f'{label}: human-readable episode counters are incomplete')
        selected_prefixes = {}
        for event, (_, _, printed_return) in zip(round_events, round_printed):
            require(math.isfinite(event['env_return']), f'{label}: nonfinite episode return')
            require(math.isclose(printed_return, event['env_return'], abs_tol=.0051),
                    f'{label}: human-readable return differs from console JSON')
            require('anchor_indices' not in event, f'{label}: generated anchor sequences were dumped')
            require(event['interactions'] > 0, f'{label}: empty generated episode')
            require(event['source_episode_id'] in source_ids, f'{label}: goal prefix is outside S')
            require(isinstance(event['prefix_anchor_position'], int) and event['prefix_anchor_position'] >= 0,
                    f'{label}: invalid inclusive prefix position')
            require(0 <= event['representative'] < report['representatives'],
                    f'{label}: selected representative is outside the frozen bank')
            if shortest_prefixes:
                prefix = (event['source_episode_id'], event['prefix_anchor_position'])
                previous_prefix = selected_prefixes.setdefault(event['representative'], prefix)
                require(prefix == previous_prefix,
                        f'{label}: a representative changed its precomputed shortest prefix')
                # Source arrays are deliberately released. Unit tests establish
                # minimality; retained logs can verify deterministic selection.
            require(event['target_count_before'] >= 1 and event['counter_increments'] >= 0,
                    f'{label}: invalid anchor counter update')
        require(report['initial_counter_total'] >= report['representatives'] > 0,
                f'{label}: initial anchor counters are invalid')
        require(report['final_counter_total'] == report['initial_counter_total'] +
                sum(event['counter_increments'] for event in round_events),
                f'{label}: anchor counter totals differ from rollout increments')
        require(report['reached'] == sum(event['reached'] for event in round_events) and
                report['takeovers'] == sum(event['takeover'] for event in round_events),
                f'{label}: handoff summaries differ from episode records')
        cumulative_assisted += sum(event['interactions'] for event in round_events)
        ordinary = number * auxiliary['interval']
        require(report['ordinary_interactions'] == ordinary,
                f'{label}: assisted interactions advanced the ordinary clock')
        require(report['assisted_interactions'] == cumulative_assisted and
                report['total_interactions'] == ordinary + cumulative_assisted,
                f'{label}: assisted or total interaction count differs')
        require(report['next_milestone'] == ordinary + auxiliary['interval'],
                f'{label}: next ordinary collection milestone is incorrect')
        boundary = ordinary * config['env']['action_repeat']
        before = [row for row in updates if row['step'] <= boundary]
        after = [row for row in updates if row['step'] > boundary]
        require(before and after and after[0]['train/opt/updates'] > before[-1]['train/opt/updates'],
                f'{label}: no logged Dreamer optimizer progress after the return phase')
        round_summaries.append(dict(round=number, source_episodes=len(source_ids), generated_episodes=100,
                                   reached=report['reached'], takeovers=report['takeovers'],
                                   episode_interactions_min=min(event['interactions'] for event in round_events),
                                   episode_interactions_max=max(event['interactions'] for event in round_events),
                                   return_min=min(event['env_return'] for event in round_events),
                                   return_max=max(event['env_return'] for event in round_events),
                                   return_mean=sum(event['env_return'] for event in round_events) / 100,
                                   ordinary_interactions=ordinary, assisted_interactions=cumulative_assisted,
                                   classifier_sha256=report['classifier_sha256']))
        consumed_sources.update(source_ids)
        previous_generated, latest_events = generated, round_events

    report = reports[-1]
    require(read_json(directory / 'report.json') == report, 'Latest cycle report differs from history')
    require(plan['trajectory_storage'] == 'memory' and plan['source_sha256'] == report['source_sha256'],
            'Latest plan differs from the in-memory source summary')
    require(plan.get('training_execution') == execution,
            'Latest plan and cycle report disagree on auxiliary training execution')
    require(plan['source_ids'] == report['source_episode_ids'] and plan['episodes'] == 100,
            'Latest plan differs from the source pool or requested generation count')
    previous_policy = exploration / f'round_{expected_rounds - 1:06d}' / 'agent' / 'latest.pt'
    require(Path(plan['previous_policy']).resolve() == previous_policy,
            'Latest plan did not fine-tune the preceding round policy')
    rollouts = read_json(directory / 'rollouts.json')
    require(rollouts == [{key: value for key, value in event.items()
                          if key not in ('event', 'round', 'episode')} for event in latest_events],
            'Latest rollout records differ from per-episode console JSON')

    classifier_metrics = read_json(directory / 'classifier.metrics.json')
    source_ids = set(report['source_episode_ids'])
    require(set(classifier_metrics['training_episode_ids']) == source_ids,
            'Latest classifier omitted source episodes')
    require(classifier_metrics['evaluation_episode_ids'] == [], 'Latest classifier reserved a holdout')
    require(classifier_metrics['archive_sha256'] == report['source_sha256'],
            'Latest classifier used another source set')
    for stage in (('visual', 'gru') if 'visual_steps' in auxiliary['classifier'] else ('gru',)):
        history = [row for row in classifier_metrics['history'] if row['stage'] == stage]
        target = auxiliary['classifier'][f'{stage}_steps']
        require(history and history[-1]['step'] == target,
                f'Latest classifier {stage} training did not reach {target} updates')
    require(str(classifier_metrics['train_config']['resolved_device']).startswith('cuda'),
            'Latest classifier was not trained on CUDA')
    require(sha256(directory / 'classifier.pt') == report['classifier_sha256'],
            'Latest classifier hash differs from the cycle summary')
    policy_updates = auxiliary['policy']['updates']
    policy_log = read_jsonl(directory / 'agent' / 'training.jsonl')
    require([row['update'] for row in policy_log] == list(range(1, policy_updates + 1)),
            f'Latest policy did not complete exactly {policy_updates} additional updates')
    require(all(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']) for row in policy_log),
            'Latest policy training has nonfinite metrics')
    require(read_json(directory / 'agent' / 'training_report.json')['update'] == policy_updates,
            'Latest policy report is incomplete')
    policy_path = directory / 'agent' / 'latest.pt'
    require(sha256(policy_path) == report['policy_sha256'], 'Latest policy hash differs from the cycle summary')
    policy = torch.load(policy_path, map_location='cpu', weights_only=True)
    metadata = policy['metadata']
    if execution == 'in_process':
        require(metadata.get('optimizer_continuation') == 'memory',
                'Latest return policy did not retain its live optimizer across cycles')
    require(policy['update'] == policy_updates, 'Latest policy checkpoint is incomplete')
    require(policy.get('cuda_rng_state') is not None, 'Latest policy checkpoint lacks CUDA training RNG state')
    require(metadata['classifier_sha256'] == report['classifier_sha256'] and
            metadata['archive_sha256'] == report['source_sha256'],
            'Latest policy classifier or archive provenance differs')
    require(set(metadata['training_episode_ids']) == source_ids, 'Latest policy source IDs differ')
    require(Path(metadata['finetuned_from']).resolve() == previous_policy,
            'Latest policy did not fine-tune the preceding policy checkpoint')
    require(metadata['finetuned_from_sha256'] == reports[-2]['policy_sha256'],
            'Fine-tuning input hash differs from the previous retained cycle summary')
    require(metadata['previous_training_updates'] == (expected_rounds - 1) * policy_updates,
            'Cumulative previous policy updates were reset')
    optimizer_steps = [int(value['step']) for value in policy['optimizer_state']['state'].values() if 'step' in value]
    require(optimizer_steps and max(optimizer_steps) == expected_rounds * policy_updates,
            'Policy optimizer step history was not retained across cycles')
    if execution == 'in_process':
        populated_states = [value for value in policy['optimizer_state']['state'].values() if value]
        require(populated_states and all('step' in value and int(value['step']) == expected_rounds * policy_updates
                                         for value in populated_states),
                'Some live optimizer states did not retain the full cumulative update count')
    del policy
    round_summaries[-1]['policy_optimizer_steps'] = max(optimizer_steps)
    ordinary_budget = config['env']['steps'] // config['env']['action_repeat']
    require(sum(row['takeovers'] for row in round_summaries) > 0,
            'No live return-to-Dreamer handoff was exercised')
    require(ordinary_budget > expected_rounds * auxiliary['interval'],
            'Run ended at the last cycle without resumed collection')
    return dict(status='passed', run_dir=str(root), device=config['device'], task=config['env']['task'],
                model='size100M', compile=True, cycle_interval=auxiliary['interval'],
                trajectory_storage='memory', training_profile='smoke' if allow_short_training else 'production',
                training_execution=execution or 'legacy',
                evaluation_disabled=evaluation_disabled, prefix_selection=prefix_selection,
                retained_round_directories=1, episode_time_limit_frames=config['env']['time_limit'],
                ordinary_interaction_budget=ordinary_budget, assisted_interactions=cumulative_assisted,
                completed_rounds=expected_rounds, completed_assisted_episodes=100 * expected_rounds,
                last_logged_dreamer_updates=updates[-1]['train/opt/updates'], rounds=round_summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--expected-rounds', type=int, default=2)
    parser.add_argument('--allow-short-training', action='store_true',
                        help='Accept positive configured auxiliary update counts instead of production counts')
    args = parser.parse_args()
    try:
        result = verify(args.run_dir, args.expected_rounds, args.allow_short_training)
    except (AssertionError, KeyError, ValueError, OSError) as error:
        parser.exit(1, f'GPU integration verification failed: {error}\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
