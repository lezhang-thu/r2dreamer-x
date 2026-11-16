"""Exercise default policy batches and long observation prefixes on a real GPU.

Uses the longest complete episode in a trusted source archive. Updates run
on an in-memory copy of the saved policy/optimizer; input checkpoints are never
rewritten. The second update reconstructs the first 2,048 observations before
differentiating through the next action window. For longer episodes, a final
window also reconstructs the largest complete 2,048-step prefix boundary.
"""

import argparse
import gc
import json
import math
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(args):
    import numpy as np
    import torch
    from torch.nn import functional as F

    variant = Path(__file__).resolve().parents[1] / 'variants-supervised-agent'
    sys.path.insert(0, str(variant))
    import train_gru_classifier as classifier_module
    import train_supervised_agent as supervised

    device = 'cuda:0'
    require(torch.cuda.is_available(), 'A CUDA GPU is required; run outside a sandbox that hides GPU devices')
    paths = dict(archive=args.archive.resolve(), classifier=args.classifier.resolve(),
                 policy=args.policy.resolve())
    require(args.output.resolve() not in paths.values(), 'Output must not replace an input artifact')
    before = {name: classifier_module.file_sha256(path) for name, path in paths.items()}
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    episodes, archive_metadata = classifier_module.load_archive(paths['archive'])
    candidates = [episode for episode in episodes if episode.get('actions') is not None]
    require(candidates, 'Archive contains no action-supervised episodes')
    source = max(candidates, key=lambda episode: len(episode['actions']))
    action_count = len(source['actions'])
    require(action_count > 2048, f'Longest source trajectory has only {action_count} actions; need more than 2048')
    del episodes, candidates
    gc.collect()

    classifier, classifier_checkpoint = classifier_module.load_classifier(paths['classifier'], device)
    demos = supervised.prepare_demonstrations([source], [source['id']], classifier,
                 classifier_checkpoint['threshold'], device=device)
    saved = torch.load(paths['policy'], map_location='cpu', weights_only=True)
    require(saved.get('kind') == 'standalone_goal_bc' and saved.get('format_version') == 2,
            'Expected a version 2 supervised policy checkpoint')
    policy = supervised.GoalPolicy(**saved['model_config']).to(device)
    policy.load_state_dict(saved['model_state'])
    metadata = saved['metadata']
    source_task = (archive_metadata.get('env_config') or {}).get('task')
    policy_task = (metadata.get('env_config') or {}).get('task')
    require(source_task == policy_task == 'atari_hero', 'Both source archive and policy must be for HERO')
    require(archive_metadata.get('num_actions') == policy.num_actions,
            'Source archive and policy action counts differ')
    if archive_metadata.get('action_meanings') and metadata.get('action_meanings'):
        require(archive_metadata['action_meanings'] == metadata['action_meanings'], 'Source and policy action mapping differs')
    require(np.all((source['actions'] >= 0) & (source['actions'] < policy.num_actions)), 'Invalid source action labels')
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0.)
    optimizer.load_state_dict(saved['optimizer_state'])

    def optimizer_steps():
        return [int(state['step']) for state in optimizer.state.values() if 'step' in state]

    initial_steps = optimizer_steps()
    require(initial_steps and len(set(initial_steps)) == 1, 'Saved optimizer does not have a uniform step history')
    initial_step = initial_steps[0]
    expected_step = metadata.get('previous_training_updates', 0) + saved['update']
    require(initial_step == expected_step, 'Saved optimizer steps disagree with cumulative policy training updates')
    del saved
    results = []
    starts = sorted({0, 2048, ((action_count - 1) // 2048) * 2048})
    for index, start in enumerate(starts, 1):
        size = min(2048, action_count - start)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        # Repeated real trajectories deliberately exercise the full batch of four
        # without replacing any training implementation with a stand-in.
        logits, labels = supervised.segment_logits(policy, demos, [(0, start, size)] * 4)
        require(tuple(logits.shape[:2]) == (4, size), 'Policy did not process the requested full batch')
        require(int((labels != -100).sum()) == 4 * size, 'Unexpected padded labels in the full batch')
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        require(bool(torch.isfinite(loss)), f'Nonfinite loss for the window starting at {start}')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        steps = optimizer_steps()
        require(steps and set(steps) == {initial_step + index}, 'Optimizer state did not advance from the saved checkpoint')
        require(all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()),
                f'Nonfinite model parameters after window starting at {start}')
        record = dict(start=start, prefix_length=start, unroll_length=size, batch_size=4,
                      supervised_actions=4 * size, loss=float(loss.detach()),
                      gradient_norm=float(norm), optimizer_step=steps[0],
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        require(math.isfinite(record['gradient_norm']), 'Nonfinite gradient norm')
        results.append(record)
        print(json.dumps(dict(event='gpu_policy_window', **record)), flush=True)
        del logits, labels, loss
    after = {name: classifier_module.file_sha256(path) for name, path in paths.items()}
    require(before == after, 'An input archive or checkpoint changed during the test')
    output = dict(status='passed', device=device, device_name=torch.cuda.get_device_name(0),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  source_archive=str(paths['archive']), source_episode_id=source['id'],
                  source_actions=action_count, source_frames=len(source['frames']),
                  source_anchors=len(demos[0]['goals']), classifier=str(paths['classifier']),
                  policy=str(paths['policy']), initial_optimizer_step=initial_step,
                  final_optimizer_step=optimizer_steps()[0], windows=results,
                  inputs_unchanged=True, input_sha256_before=before, input_sha256_after=after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(output, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--classifier', required=True, type=Path)
    parser.add_argument('--policy', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
