"""Check full-prefix classifier gradients and live optimizer moves on a GPU.

Uses one real 27,000-action trajectory as 100 logical feature sequences to
exercise the production 64-pair classifier batch without duplicating host RGB
data. Optional VRAM reservation is synthetic, not an actual resident Dreamer.
Input archives and model checkpoints are never overwritten.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--classifier', type=Path, required=True)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reserved-headroom-gib', type=float, default=0.)
    args = parser.parse_args()
    assert args.reserved_headroom_gib >= 0
    sys.path.insert(0, str(args.repo / 'variants-supervised-agent'))
    import numpy as np
    import torch
    from torch.nn import functional as F
    import train_gru_classifier as classifier_module
    import train_supervised_agent as supervised

    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    device = 'cuda:0'
    paths = dict(archive=args.archive, classifier=args.classifier)
    if args.policy:
        paths['policy'] = args.policy
    assert args.output.resolve() not in [path.resolve() for path in paths.values()]
    hashes = {name: classifier_module.file_sha256(path) for name, path in paths.items()}
    report = dict(status='running', device=device, device_name=torch.cuda.get_device_name(),
                  reserved_headroom_gib=args.reserved_headroom_gib,
                  reservation_kind='synthetic_memory_only_not_a_resident_Dreamer',
                  source_archive=str(args.archive.resolve()), input_sha256=hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    save()
    reservation = None
    try:
        if args.reserved_headroom_gib:
            reservation = torch.empty(int(args.reserved_headroom_gib * 1024**3),
                                      dtype=torch.uint8, device=device)
        episodes, metadata = classifier_module.load_archive(args.archive)
        source = max(episodes, key=lambda ep: len(ep['frames']))
        assert len(source['frames']) >= 27001
        del episodes
        classifier, checkpoint = classifier_module.load_classifier(args.classifier, device)
        report.update(source_episode_id=source['id'], source_frames=len(source['frames']))
        started = time.monotonic()
        features = classifier_module.cache_features(classifier, [source['frames']], device)[0]
        logical_episodes = [features] * 100
        sampler = classifier_module.PairSampler(logical_episodes, window=20, negative_ratio=1, seed=13)
        first, second, labels = sampler.sample(64)
        # Include the complete final prefix, while retaining the production pair count.
        first[0], second[0], labels[0] = sampler.total - 1, sampler.total - 2, 1.
        indices = np.concatenate((first, second))
        episode_ids, local = sampler.locate(indices)
        classifier.gru.requires_grad_(True)
        classifier.projection.requires_grad_(True)
        classifier.head.requires_grad_(True)
        classifier.train()
        optimizer = torch.optim.AdamW((p for p in classifier.parameters() if p.requires_grad),
                                      lr=1e-3, weight_decay=1e-4)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        embeddings = classifier_module.sampled_embeddings(classifier, logical_episodes,
                         sampler, indices, device, tbptt_chunk=0)
        logits = classifier.pair_logits(*embeddings.chunk(2))
        loss = classifier_module.balanced_loss(logits, torch.from_numpy(labels).to(device))
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        report['classifier_batch'] = dict(
            status='passed', logical_episodes=100, physical_feature_copies=1,
            batch_pairs=64, unique_sampled_episodes=len(np.unique(episode_ids)),
            longest_prefix=int(local.max()) + 1, tbptt_chunk=0, loss=float(loss.detach()),
            gradient_norm=float(norm), elapsed_seconds=time.monotonic() - started,
            gpu_baseline_allocated_bytes=baseline,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        print(json.dumps(dict(event='long_classifier_batch', **report['classifier_batch'])), flush=True)
        save()
        del optimizer, embeddings, logits, loss, logical_episodes, features, classifier
        gc.collect()
        torch.cuda.empty_cache()

        if args.policy:
            saved = torch.load(args.policy, map_location='cpu', weights_only=True)
            policy = supervised.GoalPolicy(**saved['model_config'])
            policy.load_state_dict(saved['model_state'])
            optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0.)
            optimizer.load_state_dict(saved['optimizer_state'])
            state = supervised.PolicyTrainingState(policy, optimizer, saved['metadata'],
                    saved['update'], args.policy.resolve(), hashes['policy'])
            parameter_ids = [id(parameter) for parameter in policy.parameters()]
            expected_model = classifier_module.copy_model_state(policy)
            expected_optimizer = {parameter: {key: value.clone() if torch.is_tensor(value) else value
                                  for key, value in values.items()}
                                  for parameter, values in optimizer.state.items()}
            initial_steps = {int(values['step']) for values in optimizer.state.values()}
            assert len(initial_steps) == 1
            state.to(device).to('cpu').to(device)
            assert [id(parameter) for parameter in policy.parameters()] == parameter_ids
            for name, value in policy.state_dict().items():
                assert torch.equal(value.cpu(), expected_model[name])
            for parameter, values in optimizer.state.items():
                for key, value in values.items():
                    expected = expected_optimizer[parameter][key]
                    assert torch.equal(value.cpu(), expected) if torch.is_tensor(value) else value == expected
            length = min(32, len(source['actions']))
            demo = dict(frames=source['frames'], actions=source['actions'],
                        goals=source['frames'][[length]])
            logits, labels = supervised.segment_logits(policy, [demo], [(0, 0, length)])
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            final_steps = {int(values['step']) for values in optimizer.state.values()}
            assert final_steps == {next(iter(initial_steps)) + 1}
            state.to('cpu')
            assert all(parameter.device.type == 'cpu' for parameter in policy.parameters())
            assert all(value.device.type == 'cpu' for values in optimizer.state.values()
                       for value in values.values() if torch.is_tensor(value))
            report['live_policy_roundtrip'] = dict(status='passed', loss=float(loss.detach()),
                  gradient_norm=float(norm), initial_optimizer_step=next(iter(initial_steps)),
                  final_optimizer_step=next(iter(final_steps)), final_device='cpu',
                  model_parameter_identity_preserved=True, optimizer_values_preserved=True)
        assert hashes == {name: classifier_module.file_sha256(path) for name, path in paths.items()}
        report.update(status='passed', inputs_unchanged=True)
        save()
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        report['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        save()
        raise


if __name__ == '__main__':
    main()
