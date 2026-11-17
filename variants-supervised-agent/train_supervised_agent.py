"""Train/evaluate supervised anchor-conditioned BC using a trajectory pickle.

The only local dependency is the sibling train_gru_classifier.py. Training is
offline cross-entropy; evaluation uses independent Atari starts and a frozen
classifier, similarity-based anchor LCS, and complete environment episodes.
"""

import argparse
import collections
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import secrets
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.checkpoint import checkpoint as recompute

from train_gru_classifier import (copy_model_state, encode_trajectory, file_sha256, frames_to_tensor,
                                  load_archive, load_classifier, make_encoder,
                                  model_fingerprint)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class NoUsableAnchorsError(ValueError):
    """No source episode provides a goal for supervised return training."""


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


@torch.no_grad()
def extract_anchor_indices(classifier, contexts, threshold):
    """Select changes relative to the last anchor, preserving source GRU history."""
    indices, anchor = [], 0
    for step in range(1, len(contexts)):
        probability = classifier.pair_logits(contexts[anchor:anchor+1], contexts[step:step+1]).sigmoid()
        if float(probability) < threshold:
            indices.append(step)
            anchor = step
    return indices


@torch.no_grad()
def prepare_demonstrations(episodes, ids, classifier, threshold, device='cpu', skip_empty=False):
    """Extract full anchor lists; action cloning does not use progress labels."""
    by_id = {episode['id']: episode for episode in episodes}
    demos = []
    classifier.eval().requires_grad_(False)
    for episode_id in ids:
        source = by_id[episode_id]
        frames, actions = source['frames'], source.get('actions')
        require(actions is not None and len(actions) == len(frames) - 1,
                f'Episode {episode_id}: supervised cloning requires recorded aligned actions')
        contexts = encode_trajectory(classifier, frames, device=device, batch_size=1).to(device)
        indices = extract_anchor_indices(classifier, contexts, threshold)
        # No synthetic terminal anchor is inserted to hide a degenerate classifier.
        if not indices and skip_empty:
            print(json.dumps(dict(event='skipped_empty_anchors', episode_id=episode_id,
                                  actions=len(actions))), flush=True)
            continue
        require(indices, f'Episode {episode_id} has no anchors: improve the classifier before using this task')
        demos.append(dict(id=episode_id, frames=frames, actions=np.asarray(actions, dtype=np.int64),
                          goals=frames[indices], references=contexts[indices].cpu().numpy(),
                          anchor_indices=np.asarray(indices),
                          reset_seed=source.get('reset_seed', source.get('seed'))))
        print(json.dumps(dict(event='anchors', episode_id=episode_id, anchors=len(indices),
                              actions=len(actions))), flush=True)
    if not demos:
        raise NoUsableAnchorsError('No usable anchor tasks; the classifier is not ready for policy training/evaluation')
    return demos


class GoalPolicy(nn.Module):
    """Separate policy CNN, ordered anchor GRU, controller GRU and attention."""
    def __init__(self, num_actions, hidden_dim=256, attention_heads=4):
        super().__init__()
        require(num_actions > 0 and hidden_dim > 0 and attention_heads > 0
                and hidden_dim % attention_heads == 0, 'Invalid policy dimensions')
        self.model_config = dict(num_actions=num_actions, hidden_dim=hidden_dim, attention_heads=attention_heads)
        self.num_actions, self.hidden_dim, self.heads = num_actions, hidden_dim, attention_heads
        self.vision = make_encoder('small', hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)
        self.goal_norm = nn.LayerNorm(hidden_dim)
        self.anchor_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.controller = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
        self.actor = nn.Linear(hidden_dim, num_actions)

    @property
    def device(self):
        return self.actor.weight.device

    def encode_images(self, frames):
        """Chunk visual work, avoiding an episode-sized float RGB GPU allocation."""
        raw = torch.as_tensor(np.asarray(frames))
        require(raw.dtype == torch.uint8 and raw.ndim == 4 and tuple(raw.shape[1:]) == (84, 84, 3),
                'Policy observations must be uint8 RGB84')
        if not len(raw):
            return self.actor.weight.new_empty((0, self.hidden_dim))

        def encode(chunk):
            image = chunk.to(self.device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous() / 255.
            return self.visual_norm(self.vision(image))

        return torch.cat([recompute(encode, chunk, use_reentrant=False)
                          if torch.is_grad_enabled() and len(raw) > 128 else encode(chunk)
                          for chunk in raw.split(128)])

    def encode_goals(self, goals):
        lengths = torch.tensor([len(g) for g in goals], device=self.device)
        require(bool((lengths > 0).all()), 'Every task needs at least one anchor')
        features = self.encode_images(np.concatenate(goals))
        features = pad_sequence(list(features.split(lengths.tolist())), batch_first=True)
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        memory, _ = self.anchor_gru(packed)
        memory, _ = pad_packed_sequence(memory, batch_first=True, total_length=features.shape[1])
        valid = torch.arange(features.shape[1], device=self.device)[None] < lengths[:, None]
        return self.goal_norm(memory + features) * valid[..., None], lengths

    def recurrent(self, inputs, sizes, hidden=None):
        batch, steps, _ = inputs.shape
        hidden = inputs.new_zeros((batch, self.hidden_dim)) if hidden is None else hidden
        sizes = torch.as_tensor(sizes, device=self.device, dtype=torch.long)
        require(sizes.shape == (batch,) and bool(((sizes >= 0) & (sizes <= steps)).all()), 'Invalid prefix lengths')
        states = inputs.new_zeros((batch, steps, self.hidden_dim))
        active = (sizes > 0).nonzero(as_tuple=False).flatten()
        if not len(active):
            return states, hidden
        packed = pack_padded_sequence(inputs[active], sizes[active].cpu(), batch_first=True, enforce_sorted=False)
        encoded, final = self.controller(packed, hidden[active][None])
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=steps)
        return states.index_copy(0, active, encoded), hidden.index_copy(0, active, final[0])

    def unroll(self, features, memory, lengths, sizes, hidden=None):
        batch, _, _ = features.shape
        require(memory.ndim == 3 and memory.shape[0] == batch and lengths.shape == (batch,),
                'Invalid anchor memory shapes')
        require(bool(((lengths > 0) & (lengths <= memory.shape[1])).all()), 'Invalid anchor lengths')
        states, hidden = self.recurrent(features, sizes, hidden)
        # All actual anchors remain visible at every step; only padding is masked.
        padding = torch.arange(memory.shape[1], device=self.device)[None] >= lengths[:, None]
        context, _ = self.attention(states, memory, memory, key_padding_mask=padding, need_weights=False)
        logits = self.actor(self.fusion(torch.cat((states, context), -1)))
        return logits, hidden

    def step(self, frame, memory, lengths, hidden=None):
        features = self.encode_images(frame[None])[None]
        logits, hidden = self.unroll(features, memory, lengths, [1], hidden)
        return logits[0, 0], hidden


def segment_logits(policy, demos, segments):
    """Reconstruct current-weight observation prefixes; source actions are labels only."""
    require(bool(segments), 'Empty training batch')
    selected, starts, sizes = [], [], []
    for index, start, size in segments:
        demo = demos[index]
        require(0 <= start < start + size <= len(demo['actions']), 'Invalid action window')
        selected.append(demo)
        starts.append(start)
        sizes.append(size)
    memory, lengths = policy.encode_goals([d['goals'] for d in selected])

    def features_for(spans):
        counts = [len(span) for span in spans]
        if not sum(counts):
            return policy.actor.weight.new_empty((len(spans), 0, policy.hidden_dim))
        encoded = policy.encode_images(np.concatenate(spans))
        return pad_sequence(list(encoded.split(counts)), batch_first=True)

    with torch.no_grad():
        prefixes = features_for([d['frames'][:start] for d, start in zip(selected, starts)])
        _, hidden = policy.recurrent(prefixes, starts)
    features = features_for([d['frames'][start:start+size] for d, start, size in zip(selected, starts, sizes)])
    labels = torch.full(features.shape[:2], -100, dtype=torch.long, device=policy.device)
    for row, (d, start, size) in enumerate(zip(selected, starts, sizes)):
        labels[row, :size] = torch.tensor(d['actions'][start:start+size], device=policy.device)
    logits, _ = policy.unroll(features, memory, lengths, sizes, hidden.detach())
    return logits, labels


@torch.no_grad()
def score_training(policy, demos, unroll_length):
    policy.eval()
    total = correct = 0
    loss = 0.
    for index, demo in enumerate(demos):
        for start in range(0, len(demo['actions']), unroll_length):
            size = min(unroll_length, len(demo['actions']) - start)
            logits, labels = segment_logits(policy, demos, [(index, start, size)])
            loss += float(F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='sum'))
            correct += int((logits.argmax(-1) == labels).sum())
            total += labels.numel()
    return dict(actions=total, action_accuracy=correct / total, loss=loss / total,
                scope='Teacher-forced training observations; not rollout anchor LCS')


def load_inputs(archive, classifier_path, device):
    classifier, checkpoint = load_classifier(classifier_path, device)
    require(checkpoint['archive_sha256'] == file_sha256(archive), 'Classifier was trained on a different archive')
    episodes, metadata = load_archive(archive)
    train_ids, test_ids = checkpoint['training_episode_ids'], checkpoint['evaluation_episode_ids']
    require(set(train_ids).isdisjoint(test_ids)
            and set(train_ids + test_ids) == {e['id'] for e in episodes}, 'Classifier split does not cover this archive')
    return classifier, checkpoint, episodes, metadata


@dataclass
class PolicyTrainingState:
    """Live policy and optimizer carried between online return cycles."""

    policy: GoalPolicy
    optimizer: torch.optim.Optimizer
    metadata: dict
    update: int
    checkpoint_path: Path
    checkpoint_sha256: str

    @property
    def total_updates(self):
        return self.metadata.get('previous_training_updates', 0) + self.update

    def to(self, device):
        """Offload idle state or restore it without recreating the optimizer."""
        self.policy.to(device)
        for group in self.optimizer.param_groups:
            device_step = group.get('capturable', False) or group.get('fused', False)
            for parameter in group['params']:
                for name, value in self.optimizer.state[parameter].items():
                    if torch.is_tensor(value):
                        target = device if name != 'step' or device_step else 'cpu'
                        self.optimizer.state[parameter][name] = value.to(target)
        return self


def train(args):
    """Standalone archive/checkpoint adapter around the in-process trainer."""
    classifier, checkpoint, episodes, metadata = load_inputs(args.archive, args.classifier, args.device)
    demos = prepare_demonstrations(episodes, checkpoint['training_episode_ids'], classifier,
                                  checkpoint['threshold'], args.device,
                                  skip_empty=getattr(args, 'skip_empty_anchors', False))
    return train_from_demonstrations(args, classifier, checkpoint, metadata, demos,
                                      file_sha256(args.classifier))


def train_from_demonstrations(args, classifier, classifier_ckpt, metadata, demos,
                              classifier_sha256, state=None):
    """Train on shared demonstrations, continuing live weights and Adam state.

    A standalone initial checkpoint may be loaded once. Subsequent online
    rounds use ``state`` directly and never reload models or trajectory data.
    """
    finetune = getattr(args, 'finetune', None)
    require(not (args.resume and finetune), '--resume and --finetune are mutually exclusive')
    require(state is None or not (args.resume or finetune),
            'Use either live training state or a checkpoint continuation')
    require(args.updates > 0 and args.batch_size > 0 and args.unroll_length > 0 and args.score_every > 0,
            'Training counts must be positive')
    require(math.isfinite(args.learning_rate) and args.learning_rate > 0, 'Learning rate must be positive')
    output = Path(args.output)
    require(args.resume is not None or not output.exists() or not any(output.iterdir()), 'Use a fresh output directory')
    if args.resume:
        require(Path(args.resume).resolve() == (output / 'latest.pt').resolve()
                and (output / 'best.pt').is_file(),
                'Resume with --resume OUTPUT/latest.pt --output OUTPUT, retaining its best.pt')
    output.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31 - 1)
    require(0 <= seed < 2**32, 'Optimizer seed must lie in [0, 2**32)')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    count = args.num_actions if args.num_actions is not None else metadata.get('num_actions')
    require(type(count) is int and count > 0, 'Archive needs num_actions metadata, or provide --num-actions')
    require(bool(demos), 'Need at least one usable demonstration')
    require(all(np.all((d['actions'] >= 0) & (d['actions'] < count)) for d in demos), 'Action label outside action space')
    if state is None:
        policy = GoalPolicy(count, args.hidden_dim, args.attention_heads).to(args.device)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=0.)
    else:
        require(state.policy.model_config == dict(num_actions=count, hidden_dim=args.hidden_dim,
                                                  attention_heads=args.attention_heads),
                'Live policy has another model configuration')
        state.to(args.device)
        policy, optimizer = state.policy, state.optimizer
    segments = [(i, s, min(args.unroll_length, len(d['actions']) - s)) for i, d in enumerate(demos)
                for s in range(0, len(d['actions']), args.unroll_length)]
    order, cursor, update, labels_seen, best = rng.permutation(len(segments)).tolist(), 0, 0, 0, None
    provenance = dict(archive_sha256=classifier_ckpt['archive_sha256'], classifier_sha256=classifier_sha256,
        classifier_fingerprint=model_fingerprint(classifier), threshold=classifier_ckpt['threshold'],
        training_episode_ids=classifier_ckpt['training_episode_ids'], evaluation_episode_ids=classifier_ckpt['evaluation_episode_ids'],
        usable_training_episode_ids=[d['id'] for d in demos], env_config=metadata.get('env_config'),
        skipped_empty_episode_ids=sorted(set(classifier_ckpt['training_episode_ids']) - {d['id'] for d in demos}),
        action_meanings=metadata.get('action_meanings'), seed=seed,
        optimizer_continuation='memory' if state is not None else ('checkpoint' if finetune or args.resume else 'new'),
        training_config=dict(batch_size=args.batch_size, unroll_length=args.unroll_length, learning_rate=args.learning_rate))
    if finetune or state is not None:
        if state is None:
            saved = torch.load(finetune, map_location='cpu', weights_only=True)
            require(saved.get('kind') == 'standalone_goal_bc' and saved.get('format_version') == 2,
                    'Fine-tuning needs a version 2 all-anchor policy checkpoint')
            require(saved['model_config'] == policy.model_config, 'Fine-tune checkpoint has another model configuration')
            previous = saved['metadata']
            policy.load_state_dict(saved['model_state'])
            optimizer.load_state_dict(saved['optimizer_state'])
            previous_updates = previous.get('previous_training_updates', 0) + saved['update']
            previous_path, previous_hash = Path(finetune).resolve(), file_sha256(finetune)
            del saved
        else:
            previous = state.metadata
            previous_updates = state.total_updates
            previous_path, previous_hash = state.checkpoint_path, state.checkpoint_sha256
        if previous.get('action_meanings') is not None and metadata.get('action_meanings') is not None:
            require(previous['action_meanings'] == metadata['action_meanings'], 'Fine-tuning changed the action mapping')
        previous_task = (previous.get('env_config') or {}).get('task')
        current_task = (metadata.get('env_config') or {}).get('task')
        require(previous_task is None or current_task is None or previous_task == current_task,
                'Fine-tuning changed the environment task')
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate
        provenance.update(finetuned_from=str(previous_path),
                          finetuned_from_sha256=previous_hash,
                          previous_training_updates=previous_updates)
        # A new dataset/classifier gets a fresh sampling schedule and fit score.
        # Model parameters and Adam moments retain their learned values.
    best_weights = None
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        require(saved.get('kind') == 'standalone_goal_bc' and saved.get('format_version') == 2,
                'Expected a version 2 all-anchor policy checkpoint; old progress-conditioned policies require retraining')
        require(saved['model_config'] == policy.model_config, 'Resume checkpoint has another model configuration')
        for key in ('archive_sha256', 'classifier_sha256', 'threshold', 'training_config', 'usable_training_episode_ids'):
            require(saved['metadata'][key] == provenance[key], f'Resume changed {key}')
        policy.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        update, labels_seen, best = saved['update'], saved['labels_seen'], saved['best_score']
        order, cursor = saved['schedule']['order'], saved['schedule']['cursor']
        rng.bit_generator.state = saved['schedule']['rng']
        torch.set_rng_state(saved['torch_rng_state'])
        if args.device.startswith('cuda') and saved.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state(saved['cuda_rng_state'], args.device)
        provenance = saved['metadata']
        require(update <= args.updates, '--updates is the total target, including resumed steps')
        require(sorted(order) == list(range(len(segments))) and 0 <= cursor <= len(order), 'Invalid resumed schedule')
        log_path = output / 'training.jsonl'
        rows = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
        retained = [row for row in rows if row['update'] <= update]
        require([row['update'] for row in retained] == list(range(1, update + 1)),
                'Training log does not cover the resumed checkpoint')
        if len(rows) != len(retained):
            log_path.rename(output / f'training.interrupted_{time.time_ns()}.jsonl')
            log_path.write_text(''.join(json.dumps(row) + '\n' for row in retained))
        # Standalone resume can retain a best fit from before its latest state.
        best_weights = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)['model_state']
        provenance['optimizer_continuation'] = 'checkpoint'

    def payload():
        return dict(kind='standalone_goal_bc', format_version=2, model_config=policy.model_config,
                    model_state=policy.state_dict(), optimizer_state=optimizer.state_dict(),
                    metadata=provenance, update=update, labels_seen=labels_seen, best_score=best,
                    schedule=dict(order=order, cursor=cursor, rng=rng.bit_generator.state),
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state(args.device) if args.device.startswith('cuda') else None)

    def assess():
        nonlocal best, best_weights
        result = score_training(policy, demos, args.unroll_length)
        quality = [result['action_accuracy'], -result['loss']]
        if best is None or tuple(quality) > tuple(best):
            best = quality
            save_checkpoint(output / 'best.pt', payload())
            best_weights = copy_model_state(policy)
        print(json.dumps(dict(event='train_fit', update=update, **result)), flush=True)
        save_checkpoint(output / 'latest.pt', payload())
        return result

    save_json(output / 'config.json', provenance | dict(model_config=policy.model_config))
    started = time.monotonic()
    fit = assess()
    with (output / 'training.jsonl').open('a') as stream:
        while update < args.updates:
            batch = []
            for _ in range(args.batch_size):
                if cursor == len(order):
                    order, cursor = rng.permutation(len(segments)).tolist(), 0
                batch.append(segments[order[cursor]])
                cursor += 1
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            logits, labels = segment_logits(policy, demos, batch)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
            require(bool(torch.isfinite(loss)), 'Nonfinite supervised loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(policy.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            update += 1
            valid = labels != -100
            labels_seen += int(valid.sum())
            row = dict(update=update, loss=float(loss.detach()), actions=int(valid.sum()),
                       action_accuracy=float((logits.argmax(-1)[valid] == labels[valid]).float().mean()),
                       gradient_norm=float(norm), labels_seen=labels_seen, elapsed_seconds=time.monotonic()-started)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            if update % 100 == 0 or update == args.updates:
                print(json.dumps(row), flush=True)
            if update % args.score_every == 0 or update == args.updates:
                fit = assess()
    require(model_fingerprint(classifier) == provenance['classifier_fingerprint'], 'Training altered the classifier')
    save_json(output / 'training_report.json', dict(update=update, labels_seen=labels_seen, train_fit=fit,
              best_score=best, training_environment_steps=0, heldout_labels_used=False,
              classifier_unchanged=True, checkpoint=str(output / 'best.pt')))
    optimizer.zero_grad(set_to_none=True)
    # Rollouts use the best fit, while future updates continue the latest fit.
    rollout_policy = copy.deepcopy(policy)
    rollout_policy.load_state_dict(best_weights)
    rollout_policy.eval().requires_grad_(False)
    checkpoint_path = (output / 'latest.pt').resolve()
    if state is None:
        state = PolicyTrainingState(policy, optimizer, provenance, update,
                                    checkpoint_path, file_sha256(checkpoint_path))
    else:
        state.metadata, state.update = provenance, update
        state.checkpoint_path, state.checkpoint_sha256 = checkpoint_path, file_sha256(checkpoint_path)
    return state, rollout_policy


class AtariEnv:
    """The archive's ALE action repeat/pooling/resize conventions, kept local."""
    def __init__(self, config):
        import ale_py
        import gymnasium as gym
        gym.register_envs(ale_py)
        self.config = dict(config)
        require(config.get('lives', 'unused') == 'unused', 'This standalone evaluator uses complete episodes (lives=unused)')
        task = config['task']
        require(task.startswith('atari_'), 'Evaluation currently supports Atari archives')
        games = {s.kwargs['game'].replace('_', '').lower(): key for key, s in gym.registry.items()
                 if key.startswith('ALE/') and key.endswith('-v5') and 'game' in s.kwargs}
        game = task[len('atari_'):].replace('_', '').lower()
        game = 'jamesbond' if game == 'jamesbond' else game
        self.repeat = int(config.get('action_repeat', 4))
        self.limit = int(config.get('time_limit', 108000))
        self.noops = int(config.get('noops', 30))
        self.size = tuple(config.get('size', (84, 84)))
        pooling = int(config.get('pooling', 2))
        require(self.repeat > 0 and self.limit > 0 and self.noops >= 0 and pooling > 0
                and len(self.size) == 2 and self.size[0] == self.size[1], 'Invalid Atari configuration')
        require(config.get('actions', 'needed') in ('all', 'needed')
                and config.get('aggregate', 'max') in ('max', 'mean')
                and config.get('resize', 'pillow') in ('pillow', 'opencv'), 'Unsupported Atari preprocessing')
        self.env = gym.make(games[game], obs_type='rgb', frameskip=1, mode=None, difficulty=None,
                            repeat_action_probability=.25 if config.get('sticky', False) else 0.,
                            full_action_space=config.get('actions', 'needed') == 'all',
                            max_num_frames_per_episode=0, render_mode=None)
        self.action_meanings = self.env.unwrapped.get_action_meanings()
        self.num_actions = self.env.action_space.n
        self.ale = self.env.unwrapped.ale
        self.buffers = collections.deque([np.zeros_like(self.ale.getScreenRGB()) for _ in range(pooling)], maxlen=pooling)
        self.steps = 0

    def _render(self):
        self.buffers.appendleft(self.buffers.pop())
        self.ale.getScreenRGB(self.buffers[0])

    def _image(self):
        image = (np.amax(self.buffers, 0) if self.config.get('aggregate', 'max') == 'max'
                 else np.mean(self.buffers, 0).astype(np.uint8))
        if image.shape[:2] != self.size:
            if self.config.get('resize', 'pillow') == 'opencv':
                import cv2
                image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
            else:
                image = np.asarray(Image.fromarray(image).resize(self.size, Image.Resampling.BILINEAR))
        if self.config.get('gray', False):
            image = (image * np.array([.299, .587, 1-(.299+.587)])).sum(-1).astype(np.uint8)
            image = np.repeat(image[..., None], 3, -1)
        if image.shape[:2] != (84, 84):
            image = np.asarray(Image.fromarray(image).resize((84, 84), Image.Resampling.BILINEAR))
        return np.array(image, dtype=np.uint8, copy=True)

    def reset(self, seed):
        self.env.reset(seed=int(seed))
        rng = np.random.default_rng(seed)
        for _ in range(rng.integers(self.noops + 1)):
            _, _, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated:
                self.env.reset()
        if self.config.get('autostart', False) and 'FIRE' in self.action_meanings:
            for name in ('FIRE', 'UP'):
                if name in self.action_meanings:
                    _, _, terminated, truncated, _ = self.env.step(self.action_meanings.index(name))
                    if terminated or truncated:
                        self.env.reset()
        self._render()
        for i in range(1, len(self.buffers)):
            np.copyto(self.buffers[i], self.buffers[0])
        self.steps = 0
        return self._image()

    def step(self, action):
        require(0 <= int(action) < self.num_actions, 'Action outside environment action space')
        reward = 0.
        terminated = truncated = False
        for repeat in range(self.repeat):
            _, value, terminal, truncation, _ = self.env.step(int(action))
            self.steps += 1
            reward += float(value)
            if repeat >= self.repeat - len(self.buffers):
                self._render()
            terminated |= bool(terminal)
            truncated |= bool(truncation or self.steps >= self.limit)
            if terminated or truncated:
                break
        return self._image(), reward, terminated, truncated and not terminated

    def close(self):
        self.env.close()


def video_writer(path, fps):
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (84, 84))
    require(writer.isOpened(), 'Could not create RGB84 video')
    return writer


@torch.no_grad()
def similarity_lcs_length(classifier, rollout_anchors, target_anchors, threshold, pair_batch_size=1024):
    """Maximum ordered one-to-one matches under the classifier's similarity rule.

    Both sequences contain contextual embeddings from their own complete reset
    histories. Skips are allowed in either sequence; repeats cannot reuse an
    anchor. Only one DP row and one bounded batch of pairs are kept in memory.
    """
    require(rollout_anchors.ndim == target_anchors.ndim == 2
            and rollout_anchors.shape[1] == target_anchors.shape[1], 'Invalid anchor embedding shapes')
    require(math.isfinite(threshold) and 0 <= threshold <= 1 and pair_batch_size > 0,
            'Invalid similarity threshold or pair batch size')
    previous = np.zeros(len(target_anchors) + 1, dtype=np.int64)
    if not len(target_anchors):
        return 0
    for anchor in rollout_anchors:
        matches = []
        for targets in target_anchors.split(pair_batch_size):
            probability = classifier.pair_logits(anchor[None].expand_as(targets), targets).sigmoid()
            matches.append((probability >= threshold).reshape(-1).cpu().numpy())
        matches = np.concatenate(matches)
        # D[i,j] = max(D[i-1,j], D[i,j-1], D[i-1,j-1] + match(i,j)).
        # For a nonmatch the diagonal cannot improve on the skip candidates.
        previous[1:] = np.maximum.accumulate(np.maximum(previous[1:], previous[:-1] + matches))
    return int(previous[-1])


@torch.no_grad()
def eval_one(policy, classifier, demo, env, seed, threshold, device, video_path=None):
    """Roll out a complete episode, extract its anchors, and score anchor LCS."""
    policy.eval()
    classifier.eval()
    frame = env.reset(seed)
    source_initial_image_matches = bool(np.array_equal(frame, demo['frames'][0]))
    memory, lengths = policy.encode_goals([demo['goals']])
    reference = torch.as_tensor(demo['references'], device=device)
    extraction_reference, classifier_hidden = classifier.encode_step(frames_to_tensor(frame[None], device), None)
    hidden, game_return = None, 0.
    actions, rollout_anchors, rollout_indices = [], [], []
    writer = None if video_path is None else video_writer(video_path, 60 / env.repeat)
    if writer is not None:
        writer.write(frame[..., ::-1])
    try:
        for step in range(math.ceil(env.limit / env.repeat)):
            logits, hidden = policy.step(frame, memory, lengths, hidden)
            action = int(logits.argmax())
            #action = int(torch.distributions.Categorical(logits=logits).sample())
            frame, reward, terminated, truncated = env.step(action)
            current, classifier_hidden = classifier.encode_step(frames_to_tensor(frame[None], device), classifier_hidden)
            probability = classifier.pair_logits(extraction_reference, current).sigmoid()
            if float(probability) < threshold:
                rollout_anchors.append(current.cpu())
                rollout_indices.append(step + 1)
                extraction_reference = current
            actions.append(action)
            game_return += reward
            if writer is not None:
                writer.write(frame[..., ::-1])
            if terminated or truncated:
                break
        require(terminated or truncated, 'Evaluation ended before environment termination/time limit')
    finally:
        if writer is not None:
            writer.release()
    rollout = (torch.cat(rollout_anchors).to(device) if rollout_anchors
               else reference.new_empty((0, reference.shape[1])))
    lcs_length = similarity_lcs_length(classifier, rollout, reference, threshold)
    return dict(episode_id=demo['id'], reset_seed=seed, lcs_length=lcs_length,
                rollout_anchor_count=len(rollout), target_anchor_count=len(reference),
                rollout_anchor_indices=rollout_indices,
                target_anchor_indices=np.asarray(demo['anchor_indices']).tolist(),
                env_steps=len(actions), env_return=game_return, terminated=bool(terminated),
                environment_truncated=bool(truncated), episode_end_reached=True,
                metric='similarity_lcs_length', similarity_threshold=threshold, actions=actions,
                source_initial_image_matches=source_initial_image_matches,
                source_reset_seed_reused=(None if demo.get('reset_seed') is None
                                          else int(seed) == int(demo['reset_seed'])))


def evaluate(args):
    output = Path(args.output)
    require(not output.exists() or not any(output.iterdir()), 'Use a fresh evaluation directory')
    classifier, classifier_ckpt, episodes, archive_meta = load_inputs(args.archive, args.classifier, args.device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    require(saved.get('kind') == 'standalone_goal_bc', 'Expected a checkpoint from this supervised trainer')
    require(saved.get('format_version') == 2,
            'This evaluator requires a version 2 policy without explicit progress inputs; retrain the agent')
    meta = saved['metadata']
    require(meta['archive_sha256'] == classifier_ckpt['archive_sha256']
            and meta['classifier_sha256'] == file_sha256(args.classifier)
            and meta['threshold'] == classifier_ckpt['threshold']
            and meta['classifier_fingerprint'] == model_fingerprint(classifier), 'Classifier/archive differs from training')
    require(meta['training_episode_ids'] == classifier_ckpt['training_episode_ids']
            and meta['evaluation_episode_ids'] == classifier_ckpt['evaluation_episode_ids'], 'Training/evaluation split changed')
    config = json.loads(args.env_config) if args.env_config else meta['env_config']
    require(isinstance(config, dict), 'Evaluation needs archive metadata.env_config or --env-config JSON')
    require(meta['env_config'] is None or config == meta['env_config'],
            'Evaluation preprocessing must match the archived environment configuration')
    require(args.episodes >= 0 and args.repeats >= 1, 'Invalid evaluation count')
    policy = GoalPolicy(**saved['model_config']).to(args.device)
    policy.load_state_dict(saved['model_state'])
    policy.eval()
    policy_before = model_fingerprint(policy)
    classifier_before = model_fingerprint(classifier)
    output.mkdir(parents=True, exist_ok=True)
    env = AtariEnv(config)
    try:
        require(env.num_actions == policy.num_actions, 'Evaluation environment action count differs')
        if meta.get('action_meanings') is not None:
            require(list(meta['action_meanings']) == list(env.action_meanings), 'Environment action meanings differ')
        used = set()
        for episode in episodes:
            for name in ('seed', 'reset_seed', 'sampling_seed'):
                if episode.get(name) is not None:
                    used.add(int(episode[name]))
        for row in archive_meta.get('seed_plan', []):
            used.update(int(row[k]) for k in ('reset_seed', 'sampling_seed') if k in row)
        selected, plan = {}, []
        for split in (('train', 'test') if args.split == 'both' else (args.split,)):
            ids = meta['training_episode_ids' if split == 'train' else 'evaluation_episode_ids']
            if args.episodes:
                ids = ids[-args.episodes:]
            selected[split] = prepare_demonstrations(episodes, ids, classifier, meta['threshold'], args.device)
            for demo in selected[split]:
                for repetition in range(args.repeats):
                    for _ in range(10000):
                        seed = secrets.randbelow(2**31 - 1)
                        if seed in used:
                            continue
                        used.add(seed)
                        break
                    else:
                        raise ValueError('Could not obtain an unused evaluation seed')
                    # Record the normal startup draw without conditioning seed
                    # selection on its NOOP count or resulting initial image.
                    count = int(np.random.default_rng(seed).integers(env.noops + 1))
                    plan.append(dict(split=split, episode_id=demo['id'], repetition=repetition,
                                     reset_seed=seed, reset_noops=count))
        save_json(output / 'plan.json', dict(episodes=plan, checkpoint_sha256=file_sha256(args.checkpoint),
                  classifier_sha256=file_sha256(args.classifier), env_config=config,
                  full_environment_episodes=True, reset_sampling='independent_seeds',
                  metric='similarity_lcs_length', similarity_threshold=meta['threshold'],
                  anchor_extraction='first_dissimilar_from_previous_anchor_excluding_frame_zero',
                  fresh_changed_starts=False))
        by_key = {(split, d['id']): d for split, demos in selected.items() for d in demos}
        records = []
        for index, row in enumerate(plan):
            demo = by_key[(row['split'], row['episode_id'])]
            directory = output / f"{row['split']}_{row['episode_id']}_{row['repetition']}"
            directory.mkdir()
            if args.save_videos:
                anchors = directory / 'anchors'
                anchors.mkdir()
                for k, frame in enumerate(demo['goals']):
                    Image.fromarray(frame).save(anchors / f'{k:04d}.png')
                writer = video_writer(directory / 'source.mp4', 60 / env.repeat)
                try:
                    for frame in demo['frames']:
                        writer.write(frame[..., ::-1])
                finally:
                    writer.release()
            result = eval_one(policy, classifier, demo, env, row['reset_seed'], meta['threshold'], args.device,
                              directory / 'rollout.mp4' if args.save_videos else None)
            result.update(split=row['split'], repetition=row['repetition'])
            save_json(directory / 'result.json', result)
            records.append(result)
            print(json.dumps({k: v for k, v in result.items()
                              if k not in ('actions', 'rollout_anchor_indices', 'target_anchor_indices')}), flush=True)
        summaries = {}
        for split in selected:
            rows = [r for r in records if r['split'] == split]
            summaries[split] = dict(episodes=len(rows),
                mean_lcs_length=float(np.mean([r['lcs_length'] for r in rows])),
                mean_rollout_anchor_count=float(np.mean([r['rollout_anchor_count'] for r in rows])),
                mean_target_anchor_count=float(np.mean([r['target_anchor_count'] for r in rows])),
                mean_env_return=float(np.mean([r['env_return'] for r in rows])))
        require(model_fingerprint(policy) == policy_before and model_fingerprint(classifier) == classifier_before,
                'Evaluation modified model weights')
        save_json(output / 'report.json', dict(summary=summaries, episodes=records, training_updates=0,
                  classifier_unchanged=True, policy_unchanged=True, full_environment_episodes=True,
                  reset_sampling='independent_seeds',
                  metric='similarity_lcs_length', similarity_threshold=meta['threshold'],
                  anchor_extraction='first_dissimilar_from_previous_anchor_excluding_frame_zero',
                  limitation='Independent reset seeds do not guarantee different initial images or emulator states'))
        print(json.dumps(dict(event='evaluation_complete', summary=summaries)), flush=True)
    finally:
        env.close()


def parse_args(argv=None, *, require_inputs=True):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'evaluate'):
        sub = commands.add_parser(name)
        sub.add_argument('--archive', type=Path, required=require_inputs)
        sub.add_argument('--classifier', type=Path, required=require_inputs)
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--device', default='auto')
        sub.add_argument('--threads', type=int, default=4)
        if name == 'train':
            sub.add_argument('--updates', type=int, default=10000)
            sub.add_argument('--batch-size', type=int, default=4)
            sub.add_argument('--unroll-length', type=int, default=2048)
            sub.add_argument('--score-every', type=int, default=1000)
            sub.add_argument('--learning-rate', type=float, default=3e-4)
            sub.add_argument('--hidden-dim', type=int, default=256)
            sub.add_argument('--attention-heads', type=int, default=4)
            sub.add_argument('--num-actions', type=int)
            sub.add_argument('--seed', type=int, help='Optional optimizer seed; OS randomness by default')
            continuation = sub.add_mutually_exclusive_group()
            continuation.add_argument('--resume', type=Path)
            continuation.add_argument('--finetune', type=Path,
                                      help='Keep policy and optimizer weights; run --updates additional updates on new data')
            sub.add_argument('--skip-empty-anchors', action='store_true',
                             help='Skip and report episodes without anchors; fail if no goals remain')
        else:
            sub.add_argument('--checkpoint', type=Path, required=True)
            sub.add_argument('--split', choices=('train', 'test', 'both'), default='test')
            sub.add_argument('--episodes', type=int, default=0, help='Latest N goals per split; 0 means all (default)')
            sub.add_argument('--repeats', type=int, default=1)
            sub.add_argument('--env-config', help='Explicit Atari preprocessing config as a JSON object, if absent in archive')
            sub.add_argument('--save-videos', action='store_true', help='Save RGB84 source/rollout MP4s and separate anchor PNGs')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    require(args.threads > 0, '--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.device == 'auto':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    (train if args.command == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
