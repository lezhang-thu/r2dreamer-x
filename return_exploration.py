"""Periodic first-return, then Dreamer exploration on completed trajectories.

The scheduling clock contains ordinary Dreamer interactions only. Auxiliary
rollouts pause learning and use a separate environment, preserving an unfinished
ordinary episode. Each completed auxiliary rollout is inserted into replay
before updating the frozen anchor bank's counters and sampling the next goal.
"""

import gc
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import random
import secrets
import shutil
import sys
import traceback

import numpy as np
from omegaconf import OmegaConf
import torch
from tensordict import TensorDict

from episode_archive import EpisodeArchive, EpisodeRecorder, resize_frame
from return_anchors import AnchorBank, classifier_similarity


VARIANT = Path(__file__).resolve().parent / 'variants-supervised-agent'


def variant_modules():
    if str(VARIANT) not in sys.path:
        sys.path.insert(0, str(VARIANT))
    import train_gru_classifier
    import train_supervised_agent
    return train_gru_classifier, train_supervised_agent


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def auxiliary_training_context(device):
    """Keep auxiliary seeds and CPU thread settings out of Dreamer's stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    threads = torch.get_num_threads()
    device = torch.device(device)
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == 'cuda' else [])
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_num_threads(threads)


def observation_batch(observation, reward, device):
    values = {key: torch.as_tensor(np.asarray(value), device=device).unsqueeze(0)
              for key, value in observation.items()}
    values['reward'] = torch.tensor([reward], dtype=torch.float32, device=device)
    return TensorDict({key: value.unsqueeze(-1) if value.ndim == 1 else value
                       for key, value in values.items()}, batch_size=(1,), device=device)


@torch.no_grad()
def collect_assisted_episode(agent, policy, classifier, env, goals, target,
                             threshold, segment, device):
    """Keep the same live env at handoff; condition RSSM on executed actions.

    Rows are kept on CPU until the episode is complete. Dreamer's weights do
    not change here, so observing its actual prefix incrementally is equivalent
    to warming its recurrent state on that prefix immediately before handoff.
    """
    classifier_module, _ = variant_modules()
    observation = env.reset()
    reward, done = 0., False
    recorder = EpisodeRecorder(observation['image'], source='assisted')
    memory, lengths = policy.encode_goals([goals])
    target = torch.as_tensor(np.array(target, copy=True), device=device)[None]
    state = agent.get_initial_state(1)
    classifier_hidden = policy_hidden = None
    extraction_reference = None
    rollout_anchors, anchor_indices, rows = [], [], []
    reached_at = None
    interactions, return_actions = 0, 0
    env_return = 0.
    while True:
        frame = resize_frame(observation['image'])
        current, classifier_hidden = classifier.encode_step(
            classifier_module.frames_to_tensor(frame[None], device), classifier_hidden)
        if extraction_reference is None:
            extraction_reference = current
        elif float(classifier.pair_logits(extraction_reference, current).sigmoid()) < threshold:
            rollout_anchors.append(current[0].cpu().numpy())
            anchor_indices.append(interactions)
            extraction_reference = current
        if reached_at is None and float(classifier.pair_logits(current, target).sigmoid()) >= threshold:
            reached_at = interactions

        transition = observation_batch(observation, reward, device)
        dreamer_action, state = agent.act(transition.clone(), state, eval=False)
        if done:
            action = torch.zeros_like(dreamer_action)
        elif reached_at is not None:
            action = dreamer_action
        else:
            logits, policy_hidden = policy.step(frame, memory, lengths, policy_hidden)
            action = torch.nn.functional.one_hot(logits.argmax(-1), policy.num_actions)
            action = action.to(device=device, dtype=dreamer_action.dtype)[None]
            return_actions += 1
        # Dreamer.act proposes an action; only the chosen action was executed.
        # Its RSSM must receive that action on the next observation.
        state['prev_action'] = action
        transition['action'] = action
        transition['stoch'] = state['stoch']
        transition['deter'] = state['deter']
        transition['episode'] = torch.tensor([segment], dtype=torch.int64, device=device)
        rows.append(transition.detach().to('cpu').clone())
        if done:
            break
        observation, reward, done, _ = env.step(action[0].cpu().numpy())
        env_return += float(reward)
        interactions += 1
        terminal = bool(observation['is_terminal'])
        recorder.step(int(action.argmax(-1).item()), observation['image'], reward,
                      terminated=terminal, truncated=bool(done and not terminal))
    references = (np.stack(rollout_anchors) if rollout_anchors else
                  np.empty((0, target.shape[-1]), dtype=np.float32))
    result = dict(env_return=env_return, interactions=interactions, return_actions=return_actions,
                  dreamer_actions=interactions - return_actions, reached_at=reached_at,
                  reached=reached_at is not None, takeover=reached_at is not None and reached_at < interactions,
                  anchor_indices=anchor_indices, rollout_anchor_count=len(references))
    return recorder.finish(), rows, references, result


class ReturnExploration:
    def __init__(self, config, env_config, logdir, num_actions, device, action_meanings=None):
        if int(env_config.env_num) != 1:
            raise ValueError('Return exploration currently requires env.env_num=1 for replay stream alignment')
        if env_config.gray or env_config.lives != 'unused':
            raise ValueError('Return exploration needs RGB observations and complete episodes (lives=unused)')
        self.config, self.env_config = config, env_config
        self.device = str(device)
        self.interval, self.episodes = int(config.interval), int(config.episodes)
        if self.interval < 1 or self.episodes < 1:
            raise ValueError('Return interval and episode count must be positive')
        self.root = Path(logdir).resolve() / 'return_exploration'
        if ((self.root.exists() and any(self.root.iterdir()))
                or (Path(logdir) / 'episode_archive' / 'manifest.json').exists()):
            raise ValueError('Use a fresh logdir; online replay/environment resume is not implemented. '
                             'Use return_exploration.initial_policy to warm-start a compatible return policy.')
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = dict(num_actions=int(num_actions),
                        env_config=OmegaConf.to_container(env_config, resolve=True))
        if action_meanings is not None:
            metadata['action_meanings'] = list(action_meanings)
        self.archive = EpisodeArchive(metadata)
        self.policy_state = None
        self._previous_directory = None
        self.recorder = None
        self.next_milestone = self.interval
        self.round = 0
        self.assisted_interactions = 0
        self.policy_checkpoint = (Path(config.initial_policy).expanduser().resolve()
                                  if config.initial_policy else None)
        if self.policy_checkpoint is not None and not self.policy_checkpoint.is_file():
            raise FileNotFoundError(self.policy_checkpoint)
        self.rng = np.random.default_rng(int(env_config.seed))

    def record_ordinary(self, transition, action, previous_done, done, interactions):
        """Called before action replacement; reset observations are not actions."""
        frame = transition['image'][0].cpu().numpy()
        if bool(previous_done[0]):
            if self.recorder is not None:
                raise RuntimeError('Cannot discard an unfinished ordinary trajectory')
            self.recorder = EpisodeRecorder(frame, source='ordinary', start_step=int(interactions))
            return
        if self.recorder is None:
            raise RuntimeError('Missing ordinary episode reset observation')
        terminal = bool(transition['is_terminal'][0])
        self.recorder.step(int(action[0].argmax().item()), frame,
                           float(transition['reward'][0]), terminated=terminal,
                           truncated=bool(done[0]) and not terminal)
        if bool(done[0]):
            self.archive.append(self.recorder.finish(end_step=int(interactions)))
            self.recorder = None

    def due(self, ordinary_interactions):
        return ordinary_interactions >= self.next_milestone

    def _train(self, name, function, arguments, logfile, **inputs):
        print(f'Return phase (in process): {name}; progress in {logfile}', flush=True)
        torch.set_num_threads(arguments.threads)
        with logfile.open('w') as stream, redirect_stdout(stream), redirect_stderr(stream):
            try:
                return function(arguments, **inputs)
            except Exception as error:
                traceback.print_exc()
                raise RuntimeError(f'{name} failed; see {logfile}') from error

    def _prepare_round(self, directory, ordinary_interactions):
        source_ids = self.archive.pending_ids
        source_hash = self.archive.fingerprint(source_ids)
        # Decode once, replacing compressed records in place. Every stage below
        # reads the same arrays; there is no serialized snapshot or IPC copy.
        episodes = self.archive.materialize(source_ids)
        with auxiliary_training_context(self.device):
            return self._fit_episodes(directory, ordinary_interactions, episodes,
                                      source_ids, source_hash)

    def _fit_classifier(self, args, episodes, metadata, source_sha256):
        # variant_modules() puts the standalone modules on the import path.
        from extract_dino_features import DEFAULT_BACKBONE, DinoFrameEncoder, extract_episodes
        from train_gru_classifier import train_from_features

        config = self.config.classifier
        encoder = DinoFrameEncoder(config.get('backbone', DEFAULT_BACKBONE)).to(self.device).eval()
        features = extract_episodes(episodes, encoder, self.device, config.get('feature_batch_size', 16))
        classifier, checkpoint = train_from_features(args, features, metadata, encoder.provenance(source_sha256))
        classifier.attach_encoder(encoder)
        return classifier, checkpoint, {ep['id']: ep['features'] for ep in features}

    def _fit_episodes(self, directory, ordinary_interactions, episodes, source_ids, source_hash):
        classifier_module, supervised = variant_modules()
        save_json(directory / 'plan.json', dict(source_ids=source_ids, episodes=self.episodes,
                  trajectory_storage='memory', training_execution='in_process', source_sha256=source_hash,
                  ordinary_interactions=ordinary_interactions, milestone=self.next_milestone,
                  counter_sampling='inverse_count', prefix_sampling='uniform_matching_occurrence',
                  previous_policy=str(self.policy_checkpoint) if self.policy_checkpoint else None))
        classifier_path = directory / 'classifier.pt'
        classifier_args = ['--output', str(classifier_path), '--device', self.device, '--all-data']
        for key, value in self.config.classifier.items():
            if key in ('backbone', 'feature_batch_size'):
                continue
            classifier_args.extend(['--' + key.replace('_', '-'), str(value)])
        classifier_args = classifier_module.parse_args(classifier_args, require_features=False)
        classifier, checkpoint, features_by_id = self._train(
            'classifier', self._fit_classifier, classifier_args,
            directory / 'classifier.log', episodes=episodes, metadata=self.archive.metadata,
            source_sha256=source_hash)

        # Extract once: policy fitting and the anchor bank share these demos.
        demos = supervised.prepare_demonstrations(
            episodes, checkpoint['training_episode_ids'], classifier, checkpoint['threshold'],
            self.device, skip_empty=True, features_by_id=features_by_id)
        del features_by_id
        policy_output = directory / 'agent'
        policy_args = ['train', '--output', str(policy_output), '--device', self.device,
                       '--skip-empty-anchors']
        for key, value in self.config.policy.items():
            policy_args.extend(['--' + key.replace('_', '-'), str(value)])
        if self.policy_state is None and self.policy_checkpoint is not None:
            policy_args.extend(['--finetune', str(self.policy_checkpoint)])
        policy_args = supervised.parse_args(policy_args, require_inputs=False)
        try:
            self.policy_state, policy = self._train(
                'return policy', supervised.train_from_demonstrations, policy_args,
                directory / 'policy.log', classifier=classifier, classifier_ckpt=checkpoint,
                metadata=self.archive.metadata, demos=demos,
                classifier_sha256=file_sha256(classifier_path), state=self.policy_state)
        finally:
            # Keep latest weights and Adam history for the next round, without
            # occupying GPU memory during Dreamer training or assisted rollouts.
            if self.policy_state is not None:
                self.policy_state.optimizer.zero_grad(set_to_none=True)
                self.policy_state.to('cpu')

        similarity = classifier_similarity(classifier, self.device)
        bank = AnchorBank.build(demos, similarity, checkpoint['threshold'])
        # Only goal images and lookup indices are needed for the 100 rollouts.
        goals_by_id = {demo['id']: demo['goals'] for demo in demos}
        return classifier, policy, bank, goals_by_id, source_ids, source_hash

    def run(self, agent, replay, ordinary_interactions):
        if not self.due(ordinary_interactions):
            raise ValueError('Return phase called before its ordinary-collection milestone')
        if not self.archive.pending_ids:
            return False
        self.round += 1
        directory = self.root / f'round_{self.round:06d}'
        directory.mkdir()
        classifier, policy, bank, goals_by_id, source_ids, source_hash = self._prepare_round(
            directory, ordinary_interactions)
        # All fitting and prefix preparation used shared episode arrays. Drop
        # raw S now; goal images and prefix indices are enough for returning.
        self.archive.consume(source_ids)
        initial_counter_total = int(bank.counts.sum())
        policy_output = directory / 'agent'
        from envs import make_env
        config = OmegaConf.create(OmegaConf.to_container(self.env_config, resolve=True))
        config.seed = secrets.randbelow(2**31 - 1)
        env = make_env(config, 0)
        records = []
        was_training = agent.training
        agent.eval()
        try:
            for index in range(self.episodes):
                representative, (episode_id, position) = bank.sample(self.rng)
                count_before = int(bank.counts[representative])
                goals = goals_by_id[episode_id][:position + 1]
                episode, rows, references, result = collect_assisted_episode(
                    agent, policy, classifier, env, goals, bank.references[representative],
                    bank.threshold, -self.round, self.device)
                print(f"[return round {self.round}] episode {index + 1}/{self.episodes} "
                      f"return={result['env_return']:.2f} steps={result['interactions']} "
                      f"reached={result['reached']} takeover={result['takeover']}", flush=True)
                # Complete T enters replay before its counters affect the next draw.
                for row in rows:
                    replay.add_transition(row)
                self.assisted_interactions += result['interactions']
                new_id = self.archive.append(episode | dict(end_step=int(ordinary_interactions),
                                                          return_round=self.round))
                del episode, rows
                increments = bank.update(references)
                # Keep scalar diagnostics, not generated anchor sequences, on disk.
                result.pop('anchor_indices', None)
                result.update(episode_id=new_id, representative=representative,
                              source_episode_id=episode_id, prefix_anchor_position=position,
                              target_count_before=count_before, counter_increments=int(increments.sum()))
                records.append(result)
                save_json(directory / 'rollouts.json', records)
                print(json.dumps(dict(event='return_episode', round=self.round,
                                      episode=index + 1, **result)), flush=True)
        finally:
            env.close()
            agent.train(was_training)
        self.policy_checkpoint = policy_output / 'latest.pt'
        self.next_milestone = (ordinary_interactions // self.interval + 1) * self.interval
        report = dict(ordinary_interactions=ordinary_interactions,
                  assisted_interactions=self.assisted_interactions,
                  total_interactions=ordinary_interactions + self.assisted_interactions,
                  generated_episodes=len(records), source_episodes=len(source_ids),
                  trajectory_storage='memory', training_execution='in_process',
                  source_episode_ids=list(source_ids),
                  generated_episode_ids=[row['episode_id'] for row in records],
                  source_sha256=source_hash, pending_episode_ids=list(self.archive.pending_ids),
                  initial_counter_total=initial_counter_total, final_counter_total=int(bank.counts.sum()),
                  representatives=len(bank), reached=sum(row['reached'] for row in records),
                  takeovers=sum(row['takeover'] for row in records), next_milestone=self.next_milestone,
                  environment_seed=int(config.seed), policy_checkpoint=str(self.policy_checkpoint),
                  policy_sha256=file_sha256(self.policy_checkpoint),
                  classifier_sha256=file_sha256(directory / 'classifier.pt'))
        save_json(directory / 'report.json', report)
        # One rolling weight/optimizer checkpoint; replay is never serialized.
        import tools
        temporary = self.root.parent / 'latest.pt.tmp'
        torch.save(dict(agent_state_dict=agent.state_dict(),
                        optims_state_dict=tools.recursively_collect_optim_state_dict(agent),
                        ordinary_interactions=ordinary_interactions,
                        assisted_interactions=self.assisted_interactions), temporary)
        temporary.replace(self.root.parent / 'latest.pt')
        with (self.root / 'cycles.jsonl').open('a') as stream:
            stream.write(json.dumps(report, allow_nan=False) + '\n')
        # Keep the preceding successful round until this one is fully saved.
        # Only directories created by this instance are eligible for removal;
        # an external initial_policy checkpoint is never touched.
        if self._previous_directory is not None:
            shutil.rmtree(self._previous_directory)
        self._previous_directory = directory
        del policy, classifier, bank, goals_by_id
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
