"""Periodic first-return, then Dreamer exploration on completed trajectories.

The scheduling clock contains ordinary Dreamer interactions only. Auxiliary
rollouts pause learning and use a separate environment, preserving an unfinished
ordinary episode. Each completed auxiliary rollout is inserted into replay
before updating the frozen anchor bank's counters and sampling the next goal.
"""

import gc
import json
from pathlib import Path
import secrets
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
import torch
from tensordict import TensorDict

from episode_archive import EpisodeArchive, EpisodeRecorder, resize_frame
from return_anchors import AnchorBank


VARIANT = Path(__file__).resolve().parent / 'variants-supervised-agent'


def variant_modules():
    # The standalone directory intentionally remains usable as two CLI modules.
    if str(VARIANT) not in sys.path:
        sys.path.insert(0, str(VARIANT))
    import train_gru_classifier
    import train_supervised_agent
    return train_gru_classifier, train_supervised_agent


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


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
        archive_dir = Path(logdir).resolve() / 'episode_archive'
        if ((self.root.exists() and any(self.root.iterdir()))
                or (archive_dir / 'manifest.json').exists()):
            raise ValueError('Use a fresh logdir; online replay/environment resume is not implemented. '
                             'Use return_exploration.initial_policy to warm-start a compatible return policy.')
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = dict(num_actions=int(num_actions),
                        env_config=OmegaConf.to_container(env_config, resolve=True))
        if action_meanings is not None:
            metadata['action_meanings'] = list(action_meanings)
        self.archive = EpisodeArchive(archive_dir, metadata)
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

    def _train(self, module, arguments, logfile):
        command = [sys.executable, '-u', str(VARIANT / module), *map(str, arguments)]
        print(f'Return phase: {module}; progress in {logfile}', flush=True)
        # Separate processes isolate auxiliary optimizer RNG/thread settings
        # from Dreamer and release their optimizer allocations after fitting.
        with logfile.open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f'{module} failed (exit {result.returncode}); see {logfile}')

    def run(self, agent, replay, ordinary_interactions):
        if not self.due(ordinary_interactions):
            raise ValueError('Return phase called before its ordinary-collection milestone')
        if not self.archive.pending_ids:
            # With the default episode time limit this cannot occur at 100k.
            # Keep the milestone pending until the first complete source exists.
            return False
        self.round += 1
        directory = self.root / f'round_{self.round:06d}'
        directory.mkdir()
        archive_path = directory / 'episodes.pkl'
        source_ids = self.archive.snapshot(archive_path)
        save_json(directory / 'plan.json', dict(source_ids=source_ids, episodes=self.episodes,
                  ordinary_interactions=ordinary_interactions, milestone=self.next_milestone,
                  counter_sampling='inverse_count', prefix_sampling='uniform_matching_occurrence',
                  previous_policy=str(self.policy_checkpoint) if self.policy_checkpoint else None))
        classifier_path = directory / 'classifier.pt'
        classifier_args = ['--archive', archive_path, '--output', classifier_path,
                           '--device', self.device, '--all-data']
        for key, value in self.config.classifier.items():
            classifier_args.extend(['--' + key.replace('_', '-'), value])
        self._train('train_gru_classifier.py', classifier_args, directory / 'classifier.log')
        policy_output = directory / 'agent'
        policy_args = ['train', '--archive', archive_path, '--classifier', classifier_path,
                       '--output', policy_output, '--device', self.device, '--skip-empty-anchors']
        for key, value in self.config.policy.items():
            policy_args.extend(['--' + key.replace('_', '-'), value])
        if self.policy_checkpoint is not None:
            policy_args.extend(['--finetune', self.policy_checkpoint])
        self._train('train_supervised_agent.py', policy_args, directory / 'policy.log')

        classifier_module, supervised = variant_modules()
        classifier, checkpoint = classifier_module.load_classifier(classifier_path, self.device)
        episodes, _ = classifier_module.load_archive(archive_path)
        demos = supervised.prepare_demonstrations(episodes, checkpoint['training_episode_ids'],
                    classifier, checkpoint['threshold'], self.device, skip_empty=True)
        saved = torch.load(policy_output / 'best.pt', map_location='cpu', weights_only=True)
        policy = supervised.GoalPolicy(**saved['model_config']).to(self.device)
        policy.load_state_dict(saved['model_state'])
        policy.eval().requires_grad_(False)
        del saved

        @torch.no_grad()
        def similarity(left, right):
            a = torch.as_tensor(np.array(left, copy=True), device=self.device)
            b = torch.as_tensor(np.array(right, copy=True), device=self.device)
            a = a[:, None].expand(-1, len(b), -1).reshape(-1, a.shape[-1])
            b = b[None].expand(len(left), -1, -1).reshape(-1, b.shape[-1])
            return classifier.pair_logits(a, b).sigmoid().reshape(len(left), len(right)).cpu().numpy()

        bank = AnchorBank.build(demos, similarity, checkpoint['threshold'])
        save_json(directory / 'anchors.initial.json', bank.state_dict())
        # Only goal images and lookup indices are needed for the 100 rollouts.
        goals_by_id = {demo['id']: demo['goals'] for demo in demos}
        del episodes, demos
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
                result.update(episode_id=new_id, representative=representative,
                              source_episode_id=episode_id, prefix_anchor_position=position,
                              target_count_before=count_before, counter_increments=int(increments.sum()))
                records.append(result)
                save_json(directory / 'rollouts.json', records)
                save_json(directory / 'counts.json', dict(counts=bank.counts.tolist(),
                          completed_episodes=len(records), assisted_interactions=self.assisted_interactions))
                print(json.dumps(dict(event='return_episode', round=self.round,
                                      episode=index + 1, **result)), flush=True)
        finally:
            env.close()
            agent.train(was_training)
        self.policy_checkpoint = policy_output / 'latest.pt'
        self.archive.consume(source_ids)
        self.next_milestone = (ordinary_interactions // self.interval + 1) * self.interval
        save_json(directory / 'report.json', dict(ordinary_interactions=ordinary_interactions,
                  assisted_interactions=self.assisted_interactions,
                  total_interactions=ordinary_interactions + self.assisted_interactions,
                  generated_episodes=len(records), source_episodes=len(source_ids),
                  representatives=len(bank), reached=sum(row['reached'] for row in records),
                  takeovers=sum(row['takeover'] for row in records), next_milestone=self.next_milestone,
                  environment_seed=int(config.seed), policy_checkpoint=str(self.policy_checkpoint)))
        # Persist Dreamer too; this is a weight checkpoint, not an online resume.
        import tools
        temporary = directory / 'dreamer.pt.tmp'
        torch.save(dict(agent_state_dict=agent.state_dict(),
                        optims_state_dict=tools.recursively_collect_optim_state_dict(agent),
                        ordinary_interactions=ordinary_interactions,
                        assisted_interactions=self.assisted_interactions), temporary)
        temporary.replace(directory / 'dreamer.pt')
        del policy, classifier, bank, goals_by_id
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
