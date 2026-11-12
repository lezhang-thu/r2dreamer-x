"""Coordinate online milestone learning and an independent goal-conditioned A2C."""

from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image

from extract_milestones import extract_anchors, save_contact_sheets, save_video
from goal_a2c import GoalA2C
from goal_policy import GoalPolicy
from online_milestones import CompletedTrajectoryStore, OnlineClassifier, to_rgb84


def select_recent_evaluation(episodes, budget, fraction, count):
    """Take only the latest completed episodes inside the final budget window."""
    if budget <= 0 or not 0 < fraction <= 1 or count < 1:
        raise ValueError("Invalid final evaluation budget/window/count")
    recent = [episode for episode in episodes
              if episode.complete and budget * (1 - fraction) <= episode.end_step <= budget]
    return sorted(recent, key=lambda episode: (episode.end_step, episode.id))[-count:]


class IsolatedRandomState:
    """Companion work must not consume Dreamer's Python/NumPy/Torch RNG streams."""

    def __init__(self, seed, device):
        self.python = random.Random(seed).getstate()
        self.numpy = np.random.RandomState(seed).get_state()
        self.cpu = torch.Generator().manual_seed(seed).get_state()
        device = torch.device(device)
        self.cuda_device = device if device.type == "cuda" else None
        self.cuda = (torch.Generator(device=device).manual_seed(seed).get_state()
                     if self.cuda_device is not None else None)

    @contextmanager
    def use(self):
        python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda = torch.cuda.get_rng_state(self.cuda_device) if self.cuda_device is not None else None
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        if self.cuda_device is not None:
            torch.cuda.set_rng_state(self.cuda, self.cuda_device)
        try:
            yield
        finally:
            self.python, self.numpy, self.cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
            random.setstate(python)
            np.random.set_state(numpy)
            torch.set_rng_state(cpu)
            if self.cuda_device is not None:
                self.cuda = torch.cuda.get_rng_state(self.cuda_device)
                torch.cuda.set_rng_state(cuda, self.cuda_device)


class GoalObservationEnv:
    """Match Dreamer's original image preprocessing, then resize for milestones."""

    def __init__(self, config):
        from envs.atari import Atari
        suite, task = config.task.split("_", 1)
        if suite != "atari":
            raise ValueError("Goal training currently supports Atari")
        self.env = Atari(task, action_repeat=config.action_repeat, size=config.size,
                         gray=config.gray, noops=config.noops, lives=config.lives,
                         sticky=config.sticky, actions=config.actions, length=config.time_limit,
                         pooling=config.pooling, aggregate=config.aggregate, resize=config.resize,
                         autostart=config.autostart, clip_reward=False)
        self.action_space = self.env.action_space

    @staticmethod
    def _convert(observation):
        return {**observation, "image": to_rgb84(observation["image"])}

    def reset(self, seed=None):
        return self._convert(self.env.reset(seed=seed))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self._convert(observation), reward, done, info

    def close(self):
        self.env.close()


class GoalTraining:
    """Synchronous interleaving avoids races and keeps classifier versions fixed.

    Dreamer is paused while a goal batch runs. The live classifier updates only
    between batches; a separate frozen copy extracts and scores every batch.
    Goal observations never enter Dreamer's replay or the classifier corpus.
    """

    def __init__(self, config, env_config, logger, output, env_factory=None):
        self.config, self.env_config, self.logger = config, env_config, logger
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.iterdir()):
            raise FileExistsError("Goal output is not empty; use a fresh logdir (joint resume is not implemented)")
        if config.recent_episodes < max(config.min_episodes, config.evaluation.episodes):
            raise ValueError("recent_episodes must retain min_episodes and evaluation.episodes")
        if min(config.min_episodes, config.train_every_episodes, config.batch_size) < 1:
            raise ValueError("Episode scheduling parameters must be positive")
        if not 0 < config.classifier.threshold < 1:
            raise ValueError("Classifier threshold must be in (0,1)")
        self.rng = np.random.default_rng(config.seed)
        self.random_state = IsolatedRandomState(config.seed, config.device)
        self.env_factory = env_factory or (lambda index: GoalObservationEnv(env_config))
        self.store = CompletedTrajectoryStore(self.output / "recent_trajectories", config.recent_episodes)
        self.completed = self.last_train_completed = self.batches = 0
        self.train_env_steps = self.eval_env_steps = 0
        with self.random_state.use():
            c = config.classifier
            self.classifier = OnlineClassifier(
                device=config.device, checkpoint=c.checkpoint, architecture=c.architecture,
                embedding_dim=c.embedding_dim, window=c.window, batch_size=c.batch_size,
                learning_rate=c.learning_rate, seed=config.seed, threshold=c.threshold)
            probe = self.env_factory(0)
            try:
                num_actions = probe.action_space.n
            finally:
                probe.close()
            p = config.policy
            policy = GoalPolicy(num_actions, p.hidden_dim, p.attention_heads, p.mask_completed)
            snapshot, _ = self.classifier.snapshot()
            self.agent = GoalA2C(policy, snapshot, device=config.device, threshold=c.threshold,
                                gamma=p.gamma, learning_rate=p.learning_rate,
                                entropy_coef=p.entropy_coef, value_coef=p.value_coef,
                                max_grad_norm=p.max_grad_norm)

    def observe(self, trans, dreamer_step):
        """Read CPU observations at collection time, before replay reshaping."""
        images = trans["image"].detach().cpu().numpy()
        first = trans["is_first"].detach().cpu().numpy().reshape(-1)
        last = trans["is_last"].detach().cpu().numpy().reshape(-1)
        rewards = trans["reward"].detach().cpu().numpy().reshape(-1)
        for worker, frame in enumerate(images):
            result = self.store.add_observation(worker, frame, bool(first[worker]), bool(last[worker]),
                                                int(dreamer_step), float(rewards[worker]))
            self.completed += int(result is not None)

    def _anchors(self, episodes, snapshot):
        sequences, indices = [], []
        for episode in episodes:
            anchors, _ = extract_anchors(snapshot, episode.frames, self.classifier.threshold,
                                         batch_size=256, device=self.config.device)
            # The initial observation is context, never an automatically rewarded goal.
            indices.append(anchors[1:])
            sequences.append(episode.frames[anchors[1:]])
        return sequences, indices

    def _save(self, version):
        self.classifier.save(self.output / "classifier.pt")
        self.agent.save(self.output / "goal_agent.pt", metadata={
            "classifier_version": version, "goal_train_env_steps": self.train_env_steps,
            "goal_eval_env_steps": self.eval_env_steps, "batches": self.batches})

    def _record(self, record, dreamer_step):
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if self.logger is not None:
            self.logger.scalar("goal/train_env_steps", self.train_env_steps)
            self.logger.scalar("goal/eval_env_steps", self.eval_env_steps)
            self.logger.scalar("goal/classifier_updates", self.classifier.updates)
            self.logger.scalar("goal/batches", self.batches)
            self.logger.write(dreamer_step)

    def maybe_train(self, dreamer_step):
        if (len(self.store.episodes) < self.config.min_episodes
                or self.completed - self.last_train_completed < self.config.train_every_episodes):
            return None
        with self.random_state.use():
            c = self.config.classifier
            steps = c.warmup_steps if self.classifier.updates == 0 else c.update_steps
            classifier_metrics = self.classifier.update(self.store.episodes, steps=steps)
            if self.classifier.updates == 0:
                return None  # Too little valid temporal context; never reward a random classifier.
            snapshot, version = self.classifier.snapshot()
            available = self.store.episodes
            choice = self.rng.choice(len(available), min(self.config.batch_size, len(available)), replace=False)
            episodes = [available[int(index)] for index in choice]
            sequences, indices = self._anchors(episodes, snapshot)
            self.agent.classifier = snapshot
            seeds = [int(self.config.seed + self.batches * self.config.batch_size + i) for i in range(len(episodes))]
            result = self.agent.run_batch(
                sequences, self.env_factory, seeds, train=True,
                max_steps=self.config.policy.max_steps, rollout_steps=self.config.policy.rollout_steps,
                source_ids=[episode.id for episode in episodes])
            self.train_env_steps += sum(episode["env_steps"] for episode in result["episodes"])
            self.batches += 1
            self.last_train_completed = self.completed
            record = {"phase": "train", "dreamer_step": dreamer_step, "classifier_version": version,
                      "classifier": classifier_metrics, "source_ids": [episode.id for episode in episodes],
                      "classifier_source_ids": self.classifier.last_episode_ids,
                      "source_end_steps": [episode.end_step for episode in episodes],
                      "anchor_indices": indices, "result": result}
            self._save(version)
            self._record(record, dreamer_step)
            print(f"Goal batch {self.batches}: classifier v{version}, "
                  f"{[len(seq) for seq in sequences]} goals, {result['summary']}", flush=True)
            return record

    def finish(self, dreamer_step, budget):
        """Freeze both learners, then evaluate only the most recent eligible goals."""
        with self.random_state.use():
            episodes = select_recent_evaluation(self.store.episodes, budget,
                                                self.config.evaluation.recent_fraction,
                                                self.config.evaluation.episodes)
            self.store.discard_partial()
            snapshot, version = self.classifier.snapshot()
            report = {"dreamer_step": int(dreamer_step), "dreamer_budget": int(budget),
                      "classifier_version": version, "source_ids": [ep.id for ep in episodes],
                      "source_end_steps": [ep.end_step for ep in episodes],
                      "recent_cutoff": budget * (1 - self.config.evaluation.recent_fraction),
                      "goal_train_env_steps": self.train_env_steps,
                      "classifier_updates": self.classifier.updates, "a2c_batches": self.batches,
                      "note": "Goal execution evaluation; source goals may have appeared during training."}
            if not episodes or self.classifier.updates == 0:
                report["status"] = "no_recent_completed_episodes" if not episodes else "classifier_not_trained"
                report["goal_eval_env_steps"] = self.eval_env_steps
            else:
                sequences, indices = self._anchors(episodes, snapshot)
                self.agent.classifier = snapshot
                result = self.agent.run_batch(
                    sequences, self.env_factory,
                    [int(self.config.seed + 10000000 + i) for i in range(len(episodes))],
                    train=False, max_steps=self.config.evaluation.max_steps,
                    rollout_steps=self.config.policy.rollout_steps,
                    capture_frames=self.config.evaluation.save_video,
                    deterministic=self.config.evaluation.deterministic,
                    source_ids=[episode.id for episode in episodes])
                fps = 60 / self.env_config.action_repeat
                for episode, anchors, record in zip(episodes, indices, result["episodes"]):
                    directory = self.output / "evaluation" / f"source_{episode.id:06d}"
                    directory.mkdir(parents=True, exist_ok=True)
                    record["anchor_indices"] = anchors
                    frames = record.pop("frames", None)
                    if self.config.evaluation.save_video:
                        save_video(episode.frames, directory / "source.mp4", fps)
                        if frames is not None and len(frames):
                            save_video(np.asarray(frames), directory / "goal_agent.mp4", fps)
                    save_contact_sheets(episode.frames, [0] + anchors, directory)
                    for number, index in enumerate(anchors):
                        Image.fromarray(episode.frames[index]).save(directory / f"anchor_{number:04d}_frame_{index:06d}.png")
                    (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
                self.eval_env_steps += sum(record["env_steps"] for record in result["episodes"])
                report.update(status="evaluated", result=result, goal_eval_env_steps=self.eval_env_steps)
            self._save(version)
            (self.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
            self._record({"phase": "evaluation", **report}, dreamer_step)
            print(f"Goal evaluation: {report['status']}; sources={report['source_ids']}", flush=True)
            return report
