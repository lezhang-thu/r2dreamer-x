"""Completed online trajectories and versioned visual skip-gram classifiers.

Only trajectories observed from reset through their final observation enter the
store. Classifier snapshots own independent weights, so ongoing training cannot
change a goal sequence already assigned to an environment.
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from milestone_model import PairClassifier, frames_to_tensor, load_classifier
from train_milestone_classifier import EpisodeFrames, SkipGramSampler


def to_rgb84(frame):
    """Copy one uint8 HWC/CHW observation to RGB 84x84 (gray repeats channels).

    Online Dreamer observations may be 64x64. Resizing uses the same bilinear
    Pillow convention as the Atari wrapper; no normalized float input is assumed.
    """
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 observation, got {frame.dtype}")
    if frame.ndim == 2:
        frame = frame[..., None]
    elif frame.ndim == 3 and frame.shape[-1] not in (1, 3) and frame.shape[0] in (1, 3):
        frame = np.moveaxis(frame, 0, -1)
    if frame.ndim != 3 or frame.shape[-1] not in (1, 3):
        raise ValueError(f"Expected one HWC/CHW color or gray image, got {frame.shape}")
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[:2] != (84, 84):
        frame = np.asarray(Image.fromarray(frame).resize((84, 84), Image.Resampling.BILINEAR))
    return np.array(frame, dtype=np.uint8, copy=True, order="C")


@dataclass(frozen=True)
class Trajectory:
    id: int
    end_step: int
    frames: np.ndarray
    reward: float
    complete: bool = True
    worker_id: int = 0

    @property
    def score(self):
        return self.reward


class CompletedTrajectoryStore:
    """Retain recent complete trajectories in memory and in an owned directory."""

    def __init__(self, output_dir, max_recent=20):
        if max_recent < 1:
            raise ValueError("max_recent must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_recent = int(max_recent)
        self._episodes = deque()
        self._pending = {}
        self._next_id = 0
        self.completed_total = 0
        self.dropped_partial = 0
        manifest = self.output_dir / "manifest.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved.get("kind") != "online_completed_trajectories" or saved.get("format_version") != 1:
                raise ValueError(f"{manifest} is not an online trajectory-store manifest")
            for entry in saved["episodes"]:
                episode_id = int(entry["id"])
                path = self._path(episode_id)
                if entry["file"] != path.name:
                    raise ValueError("Unexpected file path in online trajectory manifest")
                with np.load(path, allow_pickle=False) as data:
                    frames = data["frames"]
                    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (84, 84, 3):
                        raise ValueError(f"{path}: invalid trajectory frames")
                    if not bool(data["complete"]):
                        raise ValueError(f"{path}: partial trajectory in completed store")
                    frames.setflags(write=False)
                    self._episodes.append(Trajectory(episode_id, int(data["end_step"]), frames,
                                                     float(data["reward"]), True, int(data["worker_id"])))
            self._next_id = int(saved["next_id"])
            if self._episodes and self._next_id <= max(ep.id for ep in self._episodes):
                raise ValueError("Invalid next trajectory id in manifest")
            self.completed_total = int(saved.get("completed_total", self._next_id))
            self._trim_and_manifest()

    @property
    def episodes(self):
        return tuple(self._episodes)

    def latest(self, count=None):
        if count is not None and count < 1:
            raise ValueError("count must be positive")
        episodes = tuple(self._episodes)
        return episodes if count is None else episodes[-count:]

    def _path(self, episode_id):
        return self.output_dir / f"episode_{episode_id:08d}.npz"

    def _trim_and_manifest(self):
        removed = []
        while len(self._episodes) > self.max_recent:
            removed.append(self._episodes.popleft())
        manifest = {
            "format_version": 1,
            "kind": "online_completed_trajectories",
            "max_recent": self.max_recent,
            "next_id": self._next_id,
            "completed_total": self.completed_total,
            "episodes": [
                {"id": ep.id, "file": self._path(ep.id).name, "end_step": ep.end_step,
                 "frame_count": len(ep.frames), "reward": ep.reward,
                 "worker_id": ep.worker_id, "complete": True}
                for ep in self._episodes
            ],
        }
        temporary = self.output_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(self.output_dir / "manifest.json")
        # Only generated paths for episodes explicitly tracked by this store.
        for episode in removed:
            self._path(episode.id).unlink(missing_ok=True)

    def add_observation(self, worker_id, frame, is_first, is_last, dreamer_step,
                        reward=0.0, score=None):
        """Append one observation; return a trajectory only on completed episodes.

        A reset abandons an older unfinished sequence for that worker. Observations
        without a known reset are ignored, including a final frame of such a suffix.
        ``reward`` is the transition reward leading to this frame; an optional
        cumulative ``score`` overrides the running return.
        """
        worker_id = int(worker_id)
        if is_first:
            if worker_id in self._pending:
                self.dropped_partial += 1
            self._pending[worker_id] = {"frames": [], "reward": 0.0}
        if worker_id not in self._pending:
            return None
        pending = self._pending[worker_id]
        pending["frames"].append(to_rgb84(frame))
        if score is not None:
            pending["reward"] = float(score)
        elif not is_first:
            pending["reward"] += float(reward)
        if not is_last:
            return None
        frames = np.stack(pending["frames"])
        frames.setflags(write=False)
        episode = Trajectory(self._next_id, int(dreamer_step), frames,
                             pending["reward"], True, worker_id)
        path = self._path(episode.id)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite untracked trajectory {path}")
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, frames=episode.frames, reward=episode.reward,
                                score=episode.reward, complete=True,
                                end_step=episode.end_step, worker_id=worker_id,
                                trajectory_id=episode.id)
        temporary.replace(path)
        self._episodes.append(episode)
        self._next_id += 1
        self.completed_total += 1
        del self._pending[worker_id]
        self._trim_and_manifest()
        return episode

    def discard_partial(self):
        """Discard unfinished worker sequences, e.g. at the interaction budget."""
        count = len(self._pending)
        self.dropped_partial += count
        self._pending.clear()
        return count


class _ArrayEpisodes(EpisodeFrames):
    """The existing sampler's corpus interface over already resident trajectories."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.lengths = np.asarray([len(item) for item in self.frames], dtype=np.int64)
        self.ends = np.cumsum(self.lengths)
        self.starts = np.concatenate(([0], self.ends[:-1]))
        self.total = int(self.ends[-1])


class OnlineClassifier:
    """Train on completed episodes and publish independent, numbered snapshots."""

    def __init__(self, device="cpu", checkpoint=None, architecture="small", embedding_dim=128,
                 window=5, negative_ratio=1, batch_size=64, learning_rate=1e-3,
                 seed=0, threshold=0.5):
        if window < 1 or negative_ratio < 1:
            raise ValueError("window and negative_ratio must be positive")
        if batch_size < negative_ratio + 1 or batch_size % (negative_ratio + 1):
            raise ValueError("batch_size must be a positive multiple of negative_ratio + 1")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between zero and one")
        self.device = torch.device(device)
        self.window = int(window)
        self.negative_ratio = int(negative_ratio)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.threshold = float(threshold)
        self.updates = 0
        self.version = 0
        self.source = str(Path(checkpoint).resolve()) if checkpoint else "online_random_initialization"
        self.last_episode_ids = []
        self.last_metrics = {}
        # Model construction consumes only CPU RNG; preserve Dreamer's global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            if checkpoint:
                self.model, saved = load_classifier(checkpoint, self.device)
                self.updates = int(saved.get("online_updates", 0))
                self.version = int(saved.get("online_version", 0))
            else:
                self.model = PairClassifier(architecture, embedding_dim).to(self.device)
        self.model.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self._rng = np.random.default_rng(self.seed)

    def update(self, episodes, steps=20):
        """Run skip-gram updates; insufficient complete data returns a skip result."""
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        retained = []
        episode_ids = []
        for episode in episodes:
            if isinstance(episode, Trajectory):
                if not episode.complete:
                    continue
                frames = episode.frames
                episode_id = episode.id
            else:
                frames = np.asarray(episode)
                episode_id = len(episode_ids)
            if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (84, 84, 3):
                raise ValueError("Classifier episodes must be uint8 RGB [T,84,84,3]")
            if len(frames) < 2:
                continue
            retained.append(frames)
            episode_ids.append(int(episode_id))
        if steps == 0 or not retained:
            return {"status": "skipped", "reason": "no updates requested" if steps == 0 else "no complete usable episodes",
                    "steps": 0, "version": self.version, "updates": self.updates}
        corpus = _ArrayEpisodes(retained)
        if len(retained) == 1 and len(retained[0]) <= 2 * self.window + 1:
            return {"status": "skipped", "reason": "single episode too short for negative sampling",
                    "steps": 0, "version": self.version, "updates": self.updates}
        sampler = SkipGramSampler(corpus, self.window, self.negative_ratio,
                                  seed=int(self._rng.integers(0, 2**31)))
        total_loss = 0.0
        self.model.train()
        for _ in range(steps):
            first, second, labels = sampler.sample(self.batch_size)
            a = frames_to_tensor(corpus.get(first), self.device)
            b = frames_to_tensor(corpus.get(second), self.device)
            targets = torch.from_numpy(labels).to(self.device)
            logits = self.model(a, b)
            losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = (losses[targets == 1].mean() + losses[targets == 0].mean()) / 2
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            self.updates += 1
            total_loss += float(loss.detach())
        self.model.eval()
        self.version += 1
        self.last_episode_ids = episode_ids
        self.last_metrics = {"status": "updated", "steps": int(steps), "loss": total_loss / steps,
                             "version": self.version, "updates": self.updates,
                             "episodes": len(retained), "frames": corpus.total}
        return dict(self.last_metrics)

    def snapshot(self):
        """Return frozen weights and the version they represent, without aliasing."""
        model = deepcopy(self.model).eval().requires_grad_(False)
        model.snapshot_version = self.version
        return model, self.version

    def save(self, path, include_optimizer=True):
        """Save an atomic checkpoint compatible with milestone_model.load_classifier."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": 1,
            "model_config": self.model.model_config,
            "model_state": self.model.state_dict(),
            "threshold": self.threshold,
            "step": self.updates,
            "online_updates": self.updates,
            "online_version": self.version,
            "source": self.source,
            "source_episode_ids": self.last_episode_ids,
            "metrics": self.last_metrics,
            "train_config": {"mode": "online_skip_gram", "window": self.window,
                             "negative_ratio": self.negative_ratio, "batch_size": self.batch_size,
                             "seed": self.seed},
            "preprocessing": "uint8 RGB 84x84 NHWC -> float32 NCHW / 255",
            "labels": "1: same episode, 0 < distance <= window; 0: random corpus frame excluding known positives",
        }
        if include_optimizer:
            checkpoint["optimizer_state"] = self.optimizer.state_dict()
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        return path
