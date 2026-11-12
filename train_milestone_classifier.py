"""Train visual skip-gram: nearby frames are positive; random frames negative.

No rewards, actions, milestone labels, or pretrained visual weights are used.
Negatives reject known positive pairs in the same episode, with no SPTM margin.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from milestone_model import PairClassifier, frames_to_tensor


class EpisodeFrames:
    """Cache uint8 observations without duplicating the whole corpus."""

    def __init__(self, paths):
        self.paths = [Path(path) for path in paths]
        self.frames = []
        for path in self.paths:
            with np.load(path, allow_pickle=False) as data:
                frames = data["frames"]
            if frames.ndim != 4 or frames.shape[1:] != (84, 84, 3) or frames.dtype != np.uint8:
                raise ValueError(f"{path}: expected uint8 RGB frames [T,84,84,3]")
            if len(frames) < 2:
                raise ValueError(f"{path}: need at least two observations")
            self.frames.append(frames)
        if not self.frames:
            raise ValueError("No episodes found")
        self.lengths = np.array([len(frames) for frames in self.frames], dtype=np.int64)
        self.ends = np.cumsum(self.lengths)
        self.starts = np.concatenate(([0], self.ends[:-1]))
        self.total = int(self.ends[-1])

    def locate(self, indices):
        episodes = np.searchsorted(self.ends, indices, side="right")
        return episodes, indices - self.starts[episodes]

    def get(self, indices):
        episodes, local = self.locate(indices)
        result = np.empty((len(indices), 84, 84, 3), dtype=np.uint8)
        for episode in np.unique(episodes):
            selected = episodes == episode
            result[selected] = self.frames[episode][local[selected]]
        return result


class SkipGramSampler:
    """Uniform centers, uniformly selected nearby contexts, random noise frames."""

    def __init__(self, corpus, window=5, negative_ratio=1, seed=0):
        if window < 1 or negative_ratio < 1:
            raise ValueError("window and negative_ratio must be positive")
        if len(corpus.frames) == 1 and corpus.lengths[0] <= 2 * window + 1:
            raise ValueError("Single episode too short to give every center a valid negative")
        self.corpus = corpus
        self.window = window
        self.negative_ratio = negative_ratio
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size):
        if batch_size < self.negative_ratio + 1 or batch_size % (self.negative_ratio + 1):
            raise ValueError("batch_size must be divisible by negative_ratio + 1")
        n_positive = batch_size // (self.negative_ratio + 1)
        centers = self.rng.integers(self.corpus.total, size=n_positive)
        episode, local = self.corpus.locate(centers)
        lo = np.maximum(local - self.window, 0)
        hi = np.minimum(local + self.window, self.corpus.lengths[episode] - 1)
        context = self.rng.integers(lo, hi)
        context += context >= local  # Map [lo, hi) onto [lo, hi] excluding center.
        positives = context + self.corpus.starts[episode]
        negative_centers = np.repeat(centers, self.negative_ratio)
        negative_episodes, negative_local = self.corpus.locate(negative_centers)
        negatives = self.rng.integers(self.corpus.total, size=len(negative_centers))
        while True:
            sampled_episode, sampled_local = self.corpus.locate(negatives)
            bad = ((sampled_episode == negative_episodes)
                   & (np.abs(sampled_local - negative_local) <= self.window))
            if not bad.any():
                break
            negatives[bad] = self.rng.integers(self.corpus.total, size=int(bad.sum()))
        first = np.concatenate((centers, negative_centers))
        second = np.concatenate((positives, negatives))
        labels = np.concatenate((np.ones(n_positive), np.zeros(len(negative_centers)))).astype(np.float32)
        permutation = self.rng.permutation(batch_size)
        return first[permutation], second[permutation], labels[permutation]


def pair_metrics(logits, labels, threshold=0.5):
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    predicted = probabilities >= threshold
    positive = labels == 1
    tpr = float(predicted[positive].mean())
    tnr = float((~predicted[~positive]).mean())
    # Mann-Whitney AUROC with averaged ranks for tied predictions.
    _, inverse, counts = np.unique(probabilities, return_inverse=True, return_counts=True)
    average_ranks = np.cumsum(counts) - (counts - 1) / 2
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    auc = (average_ranks[inverse][positive].sum() - n_positive * (n_positive + 1) / 2)
    auc /= n_positive * n_negative
    return {
        "loss": float(np.mean(np.logaddexp(0, logits) - labels * logits)),
        "accuracy": float((predicted == positive).mean()),
        "balanced_accuracy": (tpr + tnr) / 2,
        "positive_recall": tpr, "negative_recall": tnr, "auroc": float(auc),
        "mean_positive_probability": float(probabilities[positive].mean()),
        "mean_negative_probability": float(probabilities[~positive].mean()),
        "pairs": len(labels), "threshold": threshold,
    }


@torch.inference_mode()
def evaluate(model, corpus, pairs, device, batch_size):
    model.eval()
    first, second, labels = pairs
    logits = []
    for start in range(0, len(labels), batch_size):
        end = start + batch_size
        a = frames_to_tensor(corpus.get(first[start:end]), device)
        b = frames_to_tensor(corpus.get(second[start:end]), device)
        logits.append(model(a, b).cpu().numpy())
    return pair_metrics(np.concatenate(logits), labels)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/milestones/episodes"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/milestones/classifier.pt"))
    parser.add_argument("--architecture", choices=("small", "resnet18"), default="small")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64, help="Total positive and negative pairs per update")
    parser.add_argument("--window", type=int, default=5, help="Positive radius in saved observations (not raw ALE frames)")
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-pairs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.threads < 1 or args.embedding_dim < 1:
        raise ValueError("steps, eval-every, threads and embedding-dim must be positive")
    if args.window < 1 or args.negative_ratio < 1:
        raise ValueError("window and negative-ratio must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    for name in ("batch_size", "eval_pairs"):
        count = getattr(args, name)
        if count < args.negative_ratio + 1 or count % (args.negative_ratio + 1):
            raise ValueError(f"{name.replace('_', '-')} must be a positive multiple of negative-ratio + 1")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between 0 and 1")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists; choose another --output or use --overwrite")
    paths = sorted(args.data.glob("episode_*.npz"))
    if len(paths) < 2:
        raise ValueError("Need at least two episode_*.npz files for disjoint training/validation episodes")
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    order = np.random.default_rng(args.seed).permutation(len(paths))
    n_validation = min(len(paths) - 1, max(1, round(len(paths) * args.validation_fraction)))
    validation_paths = [paths[index] for index in order[:n_validation]]
    train_paths = [paths[index] for index in order[n_validation:]]
    train_data, validation_data = EpisodeFrames(train_paths), EpisodeFrames(validation_paths)
    train_sampler = SkipGramSampler(train_data, args.window, args.negative_ratio, args.seed)
    validation_sampler = SkipGramSampler(validation_data, args.window, args.negative_ratio, args.seed + 1)
    # Fixed evaluation pairs make checkpoint comparisons meaningful.
    validation_pairs = validation_sampler.sample(args.eval_pairs)
    train_pairs = SkipGramSampler(train_data, args.window, args.negative_ratio, args.seed + 2).sample(args.eval_pairs)
    model = PairClassifier(args.architecture, args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["resolved_device"] = device
    split = {"train": [str(path) for path in train_paths], "validation": [str(path) for path in validation_paths]}
    history = []
    best_score, best_step = -1.0, 0
    started = time.monotonic()
    print(json.dumps({"device": device, "parameters": sum(p.numel() for p in model.parameters()),
                      "train_episodes": len(train_paths), "validation_episodes": len(validation_paths),
                      "train_frames": train_data.total, "validation_frames": validation_data.total,
                      "note": "Episode split does not remove shared Go-Explore prefixes; metrics are temporal proxy labels."}), flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        first, second, labels = train_sampler.sample(args.batch_size)
        a = frames_to_tensor(train_data.get(first), device)
        b = frames_to_tensor(train_data.get(second), device)
        logits = model(a, b)
        targets = torch.from_numpy(labels).to(device)
        # Equal class contribution even if multiple noise samples are requested.
        losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = (losses[targets == 1].mean() + losses[targets == 0].mean()) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, validation_data, validation_pairs, device, args.batch_size)
            record = {"step": step, "elapsed_seconds": round(time.monotonic() - started, 2),
                      "batch_loss": float(loss.detach()), "validation": metrics}
            history.append(record)
            print(json.dumps(record), flush=True)
            if metrics["balanced_accuracy"] > best_score:
                best_score, best_step = metrics["balanced_accuracy"], step
                checkpoint = {
                    "format_version": 1, "model_state": model.state_dict(),
                    "model_config": model.model_config, "train_config": config,
                    "split": split, "metrics": metrics, "step": step, "threshold": 0.5,
                    "preprocessing": "uint8 RGB 84x84 NHWC -> float32 NCHW / 255",
                    "labels": "1: same episode, 0 < distance <= window; 0: random corpus frame, excluding known positives",
                }
                temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(args.output)
            args.output.with_suffix(".metrics.json").write_text(json.dumps({
                "train_config": config, "split": split, "best_step": best_step,
                "best_validation_balanced_accuracy": best_score, "history": history,
            }, indent=2) + "\n")
    # Report training vs validation quality for the actual saved checkpoint.
    from milestone_model import load_classifier
    best_model, _ = load_classifier(args.output, device)
    train_metrics = evaluate(best_model, train_data, train_pairs, device, args.batch_size)
    metrics_path = args.output.with_suffix(".metrics.json")
    report = json.loads(metrics_path.read_text())
    report["saved_checkpoint_train_metrics"] = train_metrics
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"saved": str(args.output), "best_step": best_step, "train": train_metrics}), flush=True)


if __name__ == "__main__":
    main()
