"""Train a temporal CNN+GRU classifier directly from an Atari episode pickle.

Stage 1 learns a visual pair classifier. Stage 2 freezes its CNN and learns
causal GRU contexts. Nearby distinct frames are positives; two independently
sampled positions in the same episode are negatives, including nearby/self pairs.
This file also contains the archive/model utilities used by train_supervised_agent.py.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import random
import secrets
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_archive(path):
    """Read a trusted HERO archive or a list of episode dictionaries.

    Frames are uint8 RGB [N+1,84,84,3]; optional actions have N entries. The
    classifier does not require actions, but the supervised agent does.
    """
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if isinstance(payload, list):
        records, metadata = payload, {}
    elif isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        if payload.get("kind") == "atari_episode_archive" and payload.get("format_version") != 1:
            raise ValueError("Unsupported Atari archive version")
        records, metadata = payload["episodes"], dict(payload.get("metadata", {}))
    else:
        raise ValueError("Expected an episode list or a dictionary with an episodes list")
    episodes, ids = [], set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("Each episode must be a dictionary")
        episode = {key: value for key, value in record.items() if key != "payload_npz"}
        if "payload_npz" in record:
            with np.load(io.BytesIO(record["payload_npz"]), allow_pickle=False) as arrays:
                episode.update({key: arrays[key] for key in arrays.files})
        frames = np.asarray(episode.get("frames"))
        if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (84, 84, 3)
                or len(frames) < 2 or not episode.get("complete", True)):
            raise ValueError("Each complete episode needs at least two uint8 RGB84 frames")
        source_id = episode.get("id", index)
        if (not isinstance(source_id, (int, np.integer)) or isinstance(source_id, (bool, np.bool_))
                or int(source_id) in ids):
            raise ValueError("Episode IDs must be distinct integers")
        episode.update(id=int(source_id), frames=np.ascontiguousarray(frames), complete=True)
        ids.add(episode["id"])
        for key in ("actions", "rewards", "terminated", "truncated"):
            if episode.get(key) is None:
                continue
            values = np.asarray(episode[key])
            if values.shape != (len(frames) - 1,):
                raise ValueError(f"{key} must align with frames[t] -> frames[t+1]")
            if key == "actions" and (not np.issubdtype(values.dtype, np.integer) or np.any(values < 0)):
                raise ValueError("Actions must be nonnegative integer indices")
            if key == "rewards" and not np.isfinite(values).all():
                raise ValueError("Rewards must be finite")
            if key in ("terminated", "truncated") and values.dtype != np.bool_:
                raise ValueError("Termination flags must be boolean")
            episode[key] = values
        if "frame_count" in episode and int(episode["frame_count"]) != len(frames):
            raise ValueError("Compressed archive frame count differs from its payload")
        episodes.append(episode)
    if not episodes:
        raise ValueError("The archive contains no episodes")
    if all("end_step" in episode for episode in episodes):
        episodes.sort(key=lambda episode: (episode["end_step"], episode["id"]))
    return episodes, metadata


def split_episodes(episodes, metadata, test_episodes=None, all_data=False):
    """Preserve declared split IDs, or reserve the latest episodes as holdout."""
    ids = [episode["id"] for episode in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate episode IDs")
    if all_data:
        if test_episodes is not None:
            raise ValueError("--all-data cannot be combined with --test-episodes")
        if not ids:
            raise ValueError("Classifier fitting needs at least one complete episode")
        return ids, []
    if "training_episode_ids" in metadata or "evaluation_episode_ids" in metadata:
        train = list(metadata.get("training_episode_ids", []))
        test = list(metadata.get("evaluation_episode_ids", []))
        if (not train or not test or any(type(value) is not int for value in train + test)
                or len(set(train + test)) != len(train) + len(test) or set(train + test) != set(ids)):
            raise ValueError("Archive train/test IDs must be nonempty, disjoint, and cover every episode")
        if test_episodes is not None and test_episodes != len(test):
            raise ValueError("--test-episodes cannot override the archive's declared split")
    else:
        count = max(1, math.ceil(len(ids) * .125)) if test_episodes is None else test_episodes
        if type(count) is not int or not 1 <= count <= len(ids) - 2:
            raise ValueError("Reserve at least one test episode and two training episodes")
        train, test = ids[:-count], ids[-count:]
    if len(train) < 2:
        raise ValueError("Classifier fitting needs at least two training episodes for its internal split")
    return train, test


def frames_to_tensor(frames, device="cpu"):
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (84, 84, 3):
        raise ValueError("Expected uint8 RGB frames [B,84,84,3]")
    return torch.from_numpy(np.array(frames, copy=True)).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).div_(255)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.BatchNorm2d(channels))
        self.skip = nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels))

    def forward(self, value):
        return torch.relu(self.main(value) + self.skip(value))


def make_encoder(architecture, embedding_dim):
    if architecture == "small":
        return nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(64 * 4 * 4, embedding_dim))
    if architecture != "resnet18":
        raise ValueError("Architecture must be resnet18 or small")
    layers = [nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1)]
    previous = 64
    for channels in (64, 128, 256, 512):
        layers.extend((BasicBlock(previous, channels, 1 if channels == 64 else 2), BasicBlock(channels, channels)))
        previous = channels
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, embedding_dim))


class VisualClassifier(nn.Module):
    def __init__(self, architecture="resnet18", embedding_dim=128):
        super().__init__()
        self.model_config = dict(architecture=architecture, embedding_dim=embedding_dim)
        self.encoder = make_encoder(architecture, embedding_dim)
        self.head = nn.Sequential(nn.Linear(2 * embedding_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def pair_logits(self, first, second):
        return self.head(torch.cat(((first - second).abs(), first * second), dim=-1)).squeeze(-1)


class RecurrentClassifier(VisualClassifier):
    """Each embedding represents the actual episode prefix ending at that frame."""
    is_recurrent = True

    def __init__(self, architecture="resnet18", embedding_dim=128, hidden_dim=128, num_layers=1):
        super().__init__(architecture, embedding_dim)
        self.embedding_dim, self.hidden_dim, self.num_layers = embedding_dim, hidden_dim, num_layers
        self.model_config.update(hidden_dim=hidden_dim, num_layers=num_layers)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.projection = nn.Identity() if hidden_dim == embedding_dim else nn.Linear(hidden_dim, embedding_dim)
        self.encoder.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_visual(self, images):
        return self.encoder(images)

    def encode_features(self, features, hidden=None, lengths=None):
        if features.ndim != 3 or features.shape[-1] != self.embedding_dim:
            raise ValueError("Features must have shape [B,T,embedding_dim]")
        batch, steps, _ = features.shape
        if hidden is None:
            hidden = features.new_zeros((self.num_layers, batch, self.hidden_dim))
        if hidden.shape != (self.num_layers, batch, self.hidden_dim):
            raise ValueError("Wrong recurrent hidden-state shape")
        if not steps:
            return features.new_zeros((batch, 0, self.embedding_dim)), hidden
        if lengths is None:
            output, state = self.gru(features, hidden)
            return self.projection(output), state
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=features.device)
        if lengths.shape != (batch,) or torch.any(lengths < 0) or torch.any(lengths > steps):
            raise ValueError("Invalid sequence lengths")
        nonempty = lengths > 0
        output, state = features.new_zeros((batch, steps, self.embedding_dim)), hidden.clone()
        if nonempty.any():
            packed = pack_padded_sequence(features[nonempty], lengths[nonempty].cpu(), batch_first=True, enforce_sorted=False)
            encoded, next_state = self.gru(packed, hidden[:, nonempty].contiguous())
            values, _ = pad_packed_sequence(encoded, batch_first=True, total_length=steps)
            valid = torch.arange(steps, device=features.device)[None] < lengths[nonempty, None]
            output[nonempty] = self.projection(values) * valid[..., None]
            state[:, nonempty] = next_state
        return output, state

    def encode_step(self, images, hidden=None):
        output, state = self.encode_features(self.encode_visual(images)[:, None], hidden)
        return output[:, 0], state


@torch.no_grad()
def encode_trajectory(model, frames, device="cpu", batch_size=1):
    if batch_size < 1 or len(frames) < 1:
        raise ValueError("Need a positive batch size and nonempty frames")
    model.eval()
    pieces, hidden = [], None
    for start in range(0, len(frames), batch_size):
        visual = model.encode_visual(frames_to_tensor(frames[start:start + batch_size], device))
        encoded, hidden = model.encode_features(visual[None], hidden)
        pieces.append(encoded[0].cpu())
    return torch.cat(pieces)


def model_fingerprint(model):
    digest = hashlib.sha256(json.dumps(model.model_config, sort_keys=True).encode())
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}:".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_classifier(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("kind") != "standalone_gru_classifier" or saved.get("format_version") != 1:
        raise ValueError("Use a classifier checkpoint produced by train_gru_classifier.py")
    model = RecurrentClassifier(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval().requires_grad_(False)
    if model_fingerprint(model) != saved["model_fingerprint"]:
        raise ValueError("Classifier state does not match its saved fingerprint")
    return model, saved


class PairSampler:
    def __init__(self, sequences, window=20, negative_ratio=1, seed=0):
        self.lengths = np.array([len(sequence) for sequence in sequences], dtype=np.int64)
        if not len(self.lengths) or np.any(self.lengths < 2) or window < 1 or negative_ratio < 1:
            raise ValueError("Need valid episodes, window and negative ratio")
        self.ends = np.cumsum(self.lengths)
        self.starts = np.concatenate(([0], self.ends[:-1]))
        self.total, self.window, self.ratio = int(self.ends[-1]), window, negative_ratio
        self.rng = np.random.default_rng(seed)

    def locate(self, indices):
        episode = np.searchsorted(self.ends, indices, side="right")
        return episode, indices - self.starts[episode]

    def sample(self, batch_size):
        if batch_size < self.ratio + 1 or batch_size % (self.ratio + 1):
            raise ValueError("Pair count must be a positive multiple of negative_ratio + 1")
        count = batch_size // (self.ratio + 1)
        centers = self.rng.integers(self.total, size=count)
        episode, local = self.locate(centers)
        lower = np.maximum(local - self.window, 0)
        upper = np.minimum(local + self.window, self.lengths[episode] - 1)
        nearby = self.rng.integers(lower, upper)
        nearby += nearby >= local
        noise_episode = self.rng.choice(len(self.lengths), size=count * self.ratio,
                                        p=self.lengths / self.total)
        noise_length, noise_start = self.lengths[noise_episode], self.starts[noise_episode]
        first = np.concatenate((centers, noise_start + self.rng.integers(noise_length)))
        second = np.concatenate((self.starts[episode] + nearby, noise_start + self.rng.integers(noise_length)))
        labels = np.concatenate((np.ones(count), np.zeros(count * self.ratio))).astype(np.float32)
        order = self.rng.permutation(batch_size)
        return first[order], second[order], labels[order]


def get_frames(sequences, sampler, indices):
    episode, positions = sampler.locate(indices)
    return np.stack([sequences[int(e)][int(t)] for e, t in zip(episode, positions)])


def sampled_embeddings(model, features, sampler, indices, device, tbptt_chunk=0):
    episode, local = sampler.locate(np.asarray(indices))
    unique, inverse = np.unique(episode, return_inverse=True)
    lengths = np.zeros(len(unique), dtype=np.int64)
    np.maximum.at(lengths, inverse, local + 1)
    sequences = [features[int(e)][:int(n)].to(device) for e, n in zip(unique, lengths)]
    padded = pad_sequence(sequences, batch_first=True)
    lengths = torch.as_tensor(lengths, device=device)
    if tbptt_chunk:
        pieces, hidden = [], None
        for start in range(0, padded.shape[1], tbptt_chunk):
            chunk = padded[:, start:start + tbptt_chunk]
            encoded, hidden = model.encode_features(chunk, hidden, (lengths - start).clamp(0, chunk.shape[1]))
            pieces.append(encoded)
            hidden = hidden.detach()
        outputs = torch.cat(pieces, dim=1)
    else:
        outputs, _ = model.encode_features(padded, lengths=lengths)
    return outputs[torch.as_tensor(inverse, device=device), torch.as_tensor(local, device=device)]


def balanced_loss(logits, labels):
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (losses[labels == 1].mean() + losses[labels == 0].mean()) / 2


@torch.no_grad()
def evaluate_pairs(model, data, sampler, pairs, device, recurrent=False):
    model.eval()
    first, second, labels = pairs
    logits = []
    for start in range(0, len(labels), 128):
        a, b = first[start:start + 128], second[start:start + 128]
        if recurrent:
            embeddings = sampled_embeddings(model, data, sampler, np.concatenate((a, b)), device)
        else:
            frames = get_frames(data, sampler, np.concatenate((a, b)))
            embeddings = model.encoder(frames_to_tensor(frames, device))
        logits.append(model.pair_logits(*embeddings.chunk(2)).cpu())
    logits, target = torch.cat(logits), torch.from_numpy(labels)
    positive = target == 1
    predicted = logits.sigmoid() >= .5
    return dict(loss=float(balanced_loss(logits, target)), accuracy=float((predicted == positive).float().mean()),
                balanced_accuracy=float(((predicted[positive]).float().mean() + (~predicted[~positive]).float().mean()) / 2),
                positive_recall=float(predicted[positive].float().mean()), negative_recall=float((~predicted[~positive]).float().mean()))


@torch.no_grad()
def cache_features(model, episodes, device, batch_size=128):
    model.eval()
    return [torch.cat([model.encode_visual(frames_to_tensor(frames[start:start + batch_size], device)).cpu()
                       for start in range(0, len(frames), batch_size)]) for frames in episodes]


def atomic_save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def copy_model_state(model):
    """Keep a best-fit snapshot without retaining live GPU tensors."""
    return {name: tensor.detach().to('cpu', copy=True) for name, tensor in model.state_dict().items()}


def parse_args(argv=None, *, require_archive=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=require_archive)
    parser.add_argument("--output", type=Path, required=True, help="Final recurrent classifier .pt file")
    parser.add_argument("--architecture", choices=("resnet18", "small"), default="resnet18")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--visual-steps", type=int, default=5000)
    parser.add_argument("--gru-steps", type=int, default=5000)
    parser.add_argument("--visual-learning-rate", type=float, default=1e-3)
    parser.add_argument("--gru-learning-rate", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-pairs", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--test-episodes", type=int, help="For archives without a declared split; default latest 12.5%%")
    parser.add_argument("--all-data", action="store_true",
                        help="Fit every episode; select checkpoints using diagnostic pairs from the same episodes")
    parser.add_argument("--tbptt-chunk", type=int, default=0, help="0: full-prefix gradients; positive: detach between chunks")
    parser.add_argument("--seed", type=int, help="Optimization/split seed; default draws from OS entropy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def train_from_episodes(args, episodes, metadata, source_sha256):
    """Fit a fresh classifier directly on shared episode arrays.

    This is the online integration entry point. It never reads an archive or
    reloads its own checkpoints. The caller owns RNG/thread isolation.
    """
    for name in ("embedding_dim", "hidden_dim", "visual_steps", "gru_steps", "window", "negative_ratio", "threads", "eval_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (not 0 < args.validation_fraction < 1 or args.tbptt_chunk < 0
            or any(not math.isfinite(rate) or rate <= 0 for rate in (args.visual_learning_rate, args.gru_learning_rate))):
        raise ValueError("Invalid validation fraction, TBPTT chunk or learning rate")
    if any(value < args.negative_ratio + 1 or value % (args.negative_ratio + 1) for value in (args.batch_size, args.eval_pairs)):
        raise ValueError("--batch-size and --eval-pairs must be positive multiples of negative_ratio + 1")
    visual_path = args.output.with_name(args.output.stem + ".visual.pt")
    report_path = args.output.with_name(args.output.stem + ".metrics.json")
    if args.archive is not None and args.archive.resolve() in {path.resolve() for path in (args.output, visual_path, report_path)}:
        raise ValueError("Classifier output files must not replace the input archive")
    if any(path.exists() for path in (args.output, visual_path, report_path)) and not args.overwrite:
        raise FileExistsError("Choose fresh output paths or supply --overwrite")
    args.seed = secrets.randbelow(2**32) if args.seed is None else args.seed
    if not 0 <= args.seed < 2**32:
        raise ValueError("Seed must lie in [0, 2**32)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    train_ids, test_ids = split_episodes(episodes, metadata, args.test_episodes, args.all_data)
    rng = np.random.default_rng(args.seed)
    if args.all_data:
        fit_ids, validation_ids = list(train_ids), []
        selection_ids = fit_ids
        selection_scope = "training_episode_diagnostics_not_heldout"
    else:
        order = rng.permutation(len(train_ids))
        validation_count = min(len(train_ids) - 1, max(1, round(len(train_ids) * args.validation_fraction)))
        validation_ids = [train_ids[int(index)] for index in order[:validation_count]]
        fit_ids = [train_ids[int(index)] for index in order[validation_count:]]
        selection_ids = validation_ids
        selection_scope = "heldout_validation_episodes"
    by_id = {episode["id"]: episode for episode in episodes}
    fit_frames = [by_id[index]["frames"] for index in fit_ids]
    validation_frames = [by_id[index]["frames"] for index in selection_ids]
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(resolved_device=device, negative_mode="within_episode_random", visual_encoder_frozen_in_gru_stage=True)
    provenance = dict(archive_sha256=source_sha256, training_episode_ids=train_ids,
                      evaluation_episode_ids=test_ids, internal_train_ids=fit_ids, internal_validation_ids=validation_ids,
                      diagnostic_episode_ids=fit_ids if args.all_data else [], selection_scope=selection_scope,
                      train_config=config, threshold=.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history, started = [], time.monotonic()
    visual = VisualClassifier(args.architecture, args.embedding_dim).to(device)
    print(json.dumps(dict(event="start", device=device, seed=args.seed, fit_episodes=len(fit_ids),
                          validation_episodes=len(validation_ids), heldout_episodes=len(test_ids),
                          selection_scope=selection_scope)), flush=True)

    def train_stage(model, fit_data, validation_data, steps, rate, stage, output):
        recurrent = stage == "gru"
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=rate, weight_decay=1e-4)
        sampler = PairSampler(fit_data, args.window, args.negative_ratio, args.seed + (20 if recurrent else 10))
        validation_sampler = PairSampler(validation_data, args.window, args.negative_ratio, args.seed + 1)
        pairs = validation_sampler.sample(args.eval_pairs)
        best_key = (-1., -float("inf"))
        best_checkpoint = None
        for step in range(1, steps + 1):
            model.train()
            first, second, labels = sampler.sample(args.batch_size)
            indices = np.concatenate((first, second))
            if recurrent:
                embeddings = sampled_embeddings(model, fit_data, sampler, indices, device, args.tbptt_chunk)
            else:
                embeddings = model.encoder(frames_to_tensor(get_frames(fit_data, sampler, indices), device))
            logits = model.pair_logits(*embeddings.chunk(2))
            loss = balanced_loss(logits, torch.from_numpy(labels).to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite classifier loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            if step % args.eval_every == 0 or step == steps:
                metrics = evaluate_pairs(model, validation_data, validation_sampler, pairs, device, recurrent)
                record = dict(stage=stage, step=step, batch_loss=float(loss.detach()), gradient_norm=float(norm),
                              validation=metrics, selection_scope=selection_scope,
                              elapsed_seconds=time.monotonic() - started)
                history.append(record)
                print(json.dumps(record), flush=True)
                key = (metrics["balanced_accuracy"], -metrics["loss"])
                if key > best_key:
                    best_key = key
                    checkpoint = dict(kind="standalone_gru_classifier" if recurrent else "standalone_visual_classifier",
                        format_version=1, model_config=model.model_config, model_state=model.state_dict(),
                        model_fingerprint=model_fingerprint(model), optimizer_state=optimizer.state_dict(),
                        sampler_rng_state=sampler.rng.bit_generator.state, step=step, metrics=metrics,
                        **provenance)
                    atomic_save(output, checkpoint)
                    best_checkpoint = {key: value for key, value in checkpoint.items()
                                       if key not in ('model_state', 'optimizer_state')}
                    best_checkpoint['model_state'] = copy_model_state(model)
                report_path.write_text(json.dumps(dict(**provenance, history=history), indent=2) + "\n")
        return best_checkpoint

    visual_saved = train_stage(visual, fit_frames, validation_frames, args.visual_steps,
                               args.visual_learning_rate, "visual", visual_path)
    visual.load_state_dict(visual_saved["model_state"])
    recurrent = RecurrentClassifier(args.architecture, args.embedding_dim, args.hidden_dim).to(device)
    recurrent.encoder.load_state_dict(visual.encoder.state_dict())
    del visual
    fit_features = cache_features(recurrent, fit_frames, device)
    validation_features = fit_features if args.all_data else cache_features(recurrent, validation_frames, device)
    provenance.update(visual_checkpoint_sha256=file_sha256(visual_path), visual_best_step=visual_saved["step"])
    del visual_saved
    checkpoint = train_stage(recurrent, fit_features, validation_features, args.gru_steps,
                              args.gru_learning_rate, "gru", args.output)
    recurrent.load_state_dict(checkpoint['model_state'])
    recurrent.zero_grad(set_to_none=True)
    recurrent.eval().requires_grad_(False)
    print(json.dumps(dict(event="saved", classifier=str(args.output), fingerprint=model_fingerprint(recurrent),
                          selected_gru_step=checkpoint["step"], heldout_episodes_used_for_training=0)), flush=True)
    return recurrent, checkpoint


def main(argv=None):
    args = parse_args(argv)
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    torch.set_num_threads(args.threads)
    episodes, metadata = load_archive(args.archive)
    return train_from_episodes(args, episodes, metadata, file_sha256(args.archive))


if __name__ == "__main__":
    main()
