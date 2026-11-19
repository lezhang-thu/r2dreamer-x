"""Train a causal GRU pair classifier on frozen DINOv3 ViT-L/16 frame embeddings.

Positive pairs are nearby distinct frames. Negative pairs come from two
independently sampled positions within the same episode. Each GRU context
starts at the episode reset, including for a trajectory cut at collection end.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import secrets
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


from atari_data import file_sha256, frames_to_tensor, load_archive, state_digest
from extract_dino_features import DinoFrameEncoder, FEATURE_DIM, load_features


def split_episodes(episodes, metadata, test_episodes=None, all_data=False):
    """Preserve declared split IDs, or reserve the latest episodes as holdout."""
    ids = [episode["id"] for episode in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate episode IDs")
    if all_data:
        if test_episodes is not None:
            raise ValueError("--all-data cannot be combined with --test-episodes")
        if not ids:
            raise ValueError("Classifier fitting needs at least one episode")
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


class RecurrentClassifier(nn.Module):
    """Train on per-frame CLS vectors. Attach their frozen encoder for pixel inference."""
    is_recurrent = True

    def __init__(self, backbone, backbone_sha256, hidden_dim=128, num_layers=1, revision=None):
        super().__init__()
        self.model_config = dict(backbone=backbone, revision=revision, hidden_dim=hidden_dim, num_layers=num_layers)
        self.backbone_sha256 = backbone_sha256
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.gru = nn.GRU(FEATURE_DIM, hidden_dim, num_layers=num_layers, batch_first=True)
        self.encoder = None

    def attach_encoder(self, encoder):
        if encoder.backbone_sha256 != self.backbone_sha256:
            raise ValueError("DINOv3 encoder weights differ from the feature source")
        self.encoder = encoder.to(self.gru.weight_ih_l0.device).eval().requires_grad_(False)

    def pair_logits(self, first, second):
        return self.head(torch.cat(((first - second).abs(), first * second), dim=-1)).squeeze(-1)

    def train(self, mode=True):
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
        return self

    @torch.no_grad()
    def encode_visual(self, images):
        if self.encoder is None:
            raise ValueError("RGB inference requires the DINOv3 encoder")
        return self.encoder(images)

    def encode_features(self, features, hidden=None, lengths=None):
        if features.ndim != 3 or features.shape[-1] != self.gru.input_size:
            raise ValueError("Unexpected GRU input feature shape")
        batch, steps, _ = features.shape
        if hidden is None:
            hidden = features.new_zeros((self.num_layers, batch, self.hidden_dim))
        if hidden.shape != (self.num_layers, batch, self.hidden_dim):
            raise ValueError("Wrong recurrent hidden-state shape")
        if not steps:
            return features.new_zeros((batch, 0, self.hidden_dim)), hidden
        if lengths is None:
            return self.gru(features, hidden)
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=features.device)
        if lengths.shape != (batch,) or torch.any(lengths < 0) or torch.any(lengths > steps):
            raise ValueError("Invalid sequence lengths")
        nonempty = lengths > 0
        output, state = features.new_zeros((batch, steps, self.hidden_dim)), hidden.clone()
        if nonempty.any():
            packed = pack_padded_sequence(features[nonempty], lengths[nonempty].cpu(), batch_first=True, enforce_sorted=False)
            encoded, next_state = self.gru(packed, hidden[:, nonempty].contiguous())
            values, _ = pad_packed_sequence(encoded, batch_first=True, total_length=steps)
            output[nonempty] = values
            state[:, nonempty] = next_state
        return output, state

    def encode_step(self, images, hidden=None):
        output, state = self.encode_features(self.encode_visual(images)[:, None], hidden)
        return output[:, 0], state

    def classifier_state_dict(self):
        return {name: value for name, value in self.state_dict().items() if not name.startswith("encoder.")}

    def load_classifier_state(self, state):
        if state.keys() != self.classifier_state_dict().keys():
            raise ValueError("Checkpoint does not contain the expected GRU and pair head weights")
        self.load_state_dict(state, strict=False)


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


@torch.no_grad()
def encode_feature_trajectory(model, features, device="cpu", batch_size=256):
    """Unroll one episode of stored CLS vectors from reset, preserving state between chunks."""
    if batch_size < 1 or len(features) < 1:
        raise ValueError("Need a positive batch size and nonempty features")
    model.eval()
    pieces, hidden = [], None
    for start in range(0, len(features), batch_size):
        chunk = features[start:start + batch_size].to(device)
        encoded, hidden = model.encode_features(chunk.unsqueeze(0), hidden)
        pieces.append(encoded[0].cpu())
    return torch.cat(pieces)


def model_fingerprint(model):
    digest = hashlib.sha256(json.dumps(model.model_config, sort_keys=True).encode())
    if isinstance(model, RecurrentClassifier):
        digest.update(model.backbone_sha256.encode())
        return state_digest(digest, model.classifier_state_dict()).hexdigest()
    return state_digest(digest, model.state_dict()).hexdigest()


def load_classifier(path, device="cpu", *, load_encoder=True):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("kind") != "standalone_gru_classifier" or saved.get("format_version") != 2:
        raise ValueError("Expected a DINOv3 GRU classifier checkpoint")
    model = RecurrentClassifier(**saved["model_config"], backbone_sha256=saved["backbone_sha256"])
    model.load_classifier_state(saved["model_state"])
    if model_fingerprint(model) != saved["model_fingerprint"]:
        raise ValueError("Classifier state does not match its saved fingerprint")
    model.to(device)
    if load_encoder:
        config = saved["model_config"]
        model.attach_encoder(DinoFrameEncoder(config["backbone"], revision=config["revision"]))
    model.eval().requires_grad_(False)
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


def pack_feature_chunk(sequences, lengths, start, device):
    """Pack sorted CPU features directly in time-major order, without padding.

    At time t the active sequences occupy contiguous rows. Their first row
    is offsets[t], so endpoint (episode rank i, t) lives at offsets[t] + i.
    """
    batch_sizes = np.searchsorted(-lengths, -np.arange(lengths[0]))
    offsets = np.cumsum(batch_sizes) - batch_sizes
    values = sequences[0].new_empty((lengths.sum(), sequences[0].shape[1]))
    for rank, (sequence, length) in enumerate(zip(sequences, lengths)):
        rows = torch.as_tensor(offsets[:length] + rank, device=values.device)
        values.index_copy_(0, rows, sequence[start:start + length])
    return PackedSequence(values.to(device), torch.from_numpy(batch_sizes)), offsets


def sampled_embeddings(model, features, sampler, indices, device, tbptt_chunk=0):
    episode, local = sampler.locate(np.asarray(indices))
    unique, inverse = np.unique(episode, return_inverse=True)
    lengths = np.zeros(len(unique), dtype=np.int64)
    np.maximum.at(lengths, inverse, local + 1)
    order = np.argsort(-lengths, kind="stable")
    sequences = [features[e] for e in unique[order]]
    lengths = lengths[order]
    ranks = np.argsort(order)[inverse]
    chunk_size = tbptt_chunk or lengths[0]

    # Keep one recurrent row per active episode. Only sampled endpoints
    # contribute to the pair loss, so no dense [episodes, max_T, 1024]
    # input or padded recurrent output needs to be materialized.
    positions, contexts, hidden = [], [], None
    for start in range(0, lengths[0], chunk_size):
        active = lengths[lengths > start]
        chunk_lengths = np.minimum(active - start, chunk_size)
        packed, offsets = pack_feature_chunk(sequences, chunk_lengths, start, device)
        if hidden is not None:
            hidden = hidden[:, :len(active)].detach().contiguous()
        output, hidden = model.gru(packed, hidden)
        selected = np.flatnonzero((local >= start) & (local < start + chunk_size))
        if len(selected):
            rows = offsets[local[selected] - start] + ranks[selected]
            contexts.append(output.data[torch.as_tensor(rows, device=device)])
            positions.append(selected)
    restore_order = torch.as_tensor(np.argsort(np.concatenate(positions)), device=device)
    return torch.cat(contexts)[restore_order]


def balanced_loss(logits, labels):
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (losses[labels == 1].mean() + losses[labels == 0].mean()) / 2


@torch.no_grad()
def evaluate_pairs(model, data, sampler, pairs, device):
    model.eval()
    first, second, labels = pairs
    logits = []
    for start in range(0, len(labels), 128):
        a, b = first[start:start + 128], second[start:start + 128]
        embeddings = sampled_embeddings(model, data, sampler, np.concatenate((a, b)), device)
        logits.append(model.pair_logits(*embeddings.chunk(2)).cpu())
    logits, target = torch.cat(logits), torch.from_numpy(labels)
    positive = target == 1
    predicted = logits.sigmoid() >= .5
    return dict(loss=float(balanced_loss(logits, target)), accuracy=float((predicted == positive).float().mean()),
                balanced_accuracy=float(((predicted[positive]).float().mean() + (~predicted[~positive]).float().mean()) / 2),
                positive_recall=float(predicted[positive].float().mean()), negative_recall=float((~predicted[~positive]).float().mean()))


def atomic_save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def copy_model_state(model):
    """Keep a best-fit snapshot without retaining live GPU tensors."""
    return {name: tensor.detach().to('cpu', copy=True) for name, tensor in model.state_dict().items()}


def parse_args(argv=None, *, require_features=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=require_features,
                        help="Embedding file from extract_dino_features.py")
    parser.add_argument("--output", type=Path, required=True, help="Final recurrent classifier .pt file")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gru-steps", type=int, default=5000)
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


def train_from_features(args, episodes, metadata, provenance):
    """Fit the GRU on stored or in-memory DINOv3 feature sequences."""
    for name in ("hidden_dim", "gru_steps", "window", "negative_ratio", "threads", "eval_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (not 0 < args.validation_fraction < 1 or args.tbptt_chunk < 0
            or not math.isfinite(args.gru_learning_rate) or args.gru_learning_rate <= 0):
        raise ValueError("Invalid validation fraction, TBPTT chunk or learning rate")
    if any(value < args.negative_ratio + 1 or value % (args.negative_ratio + 1) for value in (args.batch_size, args.eval_pairs)):
        raise ValueError("--batch-size and --eval-pairs must be positive multiples of negative_ratio + 1")
    report_path = args.output.with_name(args.output.stem + ".metrics.json")
    if args.features is not None and args.features.resolve() in {path.resolve() for path in (args.output, report_path)}:
        raise ValueError("Classifier output files must not replace the feature file")
    if any(path.exists() for path in (args.output, report_path)) and not args.overwrite:
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
    fit_features = [by_id[index]["features"] for index in fit_ids]
    validation_features = [by_id[index]["features"] for index in selection_ids]
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(resolved_device=device, negative_mode="within_episode_random", frame_encoder="dinov3_vitl16_cls",
                  preprocessing=provenance["preprocessing"])
    training_info = dict(archive_sha256=provenance["archive_sha256"], training_episode_ids=train_ids,
                      evaluation_episode_ids=test_ids, internal_train_ids=fit_ids, internal_validation_ids=validation_ids,
                      diagnostic_episode_ids=fit_ids if args.all_data else [], selection_scope=selection_scope,
                      train_config=config, threshold=.5)
    if "features_sha256" in provenance:
        training_info["features_sha256"] = provenance["features_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history, started = [], time.monotonic()
    print(json.dumps(dict(event="start", device=device, seed=args.seed, fit_episodes=len(fit_ids),
                          validation_episodes=len(validation_ids), heldout_episodes=len(test_ids),
                          selection_scope=selection_scope)), flush=True)

    recurrent = RecurrentClassifier(
        backbone=provenance["backbone"], revision=provenance["revision"],
        backbone_sha256=provenance["backbone_sha256"], hidden_dim=args.hidden_dim).to(device)
    print(json.dumps(dict(event="features_ready", fit_frames=sum(map(len, fit_features)),
                          feature_dim=recurrent.gru.input_size)), flush=True)
    optimizer = torch.optim.AdamW((p for p in recurrent.parameters() if p.requires_grad),
                                  lr=args.gru_learning_rate, weight_decay=1e-4)
    sampler = PairSampler(fit_features, args.window, args.negative_ratio, args.seed + 20)
    validation_sampler = PairSampler(validation_features, args.window, args.negative_ratio, args.seed + 1)
    pairs = validation_sampler.sample(args.eval_pairs)
    best_key = (-1., -float("inf"))
    for step in range(1, args.gru_steps + 1):
        recurrent.train()
        first, second, labels = sampler.sample(args.batch_size)
        indices = np.concatenate((first, second))
        embeddings = sampled_embeddings(recurrent, fit_features, sampler, indices, device, args.tbptt_chunk)
        logits = recurrent.pair_logits(*embeddings.chunk(2))
        loss = balanced_loss(logits, torch.from_numpy(labels).to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite classifier loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(recurrent.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.gru_steps:
            metrics = evaluate_pairs(recurrent, validation_features, validation_sampler, pairs, device)
            record = dict(stage="gru", step=step, batch_loss=float(loss.detach()), gradient_norm=float(norm),
                          validation=metrics, selection_scope=selection_scope,
                          elapsed_seconds=time.monotonic() - started)
            history.append(record)
            print(json.dumps(record), flush=True)
            key = (metrics["balanced_accuracy"], -metrics["loss"])
            if key > best_key:
                best_key = key
                best_state = {name: tensor.detach().to("cpu", copy=True)
                              for name, tensor in recurrent.classifier_state_dict().items()}
                checkpoint = dict(kind="standalone_gru_classifier", format_version=2,
                                  model_config=recurrent.model_config, model_state=best_state,
                                  backbone_sha256=recurrent.backbone_sha256,
                                  model_fingerprint=model_fingerprint(recurrent),
                                  sampler_rng_state=sampler.rng.bit_generator.state,
                                  step=step, metrics=metrics, **training_info)
                atomic_save(args.output, dict(checkpoint, optimizer_state=optimizer.state_dict()))
            report_path.write_text(json.dumps(dict(**training_info, history=history), indent=2) + "\n")
    recurrent.load_classifier_state(checkpoint["model_state"])
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
    episodes, metadata, provenance = load_features(args.features)
    provenance["features_sha256"] = file_sha256(args.features)
    return train_from_features(args, episodes, metadata, provenance)


if __name__ == "__main__":
    main()
