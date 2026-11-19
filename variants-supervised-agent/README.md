# GRU classifier and supervised goal-conditioned agent

The standalone workflow uses three files.

| File | Contents | Used by |
|---|---|---|
| `episodes.pkl` | Original RGB screens and aligned trajectory data | Feature extraction, anchor PNG export, supervised policy |
| `embeddings.pt` | One 1,024-value float32 DINOv3 CLS vector per screen, grouped by episode | GRU training and anchor selection |
| `classifier.pt` | Trained GRU, similarity head, split and source fingerprints | Anchor selection and supervised policy |

## Input trajectories

With periodic return exploration enabled, the main trainer keeps completed
source trajectories compressed in RAM. It calls the classifier and policy
training functions directly in the same process. Each cycle decodes its source episodes once. GRU fitting and anchor selection
share the extracted DINOv3 vectors in memory. The decoded frames and actions
supply the supervised policy examples.
No training subprocess, memory-file snapshot, or trajectory pickle is needed.
The in-memory data includes aligned executed actions, the action count, and
environment preprocessing configuration, so no `--num-actions` or
`--env-config` override is needed. The integrated path provides no archive
path for later standalone use.
Unfinished episodes are excluded. The first cycle uses
ordinary training episodes completed so far. Subsequent cycles use the previous
cycle's 100 assisted episodes plus ordinary episodes completed since the
previous source selection. Older source episodes are excluded.

The standalone default train/validation/holdout split needs at least three
trajectories of at least two frames. The integrated cycle uses `--all-data`, described below,
and can fit a nonempty source set without reserving episodes. See the repository
README for the MontezumaRevenge launcher and return-exploration configuration.

The [MontezumaRevenge collector](../README.md#collect-100k-montezumarevenge-interactions-for-the-gru)
writes uncompressed native screens to `logdir/montezuma_100k_0/episodes.pkl`.
Use `load_archive(path)` in other experiments to get the episode dictionaries
and metadata. The version 2 file starts with a pickled header, followed by the
contiguous RGB pixels, a pickled episode index, and an eight-byte little-endian
offset to the index. A single `pickle.load` reads only the header. The supplied
reader maps the pixels read-only and exposes a NumPy array for each trajectory.
For example, from this directory:

```python
from atari_data import load_archive

episodes, metadata = load_archive("../logdir/montezuma_100k_0/episodes.pkl")
assert sum(len(ep["actions"]) for ep in episodes) == 100_000
print(episodes[0]["frames"].shape, metadata["action_meanings"])
```

The existing H.E.R.O. archive also works directly.

```text
../artifacts/hero/r2dreamer_sampled256/episodes.pkl
```

The loader supports its version 1 `atari_episode_archive` format, including compressed
`payload_npz` episode records. It also accepts a plain list of episode
dictionaries, or `{"episodes": [...], "metadata": {...}}` with uncompressed
arrays. Each episode contains:

| Field | Meaning |
|---|---|
| `id` | Unique integer; defaults to its position in a plain list |
| `frames` | `uint8` RGB array with shape `[N+1, 210, 160, 3]` (native) or `[N+1, 84, 84, 3]` (legacy) |
| `actions` | Integer array `[N]`; action `t` takes frame `t` to frame `t+1` |
| `reset_seed` | Original reset seed, when available |
| `complete` | False with `collection_cutoff=True` for the final live prefix of a native run |

DINOv3 feature extraction needs the frames only. Supervised action learning requires the
recorded actions; it does not infer them from images. Optional rewards and end
flags are preserved but are not supervision targets.

Metadata can provide `training_episode_ids`, `evaluation_episode_ids`,
`num_actions`, `action_meanings`, and `env_config`, as the H.E.R.O. archive does.
Its declared split is preserved: 224 training episodes and 32 held-out episodes.
Without a declared split, the classifier reserves the latest 12.5% of episodes
as holdout; `--test-episodes` can set that count. At least two remaining episodes
are needed for the classifier's internal train/validation split. This split is
stored in the classifier checkpoint and reused by the supervised agent.

For a plain archive without action-space metadata, supply `--num-actions` to
the agent trainer. Environment evaluation also needs the source preprocessing
configuration, either in `metadata.env_config` or via `--env-config` with a JSON
object. The configuration must describe how the source images were produced.

## Extract the DINOv3 features once

```bash
python variants-supervised-agent/extract_dino_features.py \
  --archive logdir/montezuma_100k_0/episodes.pkl \
  --output logdir/montezuma_100k_0/embeddings.pt --device cuda:0
```

The extractor preserves episode IDs, their order, split metadata, and all frames
from reset through the final screen, including collection cutoffs. Frame index
`t` in an episode's feature tensor identifies frame `t` in its RGB array.
`--feature-batch-size` controls how many frames go through DINOv3 at once
(default 16). The file records the source archive hash, backbone and preprocessing.

## Step 1: train the GRU classifier

From the repository root, after collecting the MontezumaRevenge archive, run

```bash
conda activate r2dreamer
python variants-supervised-agent/train_gru_classifier.py \
  --features logdir/montezuma_100k_0/embeddings.pt \
  --output logdir/montezuma_100k_0/gru-run1/classifier.pt \
  --gru-steps 5000 --window 20 --seed 0 --device cuda:0
```

The frozen backbone is Meta's [DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
Authenticate with an account that has access to the weights, or supply a local
Hugging Face snapshot directory to the extractor through `--backbone`. A native 210×160 screen
gets seven black rows above and below and ImageNet channel normalization. The
resulting 224×160 image has a 14×10 patch grid. The classifier takes one
1,024-dimensional final CLS vector per frame and feeds it directly to a causal
GRU. Every context `z[t]` incorporates the episode prefix through frame `t`.
The symmetric head scores `abs(z[i] - z[j])` and `z[i] * z[j]`.

The temporal skip-gram labels are:

- **Positive:** two distinct frames from the same episode, at most 20 saved
  observation steps apart by default.
- **Negative:** select an episode in proportion to its length, then draw both
  frame positions independently and uniformly within it. Nearby pairs and
  identical positions are allowed. No far-away exclusion is applied.

The GRU trainer memory-maps the saved features. Its only data input is
`embeddings.pt`. On an RTX 4090 the extractor uses bfloat16 autocast and saves
float32 embeddings. One
optimizer fits the GRU and similarity head with class-balanced binary
cross-entropy. Recurrent samples retain their complete source prefixes.
`--tbptt-chunk` detaches gradients between chunks while preserving history.

Validation episodes select the best checkpoint, and test episodes remain
reserved. The run writes `classifier.pt` and `classifier.metrics.json`.
The DINOv3 GRU checkpoint records the split, threshold, trainable weights,
backbone identity, and hashes of the original archive and feature file. Pixel
inference (for example, a live Atari rollout) also needs the frozen backbone.
Repeated GRU fits and anchor export with `--features` only use the saved vectors
for model inference.

## Export the merged anchor set

From the repository root, run this after fitting the classifier on the native
MontezumaRevenge archive. For experiments that fit all 100k interactions, add
`--all-data` to the classifier **training** command above.

```bash
python variants-supervised-agent/export_anchor_set.py \
  --archive logdir/montezuma_100k_0/episodes.pkl \
  --features logdir/montezuma_100k_0/embeddings.pt \
  --classifier logdir/montezuma_100k_0/gru-run1/classifier.pt \
  --output logdir/montezuma_100k_0/gru-run1/anchor_set --device cuda:0
```

The exporter verifies the source files against the checkpoint, scores the saved
vectors, and copies selected frames from the RGB archive into PNGs. Omitting
`--features` computes the vectors from RGB during export. It processes every trajectory in collection order. This includes held-out episodes
when the classifier used a split, and the final recorded prefix if collection
ended during a game. Use `--complete-only` to restrict the export to completed
episodes, as in the online return cycle.

Within each trajectory, the GRU starts at reset. Frame zero is the initial
reference. The first later frame with similarity below the checkpoint threshold
becomes an anchor and the next reference. Across trajectories, the exporter
applies the same representative selection as the online anchor bank. A candidate
increments the count of every existing representative it matches at or above
the threshold. A candidate with zero matches becomes a new representative.
Both comparisons use the learned similarity head on the GRU contexts.

Open the output directory to inspect one lossless 160-pixel-wide by
210-pixel-high PNG per representative, numbered in selection order. The image
comes directly from the source frame. The manifest records the threshold,
candidate and representative counts, and each PNG's source episode, frame index,
anchor position and count. An empty set produces a manifest with zero
representatives. Use a fresh output directory for each classifier checkpoint.

## Step 2: train the supervised agent

The remaining examples use paths relative to this directory and the existing
H.E.R.O. dataset. Substitute the archive and classifier paths from your run.

```bash
python train_supervised_agent.py train \
  --archive ../artifacts/hero/r2dreamer_sampled256/episodes.pkl \
  --classifier ../artifacts/minimal_pipeline/classifier.pt \
  --output ../artifacts/minimal_pipeline/agent \
  --updates 10000 --batch-size 4 --unroll-length 2048 --device cuda:0
```

The classifier is frozen throughout anchor preparation, policy training, and
evaluation. Its causal embeddings are computed from frame zero for each source
trajectory. Starting with frame zero as the reference, extraction selects the
first later frame with similarity below the saved threshold (0.5), makes that
frame the new reference, and repeats. Frame zero is excluded from the goal list.
The terminal frame is included only if the classifier selects it.

Each anchor retains its RGB image and its source-history classifier embedding.
The latter must not be recomputed by resetting the GRU at the anchor image.
No per-frame progress or completed-anchor labels are computed. Empty target
anchor lists raise an error rather than being silently discarded; improve the
classifier before training BC in that case.

The supervised policy has its own CNN and two GRUs:

- An **anchor GRU** processes the ordered anchor images and retains one memory
  vector per anchor. Packed sequences handle different goal counts without
  treating padding as real images.
- A **controller GRU** processes only current visual features, retaining the
  observation history in its recurrent state. At every step, its state attends
  to **all anchor memories**. Only padding is masked; anchors remain available
  throughout the episode.

The action head combines the controller state and attention result. The policy
must learn which anchors matter from its observation history and the ordered
anchor sequence. There is no next-anchor pointer, progress fraction, completion
signal, or completed-anchor mask. The frozen classifier's contextual embeddings
are used for anchor extraction and evaluation, not as direct policy inputs.
Neither source seeds, trajectory IDs, nor recorded actions are policy inputs.

Training minimizes cross-entropy against the recorded action sequence. It uses
recorded observation histories and full goal lists. For a truncated window, the
current policy first reconstructs its hidden state from the source observation
prefix without gradients, then differentiates through the window. It does not
reset history at the window boundary. Training itself makes no environment calls.

The trainer saves `best.pt`, selected by training action accuracy and then loss,
and `latest.pt` for resume, plus JSON logs. This selection measures imitation
fit; it does not establish environment success. Resume in the same output folder:

```bash
python train_supervised_agent.py train \
  --archive ../artifacts/hero/r2dreamer_sampled256/episodes.pkl \
  --classifier ../artifacts/minimal_pipeline/classifier.pt \
  --output ../artifacts/minimal_pipeline/agent \
  --resume ../artifacts/minimal_pipeline/agent/latest.pt \
  --updates 75000 --batch-size 4 --unroll-length 2048 --device cuda:0
```

This architecture writes version 2 policy checkpoints. Earlier checkpoints with
explicit progress inputs cannot be resumed or evaluated by this variant; train a
new policy in a fresh output directory.

`--updates` is the total target, including resumed updates. Keep the same model,
batch, unroll, and optimizer settings. New runs draw optimizer randomness from
the operating system unless `--seed` is supplied. This seed does not replay the
source environment starts; offline fitting does not reset an environment.

## Use these models in periodic return exploration

The main trainer pauses ordinary R2-Dreamer collection every 100,000 ordinary
environment interactions. Its 50-million-interaction budget counts only that
ordinary collection; assisted episodes and their Dreamer takeovers are logged
separately. Each cycle generates exactly 100 assisted episodes.

For each source set S, the integrated path trains a fresh classifier with the
equivalent of `--all-data`. This overrides any declared episode split: every episode is used
for fitting, and there is no evaluation holdout or internal validation-episode
split. Checkpoint selection uses independently sampled diagnostic pairs from
those same training episodes. Metrics label this scope as
`training_episode_diagnostics_not_heldout`; these are training diagnostics, not
estimates of generalization. The integrated classifier extracts frozen DINOv3 features from the existing RGB84
cycle data and fits the GRU for 5,000 updates with temporal window 20.

The supervised trainer uses all classifier training IDs. The integrated path
continues with the same return-policy parameters **and AdamW optimizer state
in memory**. It keeps them on CPU between cycles and moves them to the configured
device for fitting. A new source set resets only the per-dataset sampling
schedule and checkpoint-selection score. Each cycle performs **10,000 additional
updates** by default, with the same `3e-4` learning rate. The best training-fit
weights are copied for returning; the latest weights and optimizer remain the
next cycle's starting point. Auxiliary fitting restores Dreamer's random-generator
states and CPU thread count afterward.

For standalone commands, pass `--finetune PREVIOUS_OUTPUT/latest.pt` to load
the previous parameters and optimizer state into a fresh output directory.
This differs from `--resume`, which continues the same dataset and classifier
toward a total update target in the same output folder. The two options are
mutually exclusive. Fine-tuning requires a compatible policy architecture and
action mapping; it rejects a known change of game.

For example, to fine-tune manually using a separately supplied archive, replace
the paths below with your archive and model locations. The integrated trainer
passes episode arrays directly and does not use these archive-loading commands:

```bash
python extract_dino_features.py \
  --archive /path/to/source/episodes.pkl \
  --output /path/to/source/embeddings.pt --device cuda:0

python train_gru_classifier.py \
  --features /path/to/source/embeddings.pt \
  --output /path/to/new_round/classifier.pt \
  --all-data --device cuda:0

python train_supervised_agent.py train \
  --archive /path/to/source/episodes.pkl \
  --classifier /path/to/new_round/classifier.pt \
  --output /path/to/new_round/agent \
  --finetune /path/to/previous_round/agent/latest.pt \
  --skip-empty-anchors --device cuda:0
```

Omit `--finetune` for the first standalone cycle of a new policy. The integrated
trainer initializes a new policy on its first cycle unless an `initial_policy`
checkpoint is supplied, then retains the live policy and optimizer for later cycles.

In the integrated trainer, source episodes are released after fitting and
goal-prefix preparation. The goal images, anchor
bank, and prefix lookup last only until that cycle's 100 assisted episodes
finish. Dreamer's separate replay also stays in RAM, with a capacity of
1 million stored transitions by default. Neither trajectory store is included
in model checkpoints.

Only the newest completed round's model directory and an in-progress round
are retained. A successful cycle removes the previous internally created round,
but never an externally supplied `initial_policy` checkpoint. Dreamer's
checkpoint is replaced atomically at `<logdir>/latest.pt` after each cycle and
at training completion. Anchor banks, prefix lookups, and counters are not
saved as JSON dumps. Small summaries in `return_exploration/cycles.jsonl`,
console logs, and metrics preserve the round history and all generated episode
returns. Existing archives and old round directories from earlier versions are
not automatically deleted. Standalone commands continue to read your supplied
archives without changing or deleting them.

`--skip-empty-anchors` reports and omits trajectories that have no extracted
anchors from supervised fitting; it never inserts artificial goals. Those
trajectories still belong to S and train the classifier. If no usable goals
remain, the auxiliary policy trainer raises `NoUsableAnchorsError`. Without the
flag, the standalone trainer retains its strict error on any empty anchor list.

Supervised training still conditions on each trajectory's **full anchor
sequence**, using the default batch size 4 and unroll length 2,048. During
assisted collection, the return policy receives a uniformly sampled matching
source anchor prefix ending at the selected target. Each matching occurrence
has equal probability, regardless of prefix length. Prefixes are inference
inputs; this mode does not introduce additional prefix-conditioned training
examples. The classifier
checks arrival, and Dreamer then continues in the same live environment with
context from the actual generated history. Returning successfully is not
guaranteed. For standalone classifiers trained with `--all-data`, use
`--split train` when evaluating against their original archive because they
have no held-out goals. Integrated checkpoints record a source-content
fingerprint rather than the hash of a pickle file; the integrated run does
not create an archive for the standalone evaluator.

## Evaluate only when training fit is satisfactory

Evaluation is a separate command and never runs automatically after training.
Evaluate all held-out goals by default:

```bash
python train_supervised_agent.py evaluate \
  --archive ../artifacts/hero/r2dreamer_sampled256/episodes.pkl \
  --classifier ../artifacts/minimal_pipeline/classifier.pt \
  --checkpoint ../artifacts/minimal_pipeline/agent/best.pt \
  --output ../artifacts/minimal_pipeline/evaluation \
  --split test --episodes 0 --device cuda:0 --save-videos
```

Use `--split train` for familiar goals, or `--split both` to report both groups.
`--episodes N` takes the latest N declared goals per split; `--episodes 0` uses
all goals (the default). `--repeats` controls independent attempts per goal.

Each attempt samples a fresh OS-generated reset seed, excluding previously used
evaluation seeds and collection seeds when recorded in the archive. Startup
NOOP counts follow the configured distribution normally: the same NOOP count
or initial RGB image as the source or another attempt is allowed. This also
supports `noops=0` and any positive number of repeats. Different seeds do not
guarantee different initial images or emulator states.
The evaluator checks the action mapping and preserves the archive's image
preprocessing—for the existing H.E.R.O. data, pooled RGB is resized to 64×64 and
then to 84×84. ALE/Gymnasium and the Atari ROM must be available in the environment.

The policy acts using its own observations, recurrent state, and the full target
anchor sequence. Every attempt runs through environment termination or its
normal time limit. The frozen classifier processes the generated history from
its reset, independently of the target history. Rollout anchors use exactly the
same extraction rule as target anchors: start with frame zero as the reference,
select the first later embedding whose similarity to the reference is below the
saved threshold, and repeat from that selected embedding. Frame zero is excluded;
the terminal frame is included only if selected.

Evaluation compares these two ordered sequences:

1. Classifier contextual embeddings at anchors extracted from the agent rollout.
2. Classifier contextual embeddings at the target anchors supplied to the policy
   through their images and anchor-GRU memories.

The score is the **length of the longest common subsequence (LCS)**. Two anchors
match when the frozen classifier's similarity probability is at least the saved
threshold (0.5 by default). Dynamic programming finds the largest order-preserving,
one-to-one set of matches. It may skip unmatched anchors in either sequence;
matches need not be contiguous. An empty rollout anchor sequence scores zero.
For example, if the target is `[A, B, C]` and the rollout is `[A, X, C]`, with
only identical letters similar, the LCS length is 2. No online completion pointer
is used for scoring or policy control.

Each attempt's `result.json` records `lcs_length`, target and rollout anchor counts,
and their source frame indices. `report.json` gives `mean_lcs_length` separately
for familiar and held-out goals, along with per-attempt results. These replace
the old strict-completion success and completion-fraction metrics. Raw LCS length
is bounded by the shorter anchor sequence and depends on target length.

The plan and report identify the reset protocol as
`reset_sampling: independent_seeds`. Each rollout records whether its initial
image matches the source and whether its reset seed matches the source seed
(`null` when the source seed is unavailable). Existing evaluations are unchanged;
rerun into a fresh output directory to use the LCS protocol. With `--save-videos`,
each attempt also has source-resolution RGB `source.mp4`, `rollout.mp4`, and individual
target anchor PNGs. Poor imitation fit is a reason to improve training before
spending compute on these rollouts. Temporal similarity remains a learned proxy
for accomplishment; the saved videos support visual inspection.

GRU training from a feature file requires PyTorch and NumPy. RGB feature extraction
requires Transformers and the DINOv3 weights. Anchor PNG export uses Pillow.
Environment evaluation uses Gymnasium and ALE, and video output uses OpenCV.
Use `--help` on each script for its inputs and options.
