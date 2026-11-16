# GRU classifier and supervised goal-conditioned agent

This directory has exactly two Python modules and this guide:

1. **`train_gru_classifier.py`** reads trajectories from a `.pkl`, trains the
   temporal similarity classifier, and saves its checkpoint.
2. **`train_supervised_agent.py`** reads the same `.pkl` and that checkpoint,
   extracts anchors, trains a supervised policy, and evaluates it in Atari.

The second module imports shared archive/model utilities from the first. Keep
the two files together. They have no imports from the original project and do
not require R2-Dreamer, its world model, A2C, SAC, or the augmentation experiments.

## Input: an existing trajectory pickle

With periodic return exploration enabled in the main R2-Dreamer trainer, each
cycle writes a compatible snapshot of its completed source trajectories at:

```text
<logdir>/return_exploration/round_000001/episodes.pkl
```

Use the snapshot path for `--archive` in both commands below. It contains that
cycle's set S, with aligned executed actions, the action count, and environment
preprocessing configuration; no `--num-actions` or `--env-config` override is
needed. Evaluation rollouts and unfinished episodes are excluded. The first
cycle uses ordinary training episodes completed so far. Subsequent cycles use
the previous cycle's 100 assisted episodes plus ordinary episodes completed
since the previous snapshot. Older source episodes are excluded.

The standalone default train/validation/holdout split needs at least three
completed episodes. The integrated cycle uses `--all-data`, described below,
and can fit a nonempty snapshot without reserving episodes. See the repository
README for the MontezumaRevenge launcher and return-exploration configuration.

The existing H.E.R.O. archive works directly:

```text
../artifacts/hero/r2dreamer_sampled256/episodes.pkl
```

The loader supports its `atari_episode_archive` format, including compressed
`payload_npz` episode records. It also accepts a plain list of episode
dictionaries, or `{"episodes": [...], "metadata": {...}}` with uncompressed
arrays. Each episode contains:

| Field | Meaning |
|---|---|
| `id` | Unique integer; defaults to its position in a plain list |
| `frames` | `uint8` RGB array with shape `[N+1, 84, 84, 3]` |
| `actions` | Integer array `[N]`; action `t` takes frame `t` to frame `t+1` |
| `reset_seed` | Original reset seed, when available |
| `complete` | Must be true if supplied |

The classifier needs frames only. Supervised action learning requires the
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

## Step 1: train the GRU classifier

From this directory, using the existing `r2dreamer` conda environment:

```bash
conda activate r2dreamer
python train_gru_classifier.py \
  --archive ../artifacts/hero/r2dreamer_sampled256/episodes.pkl \
  --output ../artifacts/minimal_pipeline/classifier.pt \
  --architecture resnet18 --visual-steps 5000 --gru-steps 5000 \
  --window 20 --device cuda:0
```

For episode frames `f[0], ..., f[T]`, a CNN produces visual features and a causal
GRU produces contextual embeddings `z[t]`. Each embedding depends on the actual
episode prefix through `t`. The symmetric similarity head receives
`abs(z[i] - z[j])` and `z[i] * z[j]`, and predicts a binary similarity probability.

The temporal skip-gram labels are:

- **Positive:** two distinct frames from the same episode, at most 20 saved
  observation steps apart by default.
- **Negative:** select an episode in proportion to its length, then draw both
  frame positions independently and uniformly within it. Nearby pairs and
  identical positions are allowed. No far-away exclusion is applied.

Both stages use binary cross-entropy, with equal positive/negative class weight.
There are equal numbers of positives and negatives by default. No manually
annotated goals, actions, rewards, or room labels train the classifier.

The single command performs the same two-stage procedure as the previous code:

1. Train a visual pair classifier, including the ResNet-18 encoder.
2. Freeze the selected visual encoder, cache its features, and train the causal
   GRU and similarity head. Recurrent samples retain their complete source
   prefixes; `--tbptt-chunk` can detach gradients between chunks without resetting
   history. The frozen CNN remains in evaluation mode.

An internal validation split drawn only from the training episodes selects
checkpoints. Held-out episodes are not used for either stage. The output is
`classifier.pt`, with a visual-stage checkpoint `classifier.visual.pt` and
training metrics `classifier.metrics.json` beside it. The checkpoint includes
the archive hash, split, threshold, model settings, and weights.

## Step 2: train the supervised agent

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
new policy in a fresh output directory. The step 1 classifier remains compatible.

`--updates` is the total target, including resumed updates. Keep the same model,
batch, unroll, and optimizer settings. New runs draw optimizer randomness from
the operating system unless `--seed` is supplied. This seed does not replay the
source environment starts; offline fitting does not reset an environment.

## Use these models in periodic return exploration

The main trainer pauses ordinary R2-Dreamer collection every 100,000 ordinary
environment interactions. Its 50-million-interaction budget counts only that
ordinary collection; assisted episodes and their Dreamer takeovers are logged
separately. Each cycle generates exactly 100 assisted episodes.

For each source snapshot S, the integrated path trains a fresh classifier with
`--all-data`. This overrides any declared episode split: every episode is used
for fitting, and there is no evaluation holdout or internal validation-episode
split. Checkpoint selection uses independently sampled diagnostic pairs from
those same training episodes. Metrics label this scope as
`training_episode_diagnostics_not_heldout`; these are training diagnostics, not
estimates of generalization. All other classifier defaults stay the same:
ResNet-18, 5,000 visual updates, 5,000 GRU updates, and temporal window 20.

The supervised trainer uses all classifier training IDs. Pass
`--finetune PREVIOUS_OUTPUT/latest.pt` to carry the previous return-policy
parameters **and Adam optimizer state** into a fresh output directory with the
new archive and classifier. It resets the per-dataset sampling schedule and
checkpoint-selection score. `--updates 10000` then performs **10,000 additional
updates** for this cycle. This differs from `--resume`, which continues the same
dataset and classifier toward a total update target in the same output folder.
The two options are mutually exclusive. Fine-tuning requires a compatible
policy architecture and action mapping; it rejects a known change of game.

For example, from this directory, with paths replaced by actual cycle outputs:

```bash
python train_gru_classifier.py \
  --archive /path/to/new_round/episodes.pkl \
  --output /path/to/new_round/classifier.pt \
  --all-data --device cuda:0

python train_supervised_agent.py train \
  --archive /path/to/new_round/episodes.pkl \
  --classifier /path/to/new_round/classifier.pt \
  --output /path/to/new_round/agent \
  --finetune /path/to/previous_round/agent/latest.pt \
  --skip-empty-anchors --device cuda:0
```

Omit `--finetune` for the first cycle of a new policy. The integrated trainer
supplies the previous cycle's checkpoint automatically after that.
`--skip-empty-anchors` reports and omits trajectories that have no extracted
anchors from supervised fitting; it never inserts artificial goals. Those
trajectories still belong to S and train the classifier. If no usable goals
remain, the auxiliary policy trainer raises `NoUsableAnchorsError`. Without the
flag, the standalone trainer retains its strict error on any empty anchor list.

Supervised training still conditions on each trajectory's **full anchor
sequence**, using the default batch size 4 and unroll length 2,048. During
assisted collection, the return policy receives a sampled source anchor prefix
ending at the selected target. Prefixes are inference inputs; this mode does
not introduce additional prefix-conditioned training examples. The classifier
checks arrival, and Dreamer then continues in the same live environment with
context from the actual generated history. Returning successfully is not
guaranteed. All-data checkpoints have no held-out goals; use `--split train`
if running the standalone evaluation command on one of these checkpoints.

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
each attempt also has colored 84×84 `source.mp4`, `rollout.mp4`, and individual
target anchor PNGs. Poor imitation fit is a reason to improve training before
spending compute on these rollouts. Temporal similarity remains a learned proxy
for accomplishment; the saved videos support visual inspection.

Training requires PyTorch and NumPy. The second module also uses Pillow;
environment evaluation uses Gymnasium and ALE, and video output uses OpenCV.
The existing `r2dreamer` environment provides these packages. Use `--help` on
either module or either agent subcommand for all arguments. These two modules
produce their own portable checkpoint format; use the classifier checkpoint
from step 1 with the agent in step 2.
