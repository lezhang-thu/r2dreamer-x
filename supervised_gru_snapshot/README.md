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
  --window 5 --device cuda:0
```

For episode frames `f[0], ..., f[T]`, a CNN produces visual features and a causal
GRU produces contextual embeddings `z[t]`. Each embedding depends on the actual
episode prefix through `t`. The symmetric similarity head receives
`abs(z[i] - z[j])` and `z[i] * z[j]`, and predicts a binary similarity probability.

The temporal skip-gram labels are:

- **Positive:** two distinct frames from the same episode, at most five saved
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
  --updates 50000 --batch-size 4 --unroll-length 2048 --device cuda:0
```

The classifier is frozen throughout anchor preparation, policy training, and
evaluation. Its causal embeddings are computed from frame zero for each source
trajectory. Starting with frame zero as the reference, extraction selects the
first later frame with similarity below the saved threshold (0.5), makes that
frame the new reference, and repeats. Frame zero is excluded from the goal list.
The terminal frame is included only if the classifier selects it.

Each anchor retains its RGB image and its source-history classifier embedding.
The latter must not be recomputed by resetting the GRU at the anchor image.
Preparation checks that the source trajectory can accomplish its anchors in
strict order. Empty or inconsistent tasks raise an error rather than being
silently discarded; improve the classifier before training BC in that case.

The supervised policy has its own CNN and two GRUs:

- An **anchor GRU** processes the ordered anchor images and retains one memory
  vector per anchor. Packed sequences handle different goal counts without
  treating padding as real images.
- A **controller GRU** processes current visual features, the next-anchor vector,
  and progress fractions. At every step, its state attends to the remaining
  anchor memories. Completed anchors and padding are masked.

The action head combines the controller state, attention result, next anchor,
and progress fractions. The frozen classifier's latent embeddings are used to
recognize progress, not as direct inputs to the policy. Neither source seeds,
trajectory IDs, nor recorded actions are policy inputs.

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

`--updates` is the total target, including resumed updates. Keep the same model,
batch, unroll, and optimizer settings. New runs draw optimizer randomness from
the operating system unless `--seed` is supplied. This seed does not replay the
source environment starts; offline fitting does not reset an environment.

## Evaluate only when training fit is satisfactory

Evaluation is a separate command and never runs automatically after training.
Start with a few held-out goals:

```bash
python train_supervised_agent.py evaluate \
  --archive ../artifacts/hero/r2dreamer_sampled256/episodes.pkl \
  --classifier ../artifacts/minimal_pipeline/classifier.pt \
  --checkpoint ../artifacts/minimal_pipeline/agent/best.pt \
  --output ../artifacts/minimal_pipeline/evaluation \
  --split test --episodes 5 --device cuda:0 --save-videos
```

Use `--split train` for familiar goals, or `--split both` to report both groups.
`--episodes N` takes the latest N declared goals per split; `--episodes 0` uses
all goals. `--repeats` controls independent attempts per goal.

Each attempt starts from a fresh OS-generated reset seed, excluding collection
seeds. The startup NOOP count and initial RGB image must differ from that goal's
source start; repeated attempts also use different NOOP counts. These are new
goal/start combinations within the Atari startup distribution, not new levels.
The evaluator checks the action mapping and preserves the archive's image
preprocessing—for the existing H.E.R.O. data, pooled RGB is resized to 64×64 and
then to 84×84. ALE/Gymnasium and the Atari ROM must be available in the environment.

The policy acts using its own observations and recurrent state. The frozen
classifier also processes this new history from its reset. After each action,
only the next unachieved anchor is compared to the current contextual embedding.
A match advances progress by one. No anchor can be skipped, and frame zero earns
no completion. Every attempt runs through environment termination or its normal
time limit, including after goal completion.

`report.json` gives complete-sequence success and mean anchor completion,
separately for familiar and held-out goals. With `--save-videos`, each attempt
also has colored 84×84 `source.mp4`, `rollout.mp4`, and individual anchor PNGs.
Poor imitation fit is a reason to improve training before spending compute on
these rollouts. Temporal similarity remains a learned proxy for accomplishment;
the saved videos support visual inspection.

Training requires PyTorch and NumPy. The second module also uses Pillow;
environment evaluation uses Gymnasium and ALE, and video output uses OpenCV.
The existing `r2dreamer` environment provides these packages. Use `--help` on
either module or either agent subcommand for all arguments. These two modules
produce their own portable checkpoint format; use the classifier checkpoint
from step 1 with the agent in step 2.
