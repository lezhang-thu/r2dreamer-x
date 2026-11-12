# Self-supervised temporal milestones

This implements the proposed skip-gram-style experiment: nearby observations
are positive pairs and random observations are negative pairs. No rewards,
actions, room IDs, or manually labeled sub-goals enter classifier training.
This learns temporal similarity, which is a proxy for useful milestones; it
does not guarantee semantic goals such as keys or doors.

## Saved demonstration

The completed run replayed all 80 bundled trajectories successfully (307,997
observations), then trained with 64 episodes and held out 16. Both models ran
1,000 optimization updates with seed 0 and window 5; the best validation
checkpoint was retained. The following files are already saved locally:

| Model | Checkpoint | Balanced validation accuracy | Demo sub-goals |
| --- | --- | --- | --- |
| ResNet-18 | `artifacts/milestones/classifier.pt` | 95.61% | 42 |
| Compact CNN | `artifacts/milestones/classifier_small.pt` | 96.48% | 58 |

The demo uses held-out `episode_0050.npz` (3,009 frames, score 11,100), threshold
0.5. Each accuracy uses 1,024 fixed sampled pairs and the caveats below apply.
For ResNet-18, open [the contact sheet](artifacts/milestones/demo/contact_sheet_000.png)
or [the 84×84 video](artifacts/milestones/demo/trajectory.mp4).
Separate PNGs are in `artifacts/milestones/demo/anchors/`; the start image is
included in addition to the 42 sub-goals. The compact comparison is under
`artifacts/milestones/demo_small/`. See `run_summary.json` and each checkpoint's
`.metrics.json` in `artifacts/milestones/` for metrics and settings.

Median anchor gaps are 51 observations for ResNet-18 and 38.5 for the compact
model. Both sometimes select consecutive frames, especially around visual
transitions. No smoothing or minimum-gap constraint was applied, preserving
the requested first-different-frame rule exactly.

## Reproduce the pipeline

Use the repository's Python environment with `torch`, `numpy`, `Pillow`,
`opencv-python`, `gymnasium`, and `ale-py`. On the current machine it is:

```bash
conda activate r2dreamer
# Or use /home/ubuntu/anaconda3/envs/r2dreamer/bin/python for each command.

python prepare_milestone_data.py
python train_milestone_classifier.py \
  --data artifacts/milestones/episodes \
  --output artifacts/milestones/classifier.pt \
  --architecture resnet18 --batch-size 32 --eval-every 200 \
  --steps 1000 --window 5 --seed 0
python extract_milestones.py \
  --checkpoint artifacts/milestones/classifier.pt \
  --trajectory artifacts/milestones/episodes/episode_0050.npz \
  --output artifacts/milestones/demo
```

Replay follows `test-consistency.py`: `Atari("MontezumaRevenge", sticky=False)`
and `reset(seed=0)`, changing only the image output to `gray=False`. It checks
the expected cumulative reward. An archive contains `frames` as RGB `uint8`
with shape `(T,84,84,3)`, including the reset frame; `actions[t]` takes
`frames[t]` to `frames[t+1]`. Actions and un-clipped rewards have length `T-1`.
The JSON manifest records provenance and whether each sequence reached an
episode boundary or exhausted its source actions early. Use `--resume` to
validate and reuse existing exports. `--indices 0,8,16,24,32,40,48,56` or
`--limit 8` can produce a smaller experiment.

Training uses a shared image encoder and a symmetric pair head. The default
compact CNN retains spatial features for Atari's small player sprite and is
practical on CPU. `--architecture resnet18` selects a randomly initialized
ResNet-18 alternative; no pretrained weights or torchvision are needed.
The command above uses ResNet-18. For the faster compact baseline, use
`--architecture small --batch-size 64 --eval-every 100` and a separate
`--output artifacts/milestones/classifier_small.pt`.
The positive window is measured in saved observations, each normally four
emulator frames. The default window of 5 therefore covers about 1/3 second.
Negatives are sampled randomly from the training frames, excluding the same
frame and same-episode pairs inside the positive window. This avoids directly
contradictory labels; there is no larger SPTM temporal exclusion margin.

The checkpoint stores weights, model configuration, split, training settings,
metrics, and decision threshold. Training holds out entire episodes, rather
than randomly splitting pairs from the same episode. The bundled Go-Explore
trajectories can share long prefixes: held-out episode scores are useful
diagnostics but **not an independent generalization benchmark**. Random
negatives can also depict the same state on revisits or in another episode.

The inspiration is [Semi-parametric Topological Memory for Navigation](https://arxiv.org/abs/1803.00653).
This is the requested skip-gram-style baseline, not a reproduction of SPTM's
graph planner or its training sampler.

## Extract from another trajectory

```bash
python extract_milestones.py \
  --checkpoint artifacts/milestones/classifier.pt \
  --trajectory /path/to/episode.npz \
  --output artifacts/milestones/another_demo \
  --threshold 0.5
```

Input can also be an `.npy` array with the same `(T,84,84,3)` RGB `uint8`
format. Extraction scans **every** frame following the current anchor. The
first whose predicted similarity is strictly below the threshold becomes the
next anchor. Scanning resumes immediately after it. Equality means similar;
the last frame is not forced into the output. Raising the threshold generally
makes the scan more sensitive, though the greedy sequence itself can change.

Each new output directory contains:

- `anchors/anchor_NNNN_frame_TTTTTT.png`: lossless color 84×84 images, including
  the start image. Subsequent images are the selected sub-goals.
- `trajectory.mp4`: the entire supplied trajectory in color at exactly 84×84,
  with no overlays. MP4 is lossy; NPZ retains the original pixels. Default
  playback is 15 fps (60/4); override `--fps` for other capture rates.
- `contact_sheet_000.png` (and more pages if necessary): labeled previews.
- `anchors.json`: indices, timestamps, paths, threshold, and the probability
  from every comparison made by the greedy scan.

Output directories must be empty to avoid mixing anchors from different runs.
Generated data and checkpoints under `artifacts/milestones/` are ignored by Git
but remain on disk for inspection and reuse.

## Compare two frames directly

```bash
python compare_milestone_frames.py \
  --checkpoint artifacts/milestones/classifier.pt \
  --first first.png --second second.png
```

Both images must be 84×84. This prints the similarity score and binary label
(`1` means similar). Scores describe the sampled temporal classification task;
they are not calibrated probabilities of semantic equivalence.

For Python use, `milestone_model.load_classifier(path, device)` returns the
model and checkpoint. Convert RGB arrays using `frames_to_tensor`, then use
`model(first_batch, second_batch).sigmoid()` in evaluation/inference mode.

Run the focused checks with:

```bash
python -m unittest discover -s tests -p 'test_milestones.py' -v
```
