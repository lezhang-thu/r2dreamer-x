# Atari quick start

Run the commands below from the repository root. Training defaults to an NVIDIA GPU; install a driver compatible with your chosen PyTorch CUDA build.

## Installation (Conda, Python 3.12)

```bash
conda create -n r2dreamer python=3.12 pip -y
conda activate r2dreamer
python -m pip install --upgrade pip setuptools wheel
```

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to select **Stable**, your OS, **Pip**, **Python**, and a CUDA build supported by your GPU and driver. Run its installation command inside this Conda environment. Only `torch` is needed here; `torchvision` and `torchaudio` are not used.

Then install the remaining requirements:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

PyTorch, Gymnasium, and ALE are unpinned. In a fresh environment, pip selects compatible current stable releases; the command above preserves an already installed compatible CUDA build of PyTorch. TorchRL and TensorDict are included for replay storage and tensor batches. Other dependencies cover configuration, multiprocessing, image preprocessing, and plain-text metric logging.

Current [ALE installations include the Atari ROMs](https://ale.farama.org/main/getting-started/).

Check CUDA and the Atari environment:

```bash
python - <<'PYTHON'
import torch
from envs.atari import Atari

print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "Check your NVIDIA driver and PyTorch CUDA build"
print("GPU:", torch.cuda.get_device_name(0))
env = Atari("pong", sticky=False)
obs = env.reset(seed=0)
print("Atari image:", obs["image"].shape)
env.close()
PYTHON
```

## Collect 100k MontezumaRevenge interactions for the GRU

From the repository root, run the collector once on the training machine.

```bash
RUN_DIR=logdir/montezuma_100k_0 bash runs/collect_montezuma_100k.sh
```

This trains the 100M R2-Dreamer agent for 100,000 action decisions in one environment.
At action repeat 4 this is a budget of 400,000 repeated frames. The archive records
the native `210 × 160 × 3` uint8 ALE screen after reset and after each decision.
Dreamer receives its usual pooled 64×64 input. The final action and observation
are included even when the budget ends midway through a game.

The reusable, uncompressed file is `logdir/montezuma_100k_0/episodes.pkl`. For
`E` recorded trajectories, the pixels occupy `100800 × (100000 + E)` bytes,
plus about 1.4 MB for actions, rewards and end flags and a small episode index.
Allow roughly 10.1 GB for the archive and additional space for the Dreamer
checkpoint and logs. The writer builds the final archive in place under a
temporary name and publishes it when collection finishes.

Train the GRU on the same file with different options and a fresh output path for each experiment.

```bash
python variants-supervised-agent/train_gru_classifier.py \
  --archive logdir/montezuma_100k_0/episodes.pkl \
  --output logdir/montezuma_100k_0/gru-run1/classifier.pt \
  --seed 0 --device cuda:0
```

The classifier needs access to the [DINOv3 ViT-L/16 pretrained weights](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
Accept the model terms and authenticate on the training machine, or supply
`--backbone /path/to/local/huggingface-snapshot`. See the
[classifier guide](variants-supervised-agent/README.md) for splits and preprocessing.

## Run training

The default configuration now runs MontezumaRevenge with periodic return
exploration, **50 million ordinary environment interactions** (200 million
action-repeat frames), and a replay capacity of **1 million stored transitions**.
Launch the 100M model from the repository root:

```bash
conda activate r2dreamer
bash runs/montezuma_return.sh
```

`GPU_ID`, `SEED`, and `RUN_DIR` can be supplied as environment variables. Extra
Hydra overrides can be passed after the script, for example
`bash runs/montezuma_return.sh model.compile=False`.

Periodic Dreamer evaluation is disabled. Training creates only its training
environment workers; it does not construct evaluation environments, even when
an older configuration specifies a positive `env.eval_episode_num`.

Every 100,000 **ordinary** interactions, learning pauses for one return phase:

1. Select completed episodes since the preceding return phase. After the first
   phase this includes its 100 assisted episodes and newly completed ordinary
   episodes. An unfinished ordinary episode stays alive and is included only
   in a later source set after it ends.
2. Extract frozen DINOv3 ViT-L/16 CLS embeddings from the cycle's RGB84 frames.
   Fit the GRU classifier for 5,000 updates with temporal window 20.
   Checkpoint-selection diagnostics use all source episodes.
3. Train the return policy for 10,000 additional updates, batch size 4, unroll
   length 2,048. Later phases continue with the same policy parameters
   **and AdamW optimizer state in memory**. The first phase initializes a new policy unless
   `return_exploration.initial_policy=/path/to/agent/latest.pt` is supplied.
   A separate copy of the best training-fit weights controls rollouts; the
   latest weights and optimizer state remain the next cycle's starting point.
4. Build representative anchors in source order. Every similar representative
   gets its counter incremented; a new representative is added only when none
   match. Precompute all matching source prefixes against the completed set.
5. Generate **exactly 100 complete assisted episodes**, independently of the
   source-set size. Choose a representative with probability proportional to
   `1 / count`, then uniformly sample one of its matching source prefixes,
   including the target anchor. Each matching occurrence has equal probability,
   regardless of prefix length. The return policy gets that sequence of anchor
   images. At every
   observation, classifier similarity to the sampled representative at least
   0.5 triggers Dreamer takeover without resetting the environment. Dreamer
   observes the actual live history and executed actions throughout the return
   attempt. If the target is never reached, the return policy runs to episode end.
6. Insert each complete assisted episode into replay, then increment every
   matching representative counter using its extracted anchors before drawing
   the next target. Representatives and source prefixes stay frozen throughout
   the phase. Resume ordinary collection and Dreamer updates afterward.

Assisted interactions, including Dreamer takeovers, **do not advance** the 100k
clock or the 50m ordinary-interaction budget. Their count and the combined count
are logged separately. Startup NOOPs are outside these collection counters,
as in the original trainer. The replay capacity is
measured in stored observation rows, including episode-reset rows.

Every assisted episode prints a summary immediately on completion, for example
`[return round 1] episode 7/100 return=1200.00 steps=850 reached=True takeover=True`.
The return sums environment rewards across the entire episode, including both
controllers. It is also saved as `env_return` in each `return_episode` JSON log
entry and in the round's `rollouts.json`.

The integration currently requires one training environment, RGB observations,
and complete-game episodes (`env.env_num=1`, `env.gray=False`,
`env.lives=unused`). It shares the standard Atari preprocessing and action space;
no game-specific state, room labels, reset-to-state mechanism, or reward-based
goal detector is used. Episodes without classifier-selected anchors remain in
classifier training but cannot supply return-policy targets and are reported
and skipped for policy fitting. If no anchors exist at all, the phase fails
explicitly instead of inventing targets or reinitializing the policy. As in the
standalone variant, policy fitting uses full anchor sequences; prefix inputs at
rollout rely on the learned policy's generalization.

Trajectory data stays in memory. Dreamer's replay uses CPU RAM by default and
is limited to 1 million stored transitions; it is not saved to disk. The
return-exploration pool keeps compressed completed episodes in RAM until their
next phase. The classifier and return policy train sequentially inside the
main Dreamer process. Each source episode is decoded once, replacing its
compressed record, and the same episode arrays are passed directly to the
classifier, policy preparation, and anchor-prefix preparation. There are no
auxiliary training subprocesses, memory-file snapshots, or trajectory archives.
Once fitting and goal-prefix preparation finish, the source episodes are
released. Only the goal images, anchor bank, and prefix lookup remain until
the 100 assisted episodes finish. Those newly generated episodes, together
with newly completed ordinary episodes, form the next pool. This pool and the
trainers' working memory are additional to replay RAM and depend on the size
of the current source set.

The latest return policy and its optimizer state remain in CPU memory between
cycles and move to the configured device for fitting. Training restores
Dreamer's random-generator states and CPU thread count afterward. The normal
learning rate remains `3e-4` for both initial fitting and later cycles.

Model checkpoints and small logs are saved beneath the run's log directory:

- `return_exploration/round_000001/` (with the current round number) contains
  classifier and policy checkpoints, fitting logs, rollout results, and a
  summary. Only the newest completed round and an in-progress round are kept.
  The previous round is removed only after the next round succeeds; an external
  `initial_policy` checkpoint is never removed. Anchor banks, prefix lookups,
  and counter arrays are not dumped to disk.
- `latest.pt` contains Dreamer weights and optimizer state, without replay data.
  It is replaced atomically after each return phase and at training completion.
- `return_exploration/cycles.jsonl` preserves small summaries of all rounds.
  Console logs and scalar metrics also retain every assisted episode's return.

These retention rules apply to new runs. Archives and older round directories
created by earlier versions are not automatically deleted.
Use a fresh log directory for each run: full online resume of replay and live
emulator state is not implemented. `initial_policy` warm-starts the return
policy in a new run; it does not resume Dreamer or its interaction counters.

The original Atari-100k baseline sweep remains available, with return
exploration explicitly disabled:

```bash
bash runs/atari.sh
```

[runs/atari.sh](runs/atari.sh) runs the 100M R2-Dreamer model sequentially across 26 Atari games and seeds `0, 100, 200, 300, 400`. It enables model compilation and stores replay data on CPU.

Edit `GPU_ID`, `SEED_START`, `SEED_END`, `SEED_STEP`, `METHOD`, and the `tasks` array in the script to configure the sweep. `GPU_ID` selects the physical GPU; the training process uses it as `cuda:0`. Each run writes to `logdir/<date>_<method>_<game>_<seed>/`.

To change the game or seed, invoke the training entry point directly:

```bash
python train.py model=size100M env=atari100k env.task=atari_alien seed=0 return_exploration.enabled=False
```

Training writes `console.log`, scalar metrics in `metrics.jsonl`, the resolved `config.yaml`, and the final `latest.pt` checkpoint under `logdir/` by default. To follow a run's metrics:

```bash
tail -f logdir/<date>/<time>/metrics.jsonl
```

## Trajectory consistency check

```bash
python test-consistency.py
```

This replays the bundled `atari_MontezumaRevenge.pkl` dataset through `envs/atari.py` and checks cumulative rewards.
