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

Every 100,000 **ordinary** interactions, learning pauses for one return phase:

1. Snapshot completed episodes since the preceding snapshot. After the first
   phase this includes its 100 assisted episodes and newly completed ordinary
   episodes. An unfinished ordinary episode stays alive and is included only
   in a later snapshot after it ends. Evaluation episodes are never included.
2. Fit a fresh classifier on all snapshot episodes: ResNet-18, 5,000 visual
   updates, 5,000 GRU updates, temporal window 20. Unlike standalone evaluation,
   this mode has no held-out episodes; checkpoint-selection diagnostics use
   training episodes.
3. Train the return policy for 10,000 additional updates, batch size 4, unroll
   length 2,048. Later phases retain the preceding `latest.pt` policy parameters
   **and Adam state**. The first phase initializes a new policy unless
   `return_exploration.initial_policy=/path/to/agent/latest.pt` is supplied.
   The best training-fit checkpoint in the current phase controls rollouts.
4. Build representative anchors in source order. Every similar representative
   gets its counter incremented; a new representative is added only when none
   match. Precompute all matching source prefixes against the completed set.
5. Generate **exactly 100 complete assisted episodes**, independently of the
   snapshot size. Choose a representative with probability proportional to
   `1 / count`, then uniformly choose a matching source prefix through its
   anchor. The return policy gets that sequence of anchor images. At every
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
are logged separately. Startup NOOPs and evaluation interactions are outside
these collection counters, as in the original trainer. The replay capacity is
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

Artifacts are saved beneath the run's log directory:

- `episode_archive/`: compressed completed episodes and the pending-pool manifest.
- `return_exploration/round_000001/episodes.pkl`: the exact first snapshot S.
- Each round also contains classifier and policy checkpoints, their fitting
  logs, the initial anchor bank and prefix lookup, final counters, rollout
  results, a summary, and a Dreamer weight/optimizer checkpoint.

Historical source archives stay on disk. Only pending episode IDs and the
unfinished ordinary episode persist in memory between phases. The standalone
trainers load each current snapshot into memory during fitting.
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
