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

Run the Atari experiment launcher from the repository root with the Conda environment active:

```bash
bash runs/atari.sh
```

[runs/atari.sh](runs/atari.sh) runs the 100M R2-Dreamer model sequentially across 26 Atari games and seeds `0, 100, 200, 300, 400`. It enables model compilation and stores replay data on CPU.

Edit `GPU_ID`, `SEED_START`, `SEED_END`, `SEED_STEP`, `METHOD`, and the `tasks` array in the script to configure the sweep. `GPU_ID` selects the physical GPU; the training process uses it as `cuda:0`. Each run writes to `logdir/<date>_<method>_<game>_<seed>/`.

To change the game or seed, invoke the training entry point directly:

```bash
python train.py model=size100M env=atari100k env.task=atari_alien seed=0
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
