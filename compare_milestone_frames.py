"""Compare two RGB frame files using a trained temporal similarity model."""

import argparse
import json

import numpy as np
import torch
from PIL import Image

from milestone_model import frames_to_tensor, load_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_classifier(args.checkpoint, args.device)
    threshold = checkpoint.get("threshold", 0.5) if args.threshold is None else args.threshold
    if not 0 < threshold < 1:
        parser.error("--threshold must be in (0, 1)")
    frames = []
    for path in (args.first, args.second):
        with Image.open(path) as image:
            if image.size != (84, 84):
                parser.error(f"{path}: expected an 84x84 image")
            frames.append(np.asarray(image.convert("RGB")))
    batch = frames_to_tensor(np.stack(frames), args.device)
    with torch.inference_mode():
        probability = model(batch[:1], batch[1:]).sigmoid().item()
    print(json.dumps({"similarity": probability, "similar": int(probability >= threshold),
                      "threshold": threshold}))


if __name__ == "__main__":
    main()
