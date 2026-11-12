"""Extract greedy temporal milestones and a color video from one trajectory."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from milestone_model import frames_to_tensor, load_classifier


def load_frames(path):
    """Read lossless RGB uint8 frames from an NPZ (frames key) or NPY."""
    path = Path(path)
    if path.suffix == ".npy":
        frames = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            frames = data["frames"]
    else:
        raise ValueError("Trajectory must be .npz with a frames key, or .npy")
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (84, 84, 3):
        raise ValueError("Expected RGB uint8 frames with shape (T, 84, 84, 3)")
    if not len(frames):
        raise ValueError("Trajectory is empty")
    return frames


@torch.inference_mode()
def extract_anchors(model, frames, threshold=0.5, batch_size=256, device="cpu"):
    """Return start + first-different anchors and every actual comparison.

    A probability strictly below threshold means different. Embeddings and
    comparisons are batched, but the greedy scan examines every frame in order.
    The terminal frame is not appended unless the classifier selects it.
    """
    if not 0 < threshold < 1 or batch_size < 1:
        raise ValueError("threshold must be in (0, 1) and batch_size positive")
    if not len(frames):
        raise ValueError("Trajectory is empty")
    model.eval()
    embeddings = torch.cat([
        model.encode(frames_to_tensor(frames[i:i + batch_size], device)).cpu()
        for i in range(0, len(frames), batch_size)
    ])
    anchors, comparisons = [0], []
    anchor, cursor = 0, 1
    while cursor < len(frames):
        stop = min(cursor + batch_size, len(frames))
        candidates = embeddings[cursor:stop].to(device)
        reference = embeddings[anchor:anchor + 1].to(device).expand(len(candidates), -1)
        probs = model.pair_logits(reference, candidates).sigmoid().reshape(-1).cpu().numpy()
        changed = np.flatnonzero(probs < threshold)
        count = int(changed[0]) + 1 if len(changed) else len(probs)
        for offset in range(count):
            comparisons.append({"anchor_index": anchor, "frame_index": cursor + offset,
                                "similarity": float(probs[offset])})
        cursor += count
        if len(changed):
            anchor = cursor - 1
            anchors.append(anchor)
    return anchors, comparisons


def save_video(frames, path, fps):
    """OpenCV expects BGR; disable resizing by writing exactly 84 x 84."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (84, 84))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the mp4v video writer")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(frames):
            raise RuntimeError("Video verification failed: wrong number of frames")
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (84, 84):
            raise RuntimeError("Video verification failed: wrong resolution")
    finally:
        cap.release()


def save_contact_sheets(frames, anchors, output):
    """Paginate so a pathological classifier cannot create one enormous image."""
    files = []
    for page, start in enumerate(range(0, len(anchors), 80)):
        indices = anchors[start:start + 80]
        columns = min(8, len(indices))
        sheet = Image.new("RGB", (columns * 168, ((len(indices) + columns - 1) // columns) * 190), "#222222")
        draw = ImageDraw.Draw(sheet)
        for k, index in enumerate(indices):
            x, y = (k % columns) * 168, (k // columns) * 190
            sheet.paste(Image.fromarray(frames[index]).resize((168, 168), Image.Resampling.NEAREST), (x, y))
            draw.text((x + 4, y + 172), f"{'start' if index == 0 else 'goal'} {start + k}: frame {index}", fill="white")
        name = f"contact_sheet_{page:03d}.png"
        sheet.save(output / name)
        files.append(name)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/milestones/demo"))
    parser.add_argument("--threshold", type=float, help="Override saved threshold; larger means more anchors")
    parser.add_argument("--fps", type=float, default=15, help="60 Atari frames/s divided by action_repeat=4")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    torch.set_num_threads(args.threads)
    frames = load_frames(args.trajectory)
    model, checkpoint = load_classifier(args.checkpoint, args.device)
    threshold = checkpoint.get("threshold", 0.5) if args.threshold is None else args.threshold
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory to avoid stale anchors")
    anchors, comparisons = extract_anchors(model, frames, threshold, args.batch_size, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "anchors"
    image_dir.mkdir()
    entries = []
    for number, index in enumerate(anchors):
        filename = f"anchors/anchor_{number:04d}_frame_{index:06d}.png"
        Image.fromarray(frames[index]).save(args.output / filename)
        entries.append({"frame_index": index, "time_seconds": index / args.fps,
                        "kind": "start" if index == 0 else "subgoal", "file": filename})
    save_video(frames, args.output / "trajectory.mp4", args.fps)
    sheets = save_contact_sheets(frames, anchors, args.output)
    report = {"trajectory": str(args.trajectory.resolve()), "checkpoint": str(args.checkpoint.resolve()),
              "threshold": threshold, "fps": args.fps, "frame_count": len(frames),
              "anchor_count_including_start": len(anchors), "subgoal_count": len(anchors) - 1,
              "anchors": entries, "contact_sheets": sheets, "comparisons": comparisons}
    (args.output / "anchors.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("frame_count", "subgoal_count", "threshold")}))
    print(f"Saved RGB anchors, contact sheets, comparison scores, and 84x84 MP4 to {args.output}")


if __name__ == "__main__":
    main()
