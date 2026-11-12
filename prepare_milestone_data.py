"""Replay trusted Go-Explore action pickles into lossless RGB episode archives.

Frames include the reset observation: actions[t] takes frames[t] to frames[t+1].
Like test-consistency.py, replay uses seed 0 and disables sticky actions.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


REPLAY_CONFIG = {
    "game": "MontezumaRevenge",
    "seed": 0,
    "sticky": False,
    "gray": False,
    "size": [84, 84],
    "action_repeat": 4,
    "fps": 15.0,
    "pooling": 2,
    "aggregate": "max",
    "resize": "pillow",
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_existing(path, identity, expected_actions, expected_score):
    """Validate an archive before treating --resume as a cache hit."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        for key, value in identity.items():
            if metadata.get(key) != value:
                raise ValueError(f"{path}: metadata mismatch for {key}; use a fresh output directory")
        frames = data["frames"]
        actions, rewards = data["actions"], data["rewards"]
        if frames.dtype != np.uint8 or frames.shape != (len(actions) + 1, 84, 84, 3):
            raise ValueError(f"{path}: invalid RGB frames")
        if rewards.shape != actions.shape or not np.array_equal(actions, expected_actions):
            raise ValueError(f"{path}: actions/rewards do not match the source trajectory")
        if float(rewards.sum(dtype=np.float64)) != expected_score:
            raise ValueError(f"{path}: reward total does not match the source")
        if metadata["frame_count"] != len(frames) or metadata["score"] != expected_score:
            raise ValueError(f"{path}: inconsistent archive metadata")
    return metadata


def replay_episode(actions, expected_score):
    """Return all RGB observations, un-clipped rewards, and terminal flags."""
    from envs.atari import Atari

    env = Atari("MontezumaRevenge", sticky=False, gray=False, size=(84, 84))
    try:
        observation = env.reset(seed=0)
        frames = [observation["image"].copy()]
        rewards = []
        complete = False
        for index, action in enumerate(actions):
            observation, reward, done, _ = env.step(int(action))
            frames.append(observation["image"].copy())
            rewards.append(reward)
            if done:
                complete = True
                if index + 1 != len(actions):
                    raise ValueError(
                        f"Replay ended after {index + 1}/{len(actions)} actions; "
                        "the action sequence does not match this environment"
                    )
                break
        score = float(sum(rewards))
        if score != expected_score:
            raise ValueError(f"Trajectory consistency error: score {score} != expected {expected_score}")
        return (
            np.stack(frames), np.asarray(rewards, dtype=np.float32),
            complete, bool(observation["is_terminal"]),
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, default=Path(__file__).with_name("atari_MontezumaRevenge.pkl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/milestones/episodes"))
    parser.add_argument("--limit", type=int, help="Replay at most this many selected episodes")
    parser.add_argument("--indices", help="Comma-separated source episode indices; default: all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Reuse existing archives after validating source and contents")
    mode.add_argument("--overwrite", action="store_true", help="Recreate selected archives from the same source")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    # Pickle is used solely for the repository's trusted action-sequence format.
    source_bytes = args.actions.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    trajectories = pickle.loads(source_bytes)
    try:
        indices = list(range(len(trajectories))) if args.indices is None else [int(x) for x in args.indices.split(",")]
    except ValueError:
        parser.error("--indices must contain comma-separated integers")
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(trajectories) for i in indices):
        parser.error(f"--indices must contain unique indices in [0, {len(trajectories) - 1}]")
    if args.limit is not None:
        indices = indices[:args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = {
        "format_version": 1,
        "source": str(args.actions.resolve()),
        "source_sha256": source_hash,
        "replay": REPLAY_CONFIG,
        "episodes": [],
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("source_sha256") != source_hash or previous.get("replay") != REPLAY_CONFIG:
            raise ValueError("Output manifest belongs to different source/settings; choose a fresh output directory")
        manifest["episodes"] = previous["episodes"]
    entries = {entry["source_index"]: entry for entry in manifest["episodes"]}
    if not args.resume and not args.overwrite:
        existing = [args.output / f"episode_{i:04d}.npz" for i in indices if (args.output / f"episode_{i:04d}.npz").exists()]
        if existing:
            parser.error(f"{existing[0]} already exists; choose --resume, --overwrite, or a fresh output directory")

    for index in indices:
        source_actions, expected_score, timestamp = trajectories[index]
        actions = np.asarray(source_actions)
        if actions.ndim != 1 or not np.issubdtype(actions.dtype, np.integer) or np.any(actions < 0) or np.any(actions >= 18):
            raise ValueError(f"Episode {index}: expected a vector of Atari action integers in [0, 17]")
        actions = actions.astype(np.uint8)
        expected_score = float(expected_score)
        path = args.output / f"episode_{index:04d}.npz"
        identity = {
            "source_sha256": source_hash,
            "source_index": index,
            "timestamp": str(timestamp),
            "replay": REPLAY_CONFIG,
        }
        if path.exists() and args.resume:
            metadata = load_existing(path, identity, actions, expected_score)
            status = "reused"
        else:
            frames, rewards, complete, terminal = replay_episode(actions, expected_score)
            metadata = {
                **identity,
                "frame_count": len(frames),
                "action_count": len(actions),
                "score": float(rewards.sum(dtype=np.float64)),
                "expected_score": expected_score,
                "complete": complete,
                "terminal": terminal,
                "fps": REPLAY_CONFIG["fps"],
            }
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream, frames=frames, actions=actions, rewards=rewards,
                    metadata=np.asarray(json.dumps(metadata)),
                    complete=np.asarray(complete), terminal=np.asarray(terminal),
                    score=np.asarray(expected_score), source_index=np.asarray(index),
                    fps=np.asarray(REPLAY_CONFIG["fps"]),
                )
            temporary.replace(path)
            status = "saved"
        entries[index] = {"file": path.name, **metadata}
        manifest["episodes"] = [entries[i] for i in sorted(entries)]
        write_json(manifest_path, manifest)
        ending = "complete" if metadata["complete"] else "partial (source actions exhausted)"
        print(f"{status} {path}: {metadata['frame_count']} frames, score={expected_score:g}, {ending}", flush=True)


if __name__ == "__main__":
    main()
