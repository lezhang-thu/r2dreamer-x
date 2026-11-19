"""Record the native RGB screens alongside the ordinary Dreamer training stream."""

from pathlib import Path
import pickle
import struct

import gymnasium as gym
import numpy as np


class NativeEpisodeArchive(gym.Wrapper):
    """Write a v2 archive in the Atari worker without changing its observations.

    Layout: pickle header, contiguous uint8 HWC frames, pickle episode index,
    then the little-endian uint64 byte offset of the index.
    """

    def __init__(self, env, path, env_config):
        super().__init__(env)
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"Trajectory archive already exists: {path}")
        self.metadata = dict(env_config=dict(env_config, archive_frame_source="ale_rgb"),
                             num_actions=int(env.action_space.shape[0]),
                             action_meanings=list(env.unwrapped._action_meanings))
        self.partial = self.path.with_suffix(self.path.suffix + ".partial")
        self.stream = self.partial.open("xb")
        pickle.dump(dict(kind="atari_episode_archive", format_version=2), self.stream, protocol=5)
        self.records = []
        self.current = None
        self.interactions = 0
        self.frame_count = 0
        self.frame_shape = None

    def _record_screen(self):
        # The live screen includes the final emulator step even when game-over
        # interrupts action repeat before the normal pooling window begins.
        frame = self.env.unwrapped.ale.getScreenRGB()
        if self.frame_shape is None:
            self.frame_shape = frame.shape
        if (frame.shape != self.frame_shape or frame.ndim != 3
                or frame.shape[-1] != 3 or frame.dtype != np.uint8):
            raise ValueError("Expected a fixed-size native uint8 RGB screen")
        self.stream.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
        self.frame_count += 1

    def reset(self, **kwargs):
        if self.current is not None:
            raise RuntimeError("An archive trajectory was reset before its final step")
        obs = self.env.reset(**kwargs)
        self.current = dict(id=len(self.records), source="ordinary", start_step=self.interactions,
                            actions=[], rewards=[], terminated=[], truncated=[])
        if not self.records:
            # Atari seeds its first reset from the constructor, then continues
            # the same RNG stream across later resets.
            self.current["reset_seed"] = int(self.metadata["env_config"]["seed"])
        self._record_screen()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._record_screen()
        self.interactions += 1
        record = self.current
        record["actions"].append(int(np.argmax(action)))
        record["rewards"].append(float(reward))
        terminal = bool(done and obs["is_terminal"])
        record["terminated"].append(terminal)
        record["truncated"].append(bool(done and not terminal))
        if done:
            self._end_record(complete=True)
        return obs, reward, done, info

    def _end_record(self, complete):
        record = self.current
        record.update(complete=complete, collection_cutoff=not complete,
                      frame_count=len(record["actions"]) + 1, end_step=self.interactions)
        for name, dtype in (("actions", np.int64), ("rewards", np.float32),
                            ("terminated", np.bool_), ("truncated", np.bool_)):
            record[name] = np.asarray(record[name], dtype=dtype)
        self.records.append(record)
        self.current = None

    def finish_archive(self):
        """Publish the indexed archive, including the prefix active at cutoff."""
        if self.current is not None:
            self._end_record(complete=False)
        if not self.interactions:
            raise ValueError("Cannot save a trajectory archive with zero interactions")
        index_offset = self.stream.tell()
        metadata = dict(self.metadata, interactions=self.interactions,
                        frame_count=self.frame_count, frame_shape=self.frame_shape)
        pickle.dump(dict(episodes=self.records, metadata=metadata), self.stream, protocol=5)
        self.stream.write(struct.pack("<Q", index_offset))
        self.stream.close()
        self.partial.replace(self.path)
        return dict(episodes=len(self.records), transitions=self.interactions, frames=self.frame_count,
                    bytes=self.path.stat().st_size)
