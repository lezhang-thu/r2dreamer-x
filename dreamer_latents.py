"""Replay native Atari trajectories through a fixed R2-Dreamer posterior."""

from itertools import islice

import numpy as np
from omegaconf import OmegaConf
import torch

from envs import make_env
from networks import MultiEncoder
from rssm import RSSM


def _load_world_model(model_config, env, checkpoint, device):
    shapes = {key: space.shape for key, space in env.observation_space.spaces.items()}
    encoder = MultiEncoder(model_config.encoder, shapes)
    model_config.rssm.device = str(device)
    rssm = RSSM(model_config.rssm, encoder.out_dim, env.action_space.shape[0])
    # latest.pt also carries optimizers, actor, critic and frozen aliases.
    # Map the file on CPU so only the two needed modules are copied into RAM.
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)["agent_state_dict"]
    for prefix, module in (("encoder.", encoder), ("rssm.", rssm)):
        module.load_state_dict({name[len(prefix):]: value for name, value in state.items()
                                if name.startswith(prefix)}, strict=True)
        module.eval().to(device)
    return encoder, rssm


def _check_env_config(config, metadata):
    archived = metadata.get("env_config", {})
    if archived.get("archive_frame_source") != "ale_rgb":
        raise ValueError("Replay requires the native episodes.pkl from the Atari training worker")
    # Some settings (e.g. pooling, resize) affect the encoder image without
    # changing the native screen. Pixel comparisons alone cannot detect them.
    fields = ("task", "action_repeat", "size", "gray", "noops", "lives", "sticky",
              "actions", "time_limit", "pooling", "aggregate", "resize", "autostart",
              "clip_reward", "seed")
    for field in fields:
        if config[field] != archived[field]:
            raise ValueError(f"Run config and archive disagree on env.{field}")


def _verified_observations(env, episode, actions):
    """Emit the live, pooled Dreamer input at each verified native frame."""
    obs = env.reset()
    frames = episode["frames"]
    for t, frame in enumerate(frames):
        location = f"episode {episode['id']} frame {t}"
        if t:
            obs, reward, done, _ = env.step(actions[t])
            terminal = bool(done and obs["is_terminal"])
            if (reward != episode["rewards"][t - 1]
                    or terminal != episode["terminated"][t - 1]
                    or bool(done and not terminal) != episode["truncated"][t - 1]
                    or bool(done) != (t == len(frames) - 1 and episode["complete"])):
                raise ValueError(f"ALE replay transition mismatch at {location}")
        if not np.array_equal(env.unwrapped.ale.getScreenRGB(), frame):
            raise ValueError(f"ALE replay screen mismatch at {location}. Check the run seed, ROM and ALE version.")
        yield obs["image"]


@torch.inference_mode()
def _infer_chunk(encoder, rssm, state, images, actions, start, device):
    # Same uint8 HWC -> [0, 1] preprocessing as Dreamer.preprocess.
    obs = torch.from_numpy(np.stack(images)).to(device=device, dtype=torch.float32).div_(255)
    embeds = encoder({"image": obs})
    actions = torch.from_numpy(actions).to(device)
    resets = torch.arange(start, start + len(images), device=device) == 0
    stoch, deter = state
    rows = []
    for i in range(len(images)):
        stoch, deter, _ = rssm.obs_step(stoch, deter, actions[i:i + 1], embeds[i:i + 1],
                                        resets[i:i + 1], sample=False)
        rows.append(rssm.get_feat(stoch, deter))
    return (stoch, deter), torch.cat(rows).cpu()


def replay_latents(episodes, metadata, checkpoint, config_path, device="cpu", batch_size=64):
    """Yield (episode ID, frame offset, CPU float32 rows) for every archived frame.

    The encoder sees the environment's original pooled/resized observation.
    Previous recorded actions and posterior modes propagate through the RSSM
    across inference batches. Only an episode reset starts a fresh history.
    """
    if batch_size < 1:
        raise ValueError("Dreamer replay batch size must be positive")
    config = OmegaConf.load(config_path)
    _check_env_config(config.env, metadata)
    env = make_env(config.env, 0)
    try:
        action_count = env.action_space.shape[0]
        if (action_count != metadata["num_actions"]
                or list(env.unwrapped._action_meanings) != metadata["action_meanings"]):
            raise ValueError("ALE replay action space disagrees with the archive")
        encoder, rssm = _load_world_model(config.model, env, checkpoint, device)
        one_hot = np.eye(action_count, dtype=np.float32)
        step = 0
        for ep in episodes:
            recorded = ep["actions"]
            if ep["start_step"] != step or ep["end_step"] != step + len(recorded):
                raise ValueError("Replay needs all archived episodes in chronological order")
            if np.any(recorded < 0) or np.any(recorded >= action_count):
                raise ValueError(f"Invalid action in episode {ep['id']}")
            # actions[t] takes the environment to frame t. The reset input is zero.
            actions = np.zeros((len(recorded) + 1, action_count), dtype=np.float32)
            actions[1:] = one_hot[recorded]
            observations = _verified_observations(env, ep, actions)
            state = rssm.initial(1)
            for start in range(0, len(actions), batch_size):
                images = list(islice(observations, batch_size))
                state, rows = _infer_chunk(encoder, rssm, state, images,
                                          actions[start:start + len(images)], start, device)
                yield ep["id"], start, rows
            step += len(recorded)
    finally:
        env.close()
