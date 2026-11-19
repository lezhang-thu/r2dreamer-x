from . import parallel, wrappers


def make_envs(config, archive_path=None):
    """Construct training workers only; periodic evaluation is disabled."""
    if archive_path is not None and config.env_num != 1:
        raise ValueError("Saving the native trajectory archive requires env.env_num=1")

    def env_constructor(idx):
        def construct():
            env = make_env(config, idx)
            if archive_path is not None:
                from omegaconf import OmegaConf
                from .native_archive import NativeEpisodeArchive
                return NativeEpisodeArchive(env, archive_path, OmegaConf.to_container(config, resolve=True))
            return env
        return construct

    train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
    # Preserve the caller's return signature without allocating idle evaluation
    # workers, including for older configs with a positive eval_episode_num.
    eval_envs = None
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def make_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite != "atari":
        raise ValueError(f"Expected an atari_<game> task, got {config.task!r}")
    from .atari import Atari

    env = Atari(
        task,
        config.action_repeat,
        config.size,
        gray=config.gray,
        noops=config.noops,
        lives=config.lives,
        sticky=config.sticky,
        actions=config.actions,
        length=config.time_limit,
        pooling=config.pooling,
        aggregate=config.aggregate,
        resize=config.resize,
        autostart=config.autostart,
        clip_reward=config.clip_reward,
        seed=config.seed + id,
    )
    env = wrappers.OneHotAction(env)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)
