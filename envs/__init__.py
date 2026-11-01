from . import parallel, wrappers


def make_envs(config):
    def env_constructor(idx):
        return lambda: make_env(config, idx)

    train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
    eval_envs = (
        parallel.ParallelEnv(env_constructor, config.eval_episode_num, config.device)
        if config.eval_episode_num > 0
        else None
    )
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
