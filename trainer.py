import torch

import tools


class OnlineTrainer:

    def __init__(self, config, replay_buffer, logger, train_envs,
                 eval_envs):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio *
                                           config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        Atari environments step on CPU; observations are transferred to the
        agent device for policy inference.
        """
        print("Evaluating the policy...")
        envs = self.eval_envs
        agent.eval()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num,
                                dtype=torch.bool,
                                device=agent.device)
        steps = torch.zeros(envs.env_num,
                            dtype=torch.int32,
                            device=agent.device)
        returns = torch.zeros(envs.env_num,
                              dtype=torch.float32,
                              device=agent.device)
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            # Step Atari on CPU and transfer observations to the agent device.
            # (B, A), (B,)
            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            once_done |= done
        self.logger.scalar("episode/eval_score", returns.mean())
        self.logger.scalar("episode/eval_length",
                           steps.to(torch.float32).mean())
        self.logger.write(train_step)
        agent.train()

    def begin(self, agent):
        """Main online training loop.

        Transfer Atari observations from CPU workers to the agent device
        using pinned memory for asynchronous GPU transfers.
        """
        envs = self.train_envs
        step = self.replay_buffer.count() * self._action_repeat
        update_count = 0
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num,
                              dtype=torch.float32,
                              device=agent.device)
        lengths = torch.zeros(envs.env_num,
                              dtype=torch.int32,
                              device=agent.device)
        episode_ids = torch.arange(
            envs.env_num, dtype=torch.int32, device=agent.device
        )  # Kept constant so short episodes (< batch_length) remain sampable; RSSM resets via is_first.
        train_metrics = tools.MetricAccumulator()
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while step < self.steps:
            # Evaluation
            if self._should_eval(
                    step
            ) and self.eval_episode_num > 0 and self.eval_envs is not None:
                self.eval(agent, step)
            # Save metrics
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        self.logger.write(step + i)
                        returns[i] = lengths[i] = 0
            step += int(
                (~done
                 ).sum()) * self._action_repeat  # step is based on env side
            lengths += ~done

            # Step Atari on CPU and transfer observations to the agent device.
            # (B, A), (B,)
            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            act, agent_state = agent.act(trans.clone(),
                                         agent_state,
                                         eval=False)

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids  # Don't lift dim
            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]
            # Update models after enough data has accumulated
            if step // (envs.env_num *
                        self._action_repeat) > self.batch_length + 1:
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)
                for _ in range(update_num):
                    train_metrics.add(agent.update(self.replay_buffer))
                update_count += update_num
                # Log training metrics
                if self._should_log(step):
                    for name, value in train_metrics.mean().items():
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", update_count)
                    self.logger.write(step, fps=True)
