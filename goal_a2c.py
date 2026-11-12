"""On-policy recurrent A2C in independent environments for ordered milestones."""

from pathlib import Path
import math

import numpy as np
import torch
from torch.distributions import Categorical

from goal_policy import GoalPolicy, _anchor_batch_to_tensor
from milestone_model import frames_to_tensor


def compute_returns(rewards, dones, terminals, valid, bootstrap_values, final_values, gamma=0.99):
    """Masked n-step returns with a separate value bootstrap at each time limit.

    All arguments except ``final_values[B]`` have shape [T,B]. ``dones`` includes
    time limits and synthetic success, while ``terminals`` includes true terminal
    states and synthetic success. A time limit uses its final observation value,
    never a reset observation. Inactive trailing slots cannot affect returns.
    """
    if any(value.shape != rewards.shape for value in (dones, terminals, valid, bootstrap_values)):
        raise ValueError("Return tensors must have the same [T,B] shape")
    if rewards.ndim != 2 or final_values.shape != rewards.shape[1:]:
        raise ValueError("Expected [T,B] rewards and [B] final values")
    returns = torch.zeros_like(rewards)
    running = final_values.detach()
    for index in reversed(range(len(rewards))):
        continuation = torch.where(dones[index], bootstrap_values[index], running)
        continuation = torch.where(terminals[index], torch.zeros_like(continuation), continuation)
        candidate = rewards[index] + gamma * continuation
        running = torch.where(valid[index], candidate, running)
        returns[index] = torch.where(valid[index], running, torch.zeros_like(running))
    return returns.detach()


def _image(observation):
    frame = np.asarray(observation["image"])
    if frame.shape != (84, 84, 3) or frame.dtype != np.uint8:
        raise ValueError("Goal environments must provide uint8 RGB observations shaped [84,84,3]")
    return frame.copy()


def _transition(environment, action):
    result = environment.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated), bool(truncated)
    if len(result) != 4:
        raise ValueError("Expected four- or five-element environment step result")
    observation, reward, done, info = result
    terminal = bool(done and observation.get("is_terminal", not info.get("TimeLimit.truncated", False)))
    return observation, float(reward), terminal, bool(done and not terminal)


class GoalA2C:
    """Train a policy on its own sampled actions with a frozen visual reward model."""

    def __init__(self, policy, classifier, device="cpu", threshold=0.5, gamma=0.99,
                 learning_rate=3e-4, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5):
        numeric = (threshold, gamma, learning_rate, entropy_coef, value_coef, max_grad_norm)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("A2C settings must be finite")
        if not 0 <= threshold <= 1 or not 0 <= gamma <= 1:
            raise ValueError("threshold and gamma must lie in [0,1]")
        if learning_rate <= 0 or max_grad_norm <= 0 or entropy_coef < 0 or value_coef < 0:
            raise ValueError("Invalid optimizer or loss settings")
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.config = {
            "threshold": threshold, "gamma": gamma, "learning_rate": learning_rate,
            "entropy_coef": entropy_coef, "value_coef": value_coef, "max_grad_norm": max_grad_norm,
        }
        self.threshold, self.gamma = threshold, gamma
        self.entropy_coef, self.value_coef, self.max_grad_norm = entropy_coef, value_coef, max_grad_norm
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.classifier = classifier
        self.updates = 0
        self.env_steps = 0
        self.training_env_steps = 0
        self.evaluation_env_steps = 0

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, classifier):
        policy_parameters = {id(parameter) for parameter in self.policy.parameters()}
        if any(id(parameter) in policy_parameters for parameter in classifier.parameters()):
            raise ValueError("The policy and classifier must not share parameters")
        self._classifier = classifier.to(self.device).eval()
        self._classifier.requires_grad_(False)

    def set_classifier(self, classifier):
        self.classifier = classifier

    @torch.no_grad()
    def _classifier_memory(self, anchors, lengths, classifier):
        valid = torch.arange(anchors.shape[1], device=anchors.device)[None, :] < lengths.to(anchors.device)[:, None]
        indices = valid.flatten().nonzero(as_tuple=False).flatten()
        flattened = anchors.reshape(-1, 84, 84, 3)
        encoded = torch.cat([
            classifier.encode(_anchor_batch_to_tensor(flattened[selected], self.device))
            for selected in indices.split(128)
        ], 0)
        memory = encoded.new_zeros((*anchors.shape[:2], encoded.shape[-1]))
        memory[valid.to(self.device)] = encoded
        return memory

    def run_batch(self, anchor_sequences, env_factory, seeds, *, train=True, max_steps=4096,
                  rollout_steps=32, capture_frames=False, source_ids=None, deterministic=True):
        """Run one fresh environment per nonempty sequence, without autoresets.

        Anchor sequences exclude the starting frame. Only the next unachieved
        anchor is tested after each action, yielding at most one unit reward per
        transition. Environment rewards are logged but never train this policy.
        During evaluation ``deterministic=True`` selects the greedy action;
        training always samples actions from the current on-policy distribution.
        """
        if max_steps < 1 or rollout_steps < 1:
            raise ValueError("max_steps and rollout_steps must be positive")
        count = len(anchor_sequences)
        if len(seeds) != count or (source_ids is not None and len(source_ids) != count):
            raise ValueError("Each anchor sequence needs one seed and source id")
        source_ids = list(range(count)) if source_ids is None else list(source_ids)
        sequences = []
        for sequence in anchor_sequences:
            array = np.asarray(sequence)
            if array.size == 0:
                array = np.empty((0, 84, 84, 3), dtype=np.uint8)
            if array.ndim != 4 or array.shape[1:] != (84, 84, 3) or array.dtype != np.uint8:
                raise ValueError("Each anchor sequence must be uint8 [L,84,84,3]")
            sequences.append(array)
        records = [{
            "source_id": source_ids[index], "seed": int(seeds[index]),
            "goals_total": len(sequence), "goals_hit": 0, "completed_fraction": 0.0,
            "success": False, "skipped_empty": len(sequence) == 0,
            "hit_steps": [], "env_steps": 0, "env_return": 0.0, "goal_return": 0.0,
            "terminated": False, "truncated": False,
        } for index, sequence in enumerate(sequences)]
        captured = [[] for _ in sequences]
        environments = {}
        update_records = []
        self.policy.train(train)
        classifier = self.classifier  # One fixed classifier snapshot for this entire batch.
        classifier.eval()
        classifier.requires_grad_(False)
        if not any(len(sequence) for sequence in sequences):
            return self._result(records, update_records, captured, capture_frames, train)

        lengths = torch.tensor([len(sequence) for sequence in sequences], device=self.device)
        progress = torch.zeros(count, dtype=torch.long, device=self.device)
        # Keep long padded image sequences on CPU as bytes; only visual chunks
        # and their much smaller recurrent features need accelerator memory.
        anchors = torch.zeros((count, int(lengths.max()), 84, 84, 3), dtype=torch.uint8)
        for index, sequence in enumerate(sequences):
            if len(sequence):
                anchors[index, :len(sequence)] = torch.from_numpy(np.ascontiguousarray(sequence))
        classifier_memory = self._classifier_memory(anchors, lengths, classifier)
        observations = np.zeros((count, 84, 84, 3), dtype=np.uint8)
        active = lengths > 0
        hidden = self.policy.initial_hidden(count, self.device)

        try:
            for index in active.nonzero(as_tuple=False).flatten().tolist():
                environment = env_factory(index)
                environments[index] = environment
                if hasattr(environment, "action_space") and environment.action_space.n != self.policy.num_actions:
                    raise ValueError("Goal policy and environment action counts differ")
                reset = environment.reset(seed=int(seeds[index]))
                observation = reset[0] if isinstance(reset, tuple) else reset
                observations[index] = _image(observation)
                if capture_frames:
                    captured[index].append(observations[index].copy())

            while active.any():
                with torch.set_grad_enabled(train):
                    # Recompute after every update: stale anchor graphs cannot be
                    # reused after optimizer.step, and the anchor GRU must learn.
                    memory = self.policy.encode_anchors(anchors, lengths)
                    hidden = hidden.detach()
                    log_probabilities, values, entropies = [], [], []
                    rewards, dones, terminals, valid_masks, bootstraps = [], [], [], [], []
                    for _ in range(rollout_steps):
                        if not active.any():
                            break
                        valid = active.clone()
                        output = self.policy(frames_to_tensor(observations, self.device), memory,
                                             lengths, progress, hidden)
                        distribution = Categorical(logits=output["logits"])
                        actions = (output["logits"].argmax(-1) if not train and deterministic
                                   else distribution.sample())
                        hidden = torch.where(valid[:, None], output["hidden"], hidden)
                        reward = torch.zeros(count, device=self.device)
                        done = torch.zeros(count, dtype=torch.bool, device=self.device)
                        terminal = done.clone()
                        for index in valid.nonzero(as_tuple=False).flatten().tolist():
                            observation, env_reward, terminated, truncated = _transition(
                                environments[index], int(actions[index].item())
                            )
                            observations[index] = _image(observation)
                            record = records[index]
                            record["env_steps"] += 1
                            record["env_return"] += env_reward
                            self.env_steps += 1
                            if train:
                                self.training_env_steps += 1
                            else:
                                self.evaluation_env_steps += 1
                            record["terminated"] = terminated
                            record["truncated"] = truncated or (record["env_steps"] >= max_steps and not terminated)
                            done[index] = terminated or record["truncated"]
                            terminal[index] = terminated
                            if capture_frames:
                                captured[index].append(observations[index].copy())

                        # No classifier gradients, BatchNorm updates, or repeated
                        # goal tests in a transition, even if anchors look alike.
                        with torch.no_grad():
                            indices = valid.nonzero(as_tuple=False).flatten()
                            current = classifier.encode(frames_to_tensor(observations[indices.cpu().numpy()], self.device))
                            target = classifier_memory[indices, progress[indices]]
                            reached = torch.sigmoid(classifier.pair_logits(current, target)) >= self.threshold
                            hit_indices = indices[reached]
                            reward[hit_indices] = 1.0
                            progress[hit_indices] += 1
                            for index in hit_indices.tolist():
                                record = records[index]
                                record["goals_hit"] += 1
                                record["goal_return"] += 1.0
                                record["hit_steps"].append(record["env_steps"])
                            success = valid & (progress == lengths)
                            done |= success
                            terminal |= success  # Synthetic success is absorbing, not a time limit.
                            for index in success.nonzero(as_tuple=False).flatten().tolist():
                                records[index]["success"] = True

                        bootstrap = torch.zeros(count, device=self.device)
                        time_limits = done & ~terminal
                        if time_limits.any():
                            with torch.no_grad():
                                final_output = self.policy(frames_to_tensor(observations, self.device), memory,
                                                           lengths, progress, hidden)
                                bootstrap[time_limits] = final_output["value"][time_limits]
                        log_probabilities.append(distribution.log_prob(actions))
                        values.append(output["value"])
                        entropies.append(distribution.entropy())
                        rewards.append(reward)
                        dones.append(done)
                        terminals.append(terminal)
                        valid_masks.append(valid)
                        bootstraps.append(bootstrap)
                        active = valid & ~done

                    if train:
                        with torch.no_grad():
                            final_values = torch.zeros(count, device=self.device)
                            if active.any():
                                final_output = self.policy(frames_to_tensor(observations, self.device), memory,
                                                           lengths, progress, hidden)
                                final_values[active] = final_output["value"][active]
                        valid = torch.stack(valid_masks)
                        returns = compute_returns(torch.stack(rewards), torch.stack(dones),
                                                  torch.stack(terminals), valid, torch.stack(bootstraps),
                                                  final_values, self.gamma)
                        predicted_values = torch.stack(values)
                        advantages = returns - predicted_values
                        actor_loss = -(torch.stack(log_probabilities)[valid] * advantages.detach()[valid]).mean()
                        critic_loss = advantages[valid].square().mean()
                        entropy = torch.stack(entropies)[valid].mean()
                        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.updates += 1
                        update_records.append({
                            "update": self.updates, "transitions": int(valid.sum()),
                            "loss": float(loss.detach()), "actor_loss": float(actor_loss.detach()),
                            "critic_loss": float(critic_loss.detach()), "entropy": float(entropy.detach()),
                            "gradient_norm": float(norm),
                        })
                    hidden = hidden.detach()
        finally:
            for environment in environments.values():
                environment.close()
        return self._result(records, update_records, captured, capture_frames, train)

    @staticmethod
    def _result(records, updates, captured, capture_frames, train):
        for index, record in enumerate(records):
            total = record["goals_total"]
            record["completed_fraction"] = record["goals_hit"] / total if total else 0.0
            if capture_frames:
                record["frames"] = (np.stack(captured[index]) if captured[index]
                                    else np.empty((0, 84, 84, 3), dtype=np.uint8))
        attempted = [record for record in records if not record["skipped_empty"]]
        summary = {
            "train": bool(train), "episodes": len(records), "attempted_episodes": len(attempted),
            "skipped_empty": len(records) - len(attempted), "successes": sum(record["success"] for record in attempted),
            "success_rate": sum(record["success"] for record in attempted) / len(attempted) if attempted else 0.0,
            "mean_completed_fraction": sum(record["completed_fraction"] for record in attempted) / len(attempted) if attempted else 0.0,
            "goals_total": sum(record["goals_total"] for record in records),
            "goals_hit": sum(record["goals_hit"] for record in records),
            "env_steps": sum(record["env_steps"] for record in records), "updates": len(updates),
        }
        return {"episodes": records, "updates": updates, "summary": summary}

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": 1, "kind": "goal_a2c", "policy_config": self.policy.model_config,
            "policy_state": self.policy.state_dict(), "optimizer_state": self.optimizer.state_dict(),
            "a2c_config": self.config, "updates": self.updates, "env_steps": self.env_steps,
            "training_env_steps": self.training_env_steps, "evaluation_env_steps": self.evaluation_env_steps,
            "metadata": {} if metadata is None else metadata,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, classifier, device="cpu"):
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint.get("kind") != "goal_a2c":
            raise ValueError("Unsupported goal-policy checkpoint format")
        policy = GoalPolicy(**checkpoint["policy_config"])
        policy.load_state_dict(checkpoint["policy_state"])
        agent = cls(policy, classifier, device=device, **checkpoint["a2c_config"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        for name in ("updates", "env_steps", "training_env_steps", "evaluation_env_steps"):
            setattr(agent, name, checkpoint[name])
        agent.checkpoint_metadata = checkpoint.get("metadata", {})
        return agent
