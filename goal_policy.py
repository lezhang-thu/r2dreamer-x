"""Recurrent goal policy with attention over a variable-length milestone memory.

These visual weights belong to the policy. They are never copied from, shared
with, or optimized through the milestone classifier.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.checkpoint import checkpoint

from milestone_model import make_encoder


def _anchor_batch_to_tensor(frames, device):
    if frames.dtype == torch.uint8:
        return frames.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous().div_(255.0)
    return frames.to(device=device, dtype=torch.float32)


class GoalPolicy(nn.Module):
    """An anchor GRU, a recurrent controller, and cross-attention at every step."""

    def __init__(self, num_actions, hidden_dim=128, attention_heads=4, mask_completed=True):
        super().__init__()
        if num_actions < 1 or hidden_dim < 1 or attention_heads < 1:
            raise ValueError("Action count, hidden dimension and attention heads must be positive")
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)
        self.mask_completed = bool(mask_completed)
        self.model_config = {
            "num_actions": self.num_actions, "hidden_dim": self.hidden_dim,
            "attention_heads": int(attention_heads), "mask_completed": self.mask_completed,
        }
        self.vision = make_encoder("small", hidden_dim)
        self.anchor_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.controller = nn.GRUCell(2 * hidden_dim + 2, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(3 * hidden_dim + 2, hidden_dim), nn.Tanh())
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch_size, device=None):
        return self.actor.weight.new_zeros((batch_size, self.hidden_dim), device=device)

    def encode_anchors(self, anchors, lengths):
        """Encode padded RGB sequences with a packed GRU.

        Accepts float [B,L,3,84,84] or uint8 [B,L,84,84,3]; uint8 inputs can
        stay on CPU. Long sequences use 128-frame visual chunks and activation
        checkpointing, so full-resolution CNN activations do not grow with L.
        Zero-length rows have a zero dummy memory. Padding does not enter either
        the visual encoder or the GRU, and episode order need not be sorted.
        """
        image_shape = (84, 84, 3) if anchors.dtype == torch.uint8 else (3, 84, 84)
        if anchors.ndim != 5 or tuple(anchors.shape[2:]) != image_shape:
            raise ValueError("anchors must be float [B,L,3,84,84] or uint8 [B,L,84,84,3]")
        batch, length = anchors.shape[:2]
        if lengths.shape != (batch,) or torch.any(lengths < 0) or torch.any(lengths > length):
            raise ValueError("lengths must give a valid sequence length for every batch row")
        device = self.actor.weight.device
        lengths = lengths.to(device=device, dtype=torch.long)
        padded_length = max(1, length)
        memory = self.actor.weight.new_zeros((batch, padded_length, self.hidden_dim))
        nonempty = lengths > 0
        if not nonempty.any():
            return memory
        valid = torch.arange(length, device=anchors.device)[None, :] < lengths.to(anchors.device)[:, None]
        indices = valid.flatten().nonzero(as_tuple=False).flatten()
        flattened = anchors.reshape(batch * length, *image_shape)

        def encode_chunk(raw_frames):
            return self.vision(_anchor_batch_to_tensor(raw_frames, device))

        chunks = []
        for selected in indices.split(128):
            raw = flattened[selected]
            if torch.is_grad_enabled() and len(indices) > 128:
                chunks.append(checkpoint(encode_chunk, raw, use_reentrant=False))
            else:
                chunks.append(encode_chunk(raw))
        features = self.actor.weight.new_zeros((batch * length, self.hidden_dim))
        features = features.index_copy(0, indices.to(device), torch.cat(chunks))
        features = features.reshape(batch, length, self.hidden_dim)
        packed = pack_padded_sequence(
            features[nonempty], lengths[nonempty].cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.anchor_gru(packed)
        padded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=length)
        memory[nonempty] = padded
        return memory

    def forward(self, observations, memory, lengths, progress, hidden):
        """Return action logits, value, next hidden state, and attention weights.

        ``progress[b]`` counts achieved anchors; the current goal is at that
        index. Completed-anchor masking is independent of the mandatory padding
        mask. Empty and completed rows produce finite outputs and zero attention
        when no real anchor is available.
        """
        batch = observations.shape[0]
        if memory.ndim != 3 or memory.shape[0] != batch or memory.shape[2] != self.hidden_dim:
            raise ValueError("memory must have shape [B,L,hidden_dim]")
        if memory.shape[1] == 0:
            memory = memory.new_zeros((batch, 1, self.hidden_dim))
        lengths = lengths.to(device=memory.device, dtype=torch.long)
        progress = progress.to(device=memory.device, dtype=torch.long)
        if lengths.shape != (batch,) or progress.shape != (batch,):
            raise ValueError("lengths and progress must have shape [B]")
        if torch.any(lengths < 0) or torch.any(lengths > memory.shape[1]):
            raise ValueError("lengths exceed the available memory")
        if torch.any(progress < 0) or torch.any(progress > lengths):
            raise ValueError("progress must be between zero and sequence length")
        indices = progress.clamp(max=memory.shape[1] - 1)
        current_goal = memory[torch.arange(batch, device=memory.device), indices]
        current_goal = current_goal * (progress < lengths).unsqueeze(-1)
        denominator = lengths.clamp_min(1).to(memory.dtype)
        progress_features = torch.stack((progress / denominator, (lengths - progress) / denominator), -1)
        visual = self.vision(observations)
        next_hidden = self.controller(torch.cat((visual, current_goal, progress_features), -1), hidden)

        positions = torch.arange(memory.shape[1], device=memory.device)[None, :]
        allowed = positions < lengths[:, None]
        if self.mask_completed:
            allowed = allowed & (positions >= progress[:, None])
        # MultiheadAttention cannot softmax an entirely masked row. Give that
        # row one zero dummy key, then erase its context and attention afterward.
        available = allowed.any(-1)
        safe_allowed = allowed.clone()
        safe_allowed[~available, 0] = True
        safe_memory = memory * available[:, None, None]
        context, weights = self.attention(
            next_hidden[:, None, :], safe_memory, safe_memory,
            key_padding_mask=~safe_allowed, need_weights=True,
        )
        context = context[:, 0] * available[:, None]
        weights = weights[:, 0] * allowed
        fused = self.fusion(torch.cat((next_hidden, context, current_goal, progress_features), -1))
        return {
            "logits": self.actor(fused), "value": self.critic(fused).squeeze(-1),
            "hidden": next_hidden, "attention": weights,
        }
