"""Check that packed pair sampling preserves causal contexts and gradients."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'variants-supervised-agent'))
from train_gru_classifier import FramePairGRU, PairSampler, sampled_embeddings


@pytest.mark.parametrize('chunk_size', [0, 3])
def test_packed_contexts_match_episode_unrolls(chunk_size):
    torch.manual_seed(321)
    model = FramePairGRU(nn.Identity(), 7, 3, 5, 2)
    sequences = [torch.randn(n, 7) for n in [2, 8, 21, 8]]
    sampler = PairSampler(sequences)
    # Repeated positions, tied lengths and an endpoint in a partial last chunk.
    indices = [38, 0, 2, 8, 16, 8, 1, 27, 33]
    packed = sampled_embeddings(model, sequences, sampler, indices, 'cpu', chunk_size)
    params = list(model.gru.parameters()) + list(model.projection.parameters())
    gradients = torch.autograd.grad(packed.square().sum(), params)

    episodes, positions = sampler.locate(np.asarray(indices))
    reference = []
    for episode, t in zip(episodes, positions):
        sequence = sequences[episode][:t + 1].unsqueeze(0)
        width = chunk_size or sequence.shape[1]
        hidden = None
        for chunk in sequence.split(width, dim=1):
            contexts, hidden = model.encode_features(chunk, hidden)
            hidden = hidden.detach()
        reference.append(contexts[0, -1])
    reference = torch.stack(reference)
    torch.testing.assert_close(packed, reference)
    for grad, expected in zip(gradients, torch.autograd.grad(reference.square().sum(), params)):
        torch.testing.assert_close(grad, expected, atol=2e-6, rtol=2e-5)
