"""Small deterministic substitute for the pretrained model in CPU integration tests."""
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def stub_dino(monkeypatch):
    import torch

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'variants-supervised-agent'))
    import train_gru_classifier as classifier

    batches = []

    class TinyEncoder(torch.nn.Module):
        config = SimpleNamespace(model_type='dinov3_vit', patch_size=16, hidden_size=1024,
                                 num_hidden_layers=24, _commit_hash='test-revision')

        def forward(self, pixel_values):
            assert not self.training and not torch.is_grad_enabled()
            batches.append(pixel_values.detach().cpu().clone())
            vector = pixel_values.mean(dim=(1, 2, 3))[:, None].expand(-1, 1024)
            return SimpleNamespace(pooler_output=vector)

    monkeypatch.setattr(classifier, 'load_dino_backbone', lambda source, revision=None: TinyEncoder())
    return batches
