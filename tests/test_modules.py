import torch

from src.model.token_classifier import TokenClassifier


def test_smoke_token_classifier():
    batch_size, seq_len, num_classes = 5, 7, 10
    embed_dim, hidden_dims = 64, (32, 32)

    token_classifier = TokenClassifier(num_classes, embed_dim, hidden_dims=hidden_dims)
    rand_inputs = torch.rand(batch_size, seq_len, embed_dim)

    try:
        token_classifier(rand_inputs)
    except RuntimeError as e:
        assert False, f"Smoke forward pass for token classifier failed:\n{e}"
