import torch

from src.model.token_classifier import TokenClassifier


def test_token_classifier():
    embed_dim = 64

    token_classifier = TokenClassifier(10, embed_dim, hidden_dims=(32,))
    rand_inputs = torch.rand(5, 7, embed_dim)  # rand input with batch size 5 and seq len 7

    try:
        token_classifier(rand_inputs)
    except RuntimeError as e:
        assert False, f"Smoke forward pass failed:\n{e}"
