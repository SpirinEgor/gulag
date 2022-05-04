import logging
from typing import Type, Tuple

import gin
from torch import nn, Tensor


_logger = logging.getLogger(__name__)


@gin.configurable
class TokenClassifier(nn.Module):
    """Simple multi-layer perceptron to classify tokens based on their embeddings.
    Used as head after large embedding model, e.g. BERT.
    """

    def __init__(
        self,
        n_classes: int,
        embed_dim: int,
        hidden_dims: Tuple[int, ...] = (512,),
        activation_cls: Type[nn.Module] = nn.ReLU,
        dropout_rate: float = 0.0,
    ):
        """Gin configurable token classifier constructor.

        :param n_classes: number of classes to classify.
        :param embed_dim: dimension of the input embeddings.
        :param hidden_dims: Tuple of ints with the dimensions of the hidden layers, if empty then no hidden layers.
        :param activation_cls: PyTorch module for activation, by default `torch.nn.ReLU`.
        :param dropout_rate: dropout rate after each hidden layer.
        """
        super().__init__()

        if len(hidden_dims) == 0:
            _logger.warning(f"Classifier has no hidden layers, provide their sizes with 'hidden_dims' argument.")

        layers = []
        cur_size = embed_dim
        for next_size in hidden_dims:
            layers.extend([nn.Linear(cur_size, next_size), nn.Dropout(dropout_rate), activation_cls()])
            cur_size = next_size
        layers.append(nn.Linear(cur_size, n_classes))

        self._net = nn.Sequential(*layers)

    def forward(self, token_embeddings: Tensor) -> Tensor:
        """Create logits with num classes for each token embedding

        :param token_embeddings: [batch size; seq len; embed dim] -- embeddings of each token in all sequences
        :return: [batch size; seq len; n classes] -- logits for each token
        """
        return self._net(token_embeddings)
