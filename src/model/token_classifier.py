import logging
from typing import Type

import gin
from torch import nn, Tensor


_logger = logging.getLogger(__name__)


@gin.configurable
class TokenClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        embed_dim: int,
        hidden_dims: tuple[int] = (512,),
        activation_cls: Type[nn.Module] = nn.ReLU,
        dropout_rate: float = 0.0,
    ):
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
        print(token_embeddings.shape)
        return self._net(token_embeddings)
