import logging
from itertools import chain

import gin
import torch.optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import F1Score, MetricCollection
from transformers import BertModel

from src.model.token_classifier import TokenClassifier


_logger = logging.getLogger(__name__)


@gin.configurable
class MultiLanguageClassifier(LightningModule):
    def __init__(
        self, n_languages: int, embedder_name: str = "bert-base-multilingual-cased", *, freeze_embedder: bool = True
    ):
        super().__init__()

        self._token_embedder = BertModel.from_pretrained(embedder_name)
        self._token_classifier = TokenClassifier(n_languages, self._token_embedder.config.hidden_size)
        self._n_langs = n_languages

        if freeze_embedder:
            _logger.info(f"Freezing embedding model: {self._token_embedder.__class__.__name__}")
            for param in self._token_embedder.parameters():
                param.requires_grad = False

        self._metric = MetricCollection(
            {f"{split}_f1": F1Score(num_classes=n_languages) for split in ["train", "val", "test"]}
        )

    def forward(self, tokenized_texts: Tensor, attention_mask: Tensor) -> Tensor:  # type: ignore
        """Forward pass of multi-language classification model.
        Could be used during inference to classify each token in text.

        :param tokenized_texts: [batch size; seq len] -- batch with pretokenized texts
        :param attention_mask: [batch size; seq len] -- attention mask with 0 for padding tokens
        :return: [batch size; seq len] -- ids of predicted languages.
        """
        # [batch size; seq len; embed dim]
        token_embeddings = self._token_embedder(
            input_ids=tokenized_texts, attention_mask=attention_mask
        ).last_hidden_state
        # [batch size; seq len; embed dim]
        logits = self._token_classifier(token_embeddings)
        # [batch size; seq len]
        top_classes = logits.argmax(dim=-1)
        return top_classes

    @gin.configurable
    def configure_optimizers(self, lr: float = 0.001, weight_decay: float = 0.0):
        return torch.optim.AdamW(
            chain(self._token_embedder.parameters(), self._token_classifier.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    def calculate_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Calculate cross-entropy loss for predicted logits over non-padded tokens.
        Loss of the sequence is a mean of losses for non-padded tokens.
        Loss of the batch is a mean of losses for sequences.

        :param logits: [batch size; seq len; n classes] -- predicted logits
        :param labels: [batch size; seq len] -- true classes
        :return: [1] -- cross-entropy loss
        """
        logits = logits.permute(0, 2, 1)  # [batch size; n classes; seq len]
        loss = F.cross_entropy(logits, labels, reduction="none")  # [batch size; seq len]

        # label for padded tokens is `n languages`, `[0; n languages)` used for class ids.
        non_pad_mask = labels < self._n_langs  # [batch size; seq len]
        loss = loss * non_pad_mask  # [batch size; seq len]

        seq_lens = non_pad_mask.sum(-1)  # [batch size]
        sequence_loss = loss.sum(-1) / seq_lens  # [batch size]

        return sequence_loss.mean()

    def shared_step(self, batch: tuple[Tensor, ...], split: str) -> STEP_OUTPUT:
        input_ids, attention_mask, labels = batch

        embeddings = self._token_embedder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self._token_classifier(embeddings)
        loss = self.calculate_loss(logits, labels)

        with torch.no_grad():
            batch_f1 = self._get_f1_metric(split)(logits.permute(0, 2, 1), labels)

        if split == "train":
            self.log_dict({"train/step_loss": loss, "train/step_f1": batch_f1})
        return loss

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        del batch_idx
        return self.shared_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        del batch_idx
        return self.shared_step(batch, "val")

    def test_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        del batch_idx
        return self.shared_step(batch, "test")

    def shared_epoch_end(self, epoch_outputs: list[Tensor], split: str):
        mean_loss = torch.stack(epoch_outputs).mean()
        epoch_f1 = self._get_f1_metric(split).compute()
        self._get_f1_metric(split).reset()

        self.log_dict({f"{split}/loss": mean_loss, f"{split}/f1": epoch_f1})

    def training_epoch_end(self, epoch_outputs: list[Tensor]):  # type: ignore
        self.shared_epoch_end(epoch_outputs, "train")

    def validation_epoch_end(self, epoch_outputs: list[Tensor]):  # type: ignore
        self.shared_epoch_end(epoch_outputs, "val")

    def test_epoch_end(self, epoch_outputs: list[Tensor]):  # type: ignore
        self.shared_epoch_end(epoch_outputs, "test")

    def _get_f1_metric(self, split: str) -> F1Score:
        return self._metric[f"{split}_f1"]
