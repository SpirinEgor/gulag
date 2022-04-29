import logging
from typing import Optional

import gin
import torch
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.data.dataset import MultiLanguageClassificationDataset, SAMPLE


_logger = logging.getLogger(__name__)


@gin.configurable
class MultiLanguageClassificationDataModule(LightningDataModule):
    def __init__(self, languages: tuple[str] = gin.REQUIRED, batch_size: int = 128, val_batch_size: int = 256):
        super().__init__()
        self._languages = languages
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size

        # Init by setup
        self._datasets: dict[str, MultiLanguageClassificationDataset] = {}
        self._tokenizer: Tokenizer = None
        self._bos_id: int = None
        self._eos_id: int = None

    def setup(self, stage: Optional[str] = None):
        _logger.info(f"Downloading and opening 'wikiann' dataset for {', '.join(self._languages)}")
        full_data = {code: load_dataset("wikiann", code) for code in self._languages}

        _logger.info(f"Downloading and opening 'bert-base-multilingual-cased' tokenizer")
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self._bos_id = self._tokenizer.cls_token_id
        self._eos_id = self._tokenizer.sep_token_id

        for split in ["train", "validation", "test"]:
            _logger.info(f"Initializing {split} dataset")
            data = {code: full_data[code][split] for code in self._languages}
            self._datasets[split] = MultiLanguageClassificationDataset(
                data, self._tokenizer, self._bos_id, self._eos_id, is_train=(split == "train")
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self._datasets["train"], batch_size=self._batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self._datasets["validation"], batch_size=self._val_batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self._datasets["test"], batch_size=self._test_batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, samples: list[SAMPLE]) -> tuple[torch.Tensor, ...]:
        max_len = max(len(x[0]) for x in samples)
        batch_size = len(samples)

        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), len(self._languages), dtype=torch.long)

        for i, (input_seq, label) in enumerate(samples):
            c_len = len(label)
            input_ids[i, :c_len] = torch.tensor(input_seq)
            attention_mask[i, :c_len] = 1
            labels[i, :c_len] = torch.tensor(label)

        return input_ids, attention_mask, labels

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def decode_languages(self, languages: torch.Tensor):
        return [(self._languages[i] if i < len(self._languages) else "[NOT LANG]") for i in languages]
