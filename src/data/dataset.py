import logging
import re
from collections import defaultdict
from string import punctuation, digits
from typing import Iterator, Tuple

import gin
from numpy import ndarray, concatenate
from numpy.random import default_rng
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset

_logger = logging.getLogger(__name__)


SAMPLE = Tuple[ndarray, ndarray]


class MultiLanguageClassificationDataset(IterableDataset):

    _digit_or_punctuation = re.compile(f"[{digits}{punctuation}«»]")

    def __init__(self, data: dict[str, list], tokenizer: Tokenizer, bos_id: int, eos_id: int, is_train: bool = True):
        self._langs = list(data.keys())
        self._n_langs = len(self._langs)
        _logger.info(f"Initializing dataset with {', '.join(self._langs)} languages")

        self._data = defaultdict(lambda: [])
        for lang, examples in data.items():
            _logger.info(f"Processing {len(examples)} examples from {lang} lang...")
            for i, full_example in enumerate(examples):
                sentence = " ".join(full_example["tokens"])
                sentence = re.sub(self._digit_or_punctuation, "", sentence)
                tokens = tokenizer.encode(sentence, add_special_tokens=False)
                self._data[lang].append(tokens)

        self._bos_id = bos_id
        self._eos_id = eos_id

        self._rng = default_rng()

        self._is_train = is_train
        if not is_train:
            self._samples = self.generate_eval_samples()

    def __iter__(self) -> Iterator[SAMPLE]:
        if self._is_train:
            return self
        else:
            return iter(self._samples)

    @gin.configurable
    def generate_example(
        self, min_langs: int = 1, max_langs: int = 5, max_samples_per_lang: int = 5, max_seq_len: int = 512
    ) -> SAMPLE:
        n_langs = int(self._rng.uniform(min_langs, max_langs + 1))
        langs = self._rng.choice(self._langs, size=n_langs, replace=False)

        selected_samples, selected_langs = [], []
        for lang in langs:
            n_samples = int(self._rng.uniform(1, max_samples_per_lang + 1))
            cur_samples = self._rng.choice(len(self._data[lang]), size=n_samples, replace=False)
            lang_id = self._langs.index(lang)

            selected_samples += [self._data[lang][i] for i in cur_samples]
            selected_langs += [[lang_id] * len(self._data[lang][i]) for i in cur_samples]

        random_permutation = self._rng.permutation(len(selected_samples))
        input_seq = concatenate([[self._bos_id]] + [selected_samples[i] for i in random_permutation] + [[self._eos_id]])
        target = concatenate([[self._n_langs]] + [selected_langs[i] for i in random_permutation] + [[self._n_langs]])

        return input_seq[:max_seq_len], target[:max_seq_len]

    def __next__(self) -> SAMPLE:
        return self.generate_example()

    @gin.configurable
    def generate_eval_samples(self, n_samples: int = 10_000) -> list[SAMPLE]:
        _logger.info(f"Generating eval holdout with {n_samples} samples.")
        return [self.generate_example() for _ in range(n_samples)]
