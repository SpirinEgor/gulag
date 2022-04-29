import logging
import re
from collections import defaultdict
from random import randint, sample
from string import punctuation, digits
from typing import Iterator, Tuple

import gin
from numpy import ndarray, concatenate
from numpy.random import permutation
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm, trange

_logger = logging.getLogger(__name__)


SAMPLE = Tuple[ndarray, ndarray]


class MultiLanguageClassificationDataset(IterableDataset):

    _digit_or_punctuation = re.compile(f"[{digits}{punctuation}«»]")

    def __init__(self, data: dict[str, list], tokenizer: Tokenizer, bos_id: int, eos_id: int, is_train: bool = True):
        self._langs = list(data.keys())
        self._n_langs = len(self._langs)
        _logger.info(f"Using dataset with {', '.join(self._langs)} languages")

        self._data = defaultdict(lambda: [])
        with tqdm(data.items(), desc="Dataset preparation") as pbar:
            for lang, examples in pbar:
                for i, full_example in enumerate(examples):
                    sentence = " ".join(full_example["tokens"])
                    sentence = re.sub(self._digit_or_punctuation, "", sentence)
                    tokens = tokenizer.encode(sentence, add_special_tokens=False)
                    self._data[lang].append(tokens)

                    if (i + 1) % 1_000 == 0 or i == len(examples) - 1:
                        pbar.set_postfix({lang: f"{i + 1}/{len(examples)}"})

        self._bos_id = bos_id
        self._eos_id = eos_id

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
        n_langs = randint(min_langs, max_langs)
        langs = sample(self._langs, k=n_langs)

        selected_samples, selected_langs = [], []
        for lang in langs:
            n_samples = randint(1, max_samples_per_lang)
            cur_samples = sample(self._data[lang], k=n_samples)

            lang_id = self._langs.index(lang)
            lang_seq = [[lang_id] * len(s) for s in cur_samples]

            selected_samples += cur_samples
            selected_langs += lang_seq

        random_permutation = permutation(len(selected_samples))
        input_seq = concatenate([[self._bos_id]] + [selected_samples[i] for i in random_permutation] + [[self._eos_id]])
        target = concatenate([[self._n_langs]] + [selected_langs[i] for i in random_permutation] + [[self._n_langs]])

        return input_seq[:max_seq_len], target[:max_seq_len]

    def __next__(self) -> SAMPLE:
        return self.generate_example()

    @gin.configurable
    def generate_eval_samples(self, n_samples: int = 10_000) -> list[SAMPLE]:
        _logger.info(f"Generating eval holdout with {n_samples} samples.")
        return [self.generate_example() for _ in trange(n_samples, desc="Generating eval samples")]
