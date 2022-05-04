import logging
import re
from collections import defaultdict
from string import punctuation, digits
from typing import Iterator, Tuple, Dict, List

import gin
from numpy import ndarray, concatenate
from numpy.random import default_rng
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset, get_worker_info

_logger = logging.getLogger(__name__)


SAMPLE = Tuple[ndarray, ndarray]


class MultiLanguageClassificationDataset(IterableDataset):
    """Dataset for multi-language classification task.
    It uses sentences from different languages to produce multi-language synthetic data.

    As train dataset:
        class provides infinite number of samples by generating new one for each request (see `generate_example`).

    As evaluation dataset:
        class generates dataset on init for consistent evaluation during training.
    """

    _digit_or_punctuation = re.compile(f"[{digits}{punctuation}«»]")

    @classmethod
    def prepare_text(cls, text: str) -> str:
        """Public method that used to prepare raw text. For now, remove all punctuations and digits from the text."""
        return re.sub(cls._digit_or_punctuation, "", text)

    def __init__(self, data: Dict[str, List], tokenizer: Tokenizer, bos_id: int, eos_id: int, is_train: bool = True):
        """Dataset constructor.

        :param data: Dictionary with language code as keys and list of sentences on this language as values.
        :param tokenizer: Tokenizer that used to tokenize text.
        :param bos_id: Number to mark the beginning of each sample.
        :param eos_id: Number to mark the end of each sample.
        :param is_train: If `True` then generate new example on each request, otherwise pre-generate all data once on init.
        """
        self._langs = list(data.keys())
        self._n_langs = len(self._langs)
        _logger.info(f"Initializing dataset with {', '.join(self._langs)} languages")

        self._data = defaultdict(list)
        for lang, examples in data.items():
            _logger.info(f"Processing {len(examples)} examples from {lang} lang...")
            for i, full_example in enumerate(examples):
                text = " ".join(full_example["tokens"])
                text = self.prepare_text(text)
                tokens = tokenizer.encode(text, add_special_tokens=False)
                self._data[lang].append(tokens)

        self._bos_id = bos_id
        self._eos_id = eos_id

        self._rng = default_rng()

        self._is_train = is_train
        if not is_train:
            self._samples = self.generate_eval_samples()

    def __iter__(self) -> Iterator[SAMPLE]:
        # On training each sample is unique => no care about workers
        if self._is_train:
            return self

        worker_info = get_worker_info()
        # single-process data loading, return the full iterator
        # eval data already preprocessed, no need to parallelize
        if worker_info is None or worker_info.id == 0:
            return iter(self._samples)
        return iter([])

    @gin.configurable
    def generate_example(
        self, min_langs: int = 1, max_langs: int = 5, max_samples_per_lang: int = 5, max_seq_len: int = 512
    ) -> SAMPLE:
        """Main method to generate examples, use gin to configure the generation details.
        The generation strategy:
        – select number of languages in next example (min_langs <= n <= max_langs)
        – for each language select number of sentences in next example (1 <= x_i <= max_samples_per_lang)
        – randomly select `x_i` sentences for each language
        – shuffle all sentences across all languages and concatenate them
            (total x_1 + x_2 + ... + x_n sentences. per example)
        – trim sentence by max_seq_len

        :param min_langs: minimum number of languages per example
        :param max_langs: maximum number of languages per example
        :param max_samples_per_lang: maximum number of sentences on each language per example
        :param max_seq_len: maximum length of sequence
        :return: Tuple of token ids of sampled example and sequence of corresponding language ids
        """
        max_langs = min(max_langs, self._n_langs)
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
    def generate_eval_samples(self, n_samples: int = 10_000) -> List[SAMPLE]:
        """Gin configurable method to generate evaluation dataset.

        :param n_samples: number of sample in evaluation dataset
        :return: List of samples that used during evaluation
        """
        _logger.info(f"Generating eval holdout with {n_samples} samples.")
        return [self.generate_example() for _ in range(n_samples)]
