import logging
import logging as py_logging
from dataclasses import dataclass, field
from math import sqrt
from os.path import join, exists
from typing import Tuple, Callable, List

import gin
import torch
from datasets.utils import logging as ds_logging
from tokenizers import Tokenizer
from transformers import BertTokenizer, AutoTokenizer
from transformers.utils import logging as tr_logging
from wandb import Api

from src.data import MultiLanguageClassificationDataset
from src.model import MultiLanguageClassifier


CHECKPOINT_DIR = "checkpoints"

_logger = logging.getLogger(__name__)


def setup_logging():
    py_logging.basicConfig(level=py_logging.INFO)
    tr_logging.set_verbosity_info()
    tr_logging.disable_progress_bar()
    ds_logging.set_verbosity_info()
    ds_logging.disable_progress_bar()


@gin.configurable
def rsqrt_with_warmup(step: int, warmup_steps: int = gin.REQUIRED) -> float:
    """Define a scheduler for learning rate with a form of reverse sqrt with respect to the current step.
    Warmup points to constant learning rate for first 'n' steps.
    Implemented as multiplier for initial learning rate, i.e. to use with `torch.optim.LambdaLR`.

    For example, if `warmup_steps = 10_000`, then:
    – First 10_000 steps use initial learning rate;
    – After that multiply learning rate by `sqrt(warmup_steps) / sqrt(cur_step)`;

    :param step: current step
    :param warmup_steps: number of warmup steps
    :return: multiplier for the next learning rate
    """
    return sqrt(warmup_steps) / sqrt(max(step, warmup_steps))


@dataclass
class MultiLanguageText:
    text_spans: List[str] = field(default_factory=list)
    text_langs: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        parts = []
        for lang, span in zip(self.text_langs, self.text_spans):
            parts.append(f"{lang} -- {span}")
        return "\n".join(parts)

    def unique_languages(self) -> List[str]:
        return list(set(self.text_langs))


def get_infer_fn(wandb_run_path: str, ckpt_name: str) -> Callable[[str], MultiLanguageText]:
    api = Api()
    run = api.run(wandb_run_path)

    output_dir = join(CHECKPOINT_DIR, wandb_run_path.replace("/", "_"))
    ckpt_path = join(output_dir, ckpt_name)
    config_path = join(output_dir, "config.gin")
    if not exists(ckpt_path):
        ckpt_run_file = run.file(ckpt_name)
        ckpt_size_mb = round(ckpt_run_file.size / 1024 / 1024, 2)
        _logger.info(f"Downloading {ckpt_name} with its config into {output_dir} dir, size: {ckpt_size_mb} MB")
        ckpt_run_file.download(output_dir)
        run.file("config.gin").download(output_dir, replace=True)
    else:
        _logger.info(f"Checkpoint already downloaded, use it")

    gin.parse_config_file(config_path, skip_unknown=True)
    data_module_params = gin.get_bindings("MultiLanguageClassificationDataModule")

    languages = data_module_params["languages"]
    _logger.info(f"Provided checkpoint classify following languages: {', '.join(languages)}")

    tokenizer = AutoTokenizer.from_pretrained(data_module_params["tokenizer_name"])

    model = MultiLanguageClassifier(len(languages))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def infer_fn(input_text: str) -> MultiLanguageText:
        clean_text = MultiLanguageClassificationDataset.prepare_text(input_text)
        tokens = tokenizer.encode(clean_text)
        with torch.no_grad():
            b_tokens = torch.tensor(tokens, dtype=torch.long).reshape(1, -1)
            b_attn = torch.ones_like(b_tokens)
            predictions = model(b_tokens, b_attn)

        # Keep only current string without CLS and SEP tokens
        predictions = predictions[0, 1:-1]
        tokens = tokens[1:-1]

        start_pos, cur_lang = 0, predictions[0].item()
        spans, langs = [], []
        for pos, lang in enumerate(predictions[1:]):
            pos += 1
            lang = lang.item()
            if lang != cur_lang:
                spans.append(tokenizer.decode(tokens[start_pos:pos]))
                langs.append(languages[cur_lang])
                start_pos, cur_lang = pos, lang
        spans.append(tokenizer.decode(tokens[start_pos:]))
        langs.append(languages[cur_lang])
        return MultiLanguageText(spans, langs)

    return infer_fn
