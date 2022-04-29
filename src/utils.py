import logging as py_logging

from datasets.utils import logging as ds_logging
from transformers.utils import logging as tr_logging


def setup_logging():
    py_logging.basicConfig(level=py_logging.INFO)
    tr_logging.set_verbosity_info()
    tr_logging.disable_progress_bar()
    ds_logging.set_verbosity_info()
    ds_logging.disable_progress_bar()
