import logging as py_logging
from math import sqrt

import gin
from datasets.utils import logging as ds_logging
from transformers.utils import logging as tr_logging


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
