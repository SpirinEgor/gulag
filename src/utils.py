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
def lr_scheduler_rsqrt_with_warmup(step: int, warmup_steps: int = gin.REQUIRED, multiplier: float = 1.0) -> float:
    """Define the learning rate as reverse sqrt of the current step.
    Warmup points to constant learning rate for first 'n' steps.

    For example, if `warmup_steps = 1e4` and `multiplier = 2.0`, then:
    - First 10_000 steps use (2.0 / sqrt(1e4) = 2.0 * 1e-2 = 0.02) learning rate;
    - After that use (2.0 / sqrt(cur_step)) learning rate;

    :param step: current step
    :param warmup_steps: number of warmup steps
    :param multiplier: multiplier to adjust learning rate
    :return: next learning rate
    """
    return multiplier / sqrt(max(step, warmup_steps))
