import logging
from argparse import ArgumentParser
from os.path import join

import gin
import wandb
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.data import MultiLanguageClassificationDataModule
from src.model import MultiLanguageClassifier
from src.utils import setup_logging, get_infer_fn

_logger = logging.getLogger(__name__)


def configure_arg_parser() -> ArgumentParser:
    argument_parser = ArgumentParser(description="CLI for GULAG: GUess LAnGuages with neural networks.")
    subparsers = argument_parser.add_subparsers(dest="mode", help="Sub-commands to select model training or inference.")

    parser_train = subparsers.add_parser("train", help="Run model training.")
    parser_train.add_argument("--gin-file", action="append", help="Gin file configuration to parse")
    parser_train.add_argument("--gin-param", action="append", help="Additional gin param to bind")

    parser_infer = subparsers.add_parser("infer", help="Run model inference.")
    parser_infer.add_argument("--wandb", type=str, default="voudy/gulag/a55dbee8", help="W&B run path")
    parser_infer.add_argument("--ckpt", type=str, default="step_20000.ckpt", help="Checkpoint name")

    return argument_parser


@gin.configurable
def train(
    n_steps: int = gin.REQUIRED,
    accelerator: str = gin.REQUIRED,
    eval_steps: int = gin.REQUIRED,
    gradient_clip: float = 0.0,
    log_steps: int = 50,
    seed: int = 7,
    wandb_project_name: str = None,
):
    """Main function to run model training.

    :param n_steps: number of training steps.
    :param accelerator: name of accelerator, e.g. "gpu".
    :param eval_steps: period of evaluation.
    :param gradient_clip: gradient clipping value, 0.0 means no gradient clipping.
    :param log_steps: period of logging step.
    :param seed: random seed.
    :param wandb_project_name: name of W&B project to log progress.
    """
    seed_everything(seed)
    setup_logging()

    wandb_logger = WandbLogger(project=wandb_project_name)
    checkpoint_callback = ModelCheckpoint(
        wandb_logger.experiment.dir,
        filename="step_{step}",
        every_n_train_steps=eval_steps,
        save_top_k=-1,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
    )
    lr_logger = LearningRateMonitor("step")

    data_module = MultiLanguageClassificationDataModule()
    model = MultiLanguageClassifier(data_module.n_languages)

    trainer = Trainer(
        accelerator=accelerator,
        callbacks=[lr_logger, checkpoint_callback],
        gradient_clip_val=gradient_clip,
        log_every_n_steps=log_steps,
        logger=wandb_logger,
        max_steps=n_steps,
        val_check_interval=eval_steps,
    )

    with open(join(wandb_logger.experiment.dir, "config.gin"), "w") as f:
        f.write(gin.config_str())

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)

    wandb.finish()


def infer(wandb_run_path: str, ckpt_name: str):
    """Function to run model inference via CLI.

    :param wandb_run_path: W&B run path where model is saved.
    :param ckpt_name: Name of checkpoint in specified run.
    """
    setup_logging()
    infer_fn = get_infer_fn(wandb_run_path, ckpt_name)

    while True:
        text = input("Enter text to classify languages (Ctrl-C to exit):\n")
        prediction = infer_fn(text)
        print(prediction)


if __name__ == "__main__":
    _arg_parser = configure_arg_parser()
    _args = _arg_parser.parse_args()

    if _args.mode == "train":
        gin.parse_config_files_and_bindings(_args.gin_file, _args.gin_param)
        train()
    else:
        infer(_args.wandb, _args.ckpt)
