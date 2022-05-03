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
from src.utils import setup_logging

_logger = logging.getLogger(__name__)


def configure_arg_parser() -> ArgumentParser:
    argument_parser = ArgumentParser(description="Train or infer model to guess the languages in the text.")
    argument_parser.add_argument("mode", choices=["train", "infer"], help="Mode to run model")
    argument_parser.add_argument("--gin-file", action="append", help="Gin file to parse")
    argument_parser.add_argument("--gin-param", action="append", help="Gin param to bind")

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


def infer():
    pass


if __name__ == "__main__":
    _arg_parser = configure_arg_parser()
    _args = _arg_parser.parse_args()

    gin.parse_config_files_and_bindings(_args.gin_file, _args.gin_param)

    if _args.mode == "train":
        train()
    else:
        infer()
