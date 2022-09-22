import logging
import shutil
from pathlib import Path

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from brainways_reg_model.model.dataset import BrainwaysDataModule
from brainways_reg_model.model.model import BrainwaysRegModel
from brainways_reg_model.utils.config import load_config
from brainways_reg_model.utils.milestones_finetuning import MilestonesFinetuning
from brainways_reg_model.utils.paths import (
    REAL_DATA_ZIP_PATH,
    REAL_TRAINED_MODEL_ROOT,
    SYNTH_TRAINED_MODEL_ROOT,
)

log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config",
    "config_name",
    default="finetune",
    help="Config section name.",
    show_default=True,
)
@click.option(
    "--output",
    default=REAL_TRAINED_MODEL_ROOT,
    type=Path,
    help="Model output path.",
    show_default=True,
)
@click.option(
    "--synth-model",
    default=SYNTH_TRAINED_MODEL_ROOT,
    type=Path,
    help="Trained synth model path.",
    show_default=True,
)
@click.option("--num-workers", default=32, help="Number of data workers.")
def finetune(config_name: str, output: Path, synth_model: Path, num_workers: int):
    config = load_config(config_name)
    pl.seed_everything(config.seed, workers=True)

    # init model
    model = BrainwaysRegModel.load_from_checkpoint(
        str(synth_model / "model.ckpt"), config=config
    )

    # init data
    datamodule = BrainwaysDataModule(
        data_paths={
            "train": REAL_DATA_ZIP_PATH,
            "val": REAL_DATA_ZIP_PATH,
            "test": REAL_DATA_ZIP_PATH,
        },
        data_config=config.data,
        num_workers=num_workers,
        transform=model.transform,
        target_transform=model.target_transform,
        augmentation=model.augmentation,
    )

    finetuning_callback = MilestonesFinetuning(
        milestones=config.opt.milestones, train_bn=config.opt.train_bn
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=config.opt.monitor.metric, mode=config.opt.monitor.mode
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        default_root_dir=str(output),
        callbacks=[finetuning_callback, checkpoint_callback],
        accelerator="auto",
        max_epochs=config.opt.max_epochs,
        num_sanity_val_steps=0,
    )

    # Train the model ⚡a
    trainer.fit(model, datamodule=datamodule)

    output_checkpoint_path = output / "model.ckpt"
    output_logs_path = output / "logs"
    output_checkpoint_path.unlink(missing_ok=True)
    shutil.rmtree(output_logs_path)

    shutil.move(checkpoint_callback.best_model_path, output_checkpoint_path)
    shutil.copytree(trainer.log_dir, output_logs_path)
