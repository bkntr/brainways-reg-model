import argparse
import logging
import shutil
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from duracell.dataset import BrainwaysDataModule
from duracell.models.reg.model import BrainwaysRegModel
from duracell.utils.config import load_config
from duracell.utils.training.milestones_finetuning import MilestonesFinetuning

log = logging.getLogger(__name__)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=("reg", "finetune"))
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    pl.seed_everything(config.seed, workers=True)

    # init model
    model = BrainwaysRegModel.load_from_checkpoint(
        "outputs/reg/synth/model.ckpt", config=config
    )

    # init data
    datamodule = BrainwaysDataModule(
        data_paths={
            "train": "data/real.zip",
            "val": "data/real.zip",
            "test": "data/real.zip",
        },
        data_config=config.data,
        num_workers=args.num_workers,
        transform=model.transform,
        target_transform=model.target_transform,
    )

    finetuning_callback = MilestonesFinetuning(
        milestones=config.opt.milestones, train_bn=config.opt.train_bn
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=config.opt.monitor.metric, mode=config.opt.monitor.mode
    )

    output_dir = Path("outputs/reg/real")

    # Initialize a trainer
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        callbacks=[finetuning_callback, checkpoint_callback],
        accelerator="auto",
        max_epochs=config.opt.max_epochs,
        num_sanity_val_steps=0,
    )

    # Train the model ⚡a
    trainer.fit(model, datamodule=datamodule)

    shutil.move(checkpoint_callback.best_model_path, output_dir / "model.ckpt")
    shutil.copytree(trainer.log_dir, output_dir / "logs")


if __name__ == "__main__":
    cli_main()
