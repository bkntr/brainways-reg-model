# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Computer vision example on Transfer Learning.
This computer vision example illustrates how one could fine-tune a pre-trained
network (by default, a ResNet50 is used) using pytorch-lightning. For the sake
of this example, the 'cats and dogs dataset' (~60MB, see `DATA_URL` below) and
the proposed network (denoted by `TransferLearningModel`, see below) is
trained for 15 epochs.

The training consists of three stages.

From epoch 0 to 4, the feature extractor (the pre-trained network) is frozen except
maybe for the BatchNorm layers (depending on whether `train_bn = True`). The BatchNorm
layers (if `train_bn = True`) and the parameters of the classifier are trained as a
single parameters group with lr = 1e-2.

From epoch 5 to 9, the last two layer groups of the pre-trained network are unfrozen
and added to the optimizer as a new parameter group with lr = 1e-4 (while lr = 1e-3
for the first parameter group in the optimizer).

Eventually, from epoch 10, all the remaining layer groups of the pre-trained network
are unfrozen and added to the optimizer as a third parameter group. From epoch 10,
the parameters of the pre-trained network are trained with lr = 1e-5 while those of
the classifier is trained with lr = 1e-4.

Note:
    See: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

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
    parser.add_argument("--config", default="reg")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = load_config("reg")
    pl.seed_everything(config.seed, workers=True)

    # init model
    model = BrainwaysRegModel(config)

    # init data
    datamodule = BrainwaysDataModule(
        data_paths={
            "train": "data/synth.zip",
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

    output_dir = Path("outputs/reg/synth")

    # Initialize a trainer
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        callbacks=[finetuning_callback, checkpoint_callback],
        accelerator="auto",
        max_epochs=config.opt.max_epochs,
        num_sanity_val_steps=0,
    )

    # Train the model ⚡
    trainer.fit(model, datamodule=datamodule)

    shutil.move(checkpoint_callback.best_model_path, output_dir / "model.ckpt")
    shutil.copytree(trainer.log_dir, output_dir / "logs")


if __name__ == "__main__":
    cli_main()
